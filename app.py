import os
import json
import streamlit as st
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
import voyageai
from neo4j import GraphDatabase

# -------------------------------------------
# Title and Introduction
st.title("RFP Response Generator")
st.markdown("""
    **Transforming RFP Responses with AI**
    
    This demo showcases an end-to-end solution that processes past proposal documents, enriches them with metadata via knowledge graph (in Neo4j), and uses a Retrieval Augmented Generation (RAG) pipeline to generate draft responses (and clarifying questions) for new RFP questions.
    """)

# -------------------------------------------
# Sidebar for Configuration
st.sidebar.header("Configuration")

# API keys
default_voyage_key = st.secrets["api_keys"]["VOYAGE_API_KEY"]
default_groq_key = st.secrets["api_keys"]("GROQ_API_KEY")

# Sidebar API key inputs with default values (masked)
voyage_api_key_sidebar = st.sidebar.text_input(
    "Voyage AI API Key", 
    type="password",
    value=default_voyage_key,
    help="Enter your Voyage AI API key."
)
groq_api_key_sidebar = st.sidebar.text_input(
    "Groq AI API Key", 
    type="password",
    value=default_groq_key,
    help="Enter your Groq AI API key."
)

# Model Selection
model_options = [
    "llama3-8b-8192", 
    "llama3-70b-8192", 
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-text-preview",
    "llama-3.2-90b-text-preview"
]
selected_model = st.sidebar.selectbox(
    "Select LLaMA 3 Model", 
    options=model_options,
    index=0
)

# Prompt Template Input
default_prompt = (
    "Based on the following historical proposal responses and their metadata, "
    "provide three possible draft responses for the given RFP question. "
    "If any part of the question is ambiguous, please suggest clarifying questions (RFIs) "
    "to ensure all requirements are fully addressed."
)
prompt_template = st.sidebar.text_area(
    "Prompt Template",
    value=default_prompt,
    help="Customize the prompt sent to the model for draft responses."
)

# -------------------------------------------
# Config from streamlit
NEO4J_URI = st.secrets["api_keys"]("NEO4J_URI")
NEO4J_USER = st.secrets["api_keys"]("NEO4J_USER")
NEO4J_PASSWORD = st.secrets["api_keys"]("NEO4J_PASSWORD")

# Here, I use API keys primarily from st.secrets, with sidebar override
VOYAGE_API_KEY = voyage_api_key_sidebar or default_voyage_key
GROQ_API_KEY = groq_api_key_sidebar or default_groq_key

# Set the Voyage AI API key in the voyageai module
if VOYAGE_API_KEY:
    voyageai.api_key = VOYAGE_API_KEY

# -------------------------------------------
# Init VoyageAI and ChatGroq
if VOYAGE_API_KEY and GROQ_API_KEY:
    vo = voyageai.Client(api_key=VOYAGE_API_KEY)
    model_mapping = {
        "llama3-8b-8192": "llama3-8b-8192", 
        "llama3-70b-8192": "llama3-70b-8192", 
        "llama-3.2-1b-preview": "llama-3.2-1b-preview",
        "llama-3.2-3b-preview": "llama-3.2-3b-preview",
        "llama-3.2-11b-text-preview": "llama-3.2-11b-text-preview",
        "llama-3.2-90b-text-preview": "llama-3.2-90b-text-preview"
    }
    chat = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model=model_mapping[selected_model], 
        temperature=0.0, 
        max_tokens=None, 
        timeout=None, 
        max_retries=2
    )
else:
    vo = None
    chat = None

# Init Neo4j Driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------------------------
# Neo4j Integration (convert metadata to JSON string)
def create_or_update_proposal(tx, proposal):
    query = """
    MERGE (d:Document {document_id: $document_id})
    SET d.title = $title,
        d.content = $content,
        d.metadata = $metadata
    RETURN d
    """
    metadata_str = json.dumps(proposal["metadata"])
    tx.run(query,
           document_id=proposal["document_id"],
           title=proposal["title"],
           content=proposal["content"],
           metadata=metadata_str)

def save_proposals_to_neo4j(proposals):
    with driver.session() as session:
        for proposal in proposals:
            session.write_transaction(create_or_update_proposal, proposal)

# -------------------------------------------
# Utility Functions
def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def chunk_text(text, chunk_size=8000, chunk_overlap=500):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def generate_embeddings(texts):
    result = vo.embed(texts, model="voyage-large-2-instruct", input_type="document")
    if hasattr(result, "embeddings"):
        result_list = result.embeddings
    else:
        result_list = result
    normalized = [normalize_vector(np.array(vec, dtype='float32')) for vec in result_list]
    return np.array(normalized)

def filter_redundant_chunks(chunks, vectors, similarity_threshold=0.8):
    unique_chunks = []
    unique_vectors = []
    unique_metadata = []
    for i, vec in enumerate(vectors):
        if not unique_vectors:
            unique_chunks.append(chunks[i])
            unique_vectors.append(vec)
            unique_metadata.append(chunk_metadata[i])
        else:
            sim = cosine_similarity([vec], unique_vectors)[0]
            if max(sim) < similarity_threshold:
                unique_chunks.append(chunks[i])
                unique_vectors.append(vec)
                unique_metadata.append(chunk_metadata[i])
    return unique_chunks, np.array(unique_vectors), unique_metadata

def build_faiss_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

# -------------------------------------------
# Ingest Historical Proposals from JSON
input_folder = "input-example"
os.makedirs(input_folder, exist_ok=True)
json_file_path = os.path.join(input_folder, "example-data.json")

historical_texts = []  
filenames = []         
metadata_dict = {}     

if os.path.exists(json_file_path):
    with open(json_file_path, "r") as f:
        proposals = json.load(f)
    save_proposals_to_neo4j(proposals)
    for proposal in proposals:
        text = proposal["content"]
        doc_id = proposal["document_id"]
        historical_texts.append(text)
        filenames.append(doc_id)
        metadata_dict[doc_id] = proposal["metadata"]
else:
    st.error(f"JSON file 'example-data.json' not found in {input_folder}.")

# -------------------------------------------
# Process historical texts: Chunking and associate metadata
all_chunks = []
chunk_metadata = []
for i, doc_text in enumerate(historical_texts):
    chunks = chunk_text(doc_text)
    all_chunks.extend(chunks)
    chunk_metadata.extend([metadata_dict[filenames[i]]] * len(chunks))

# Generate embeddings for all chunks if Voyage client is initialized
if vo is not None:
    vectors = generate_embeddings(all_chunks)
else:
    st.error("Voyage AI API key not set. Please enter your key in the sidebar before generating a response.")
    st.stop()

# Filter redundant chunks
unique_chunks, unique_vectors, unique_metadata = filter_redundant_chunks(all_chunks, vectors, similarity_threshold=0.8)

# Build FAISS index for retrieval
faiss_index = build_faiss_index(unique_vectors)

# -------------------------------------------
# New RFP Query, Retrieval, and Draft Generation
st.subheader("Generate Draft Response for a New RFP Question")
new_query = st.text_area("Enter a new RFP question:")

if st.button("Generate Draft Response") and new_query:
    if not (VOYAGE_API_KEY and GROQ_API_KEY):
        st.error("Please enter your Voyage and Groq AI API keys in the sidebar before generating a response.")
    else:
        # Embed the new RFP question
        query_vector = generate_embeddings([new_query])[0]
        query_vector = np.expand_dims(query_vector, axis=0)
        
        # Retrieve top 3 similar chunks from FAISS index
        k = 3
        distances, indices = faiss_index.search(query_vector, k)
        retrieved_contexts = [unique_chunks[i] for i in indices[0]]
        retrieved_metadata = [unique_metadata[i] for i in indices[0]]
        
        # Compose dynamic prompt using the prompt template from the sidebar
        prompt = "Based on the following historical proposal responses and their metadata:\n"
        for i, (context, metadata) in enumerate(zip(retrieved_contexts, retrieved_metadata), start=1):
            prompt += f"{i}. Response: {context}\n   Metadata: {metadata}\n"
        prompt += (
            "\nFor the following RFP question, provide three possible draft responses. "
            "If any part of the question is ambiguous, please also suggest clarifying questions (RFIs) "
            "to ensure all requirements are fully addressed:\n"
        )
        prompt += f"RFP Question: {new_query}\n"
        
        # Save the prompt in session state for later updating
        st.session_state["prompt"] = prompt
        
        st.subheader("Prompt for LLM")
        st.code(prompt)
        
        if chat is not None:
            draft_response = chat.predict(prompt)
            st.session_state["draft_response"] = draft_response
            st.subheader("Draft Response")
            st.write(draft_response)
        else:
            st.error("ChatGroq not initialized. Please check your Groq API key.")

# -------------------------------------------
# Updated Draft Response with User Reply
if "draft_response" in st.session_state:
    st.subheader("Enter Your Reply to the RFIs")
    user_reply = st.text_area("Enter your reply:")
    if st.button("Generate Updated Draft Response") and user_reply:
        # Append user reply to the original prompt
        updated_prompt = st.session_state["prompt"] + f"\nUser Reply: {user_reply}\n"
        updated_prompt += "Please generate an updated draft response that incorporates the user's clarifications."
        st.subheader("Updated Prompt for LLM")
        st.code(updated_prompt)
        
        if chat is not None:
            updated_draft = chat.predict(updated_prompt)
            st.subheader("Updated Draft Response")
            st.write(updated_draft)
        else:
            st.error("ChatGroq not initialized. Please check your Groq API key.")

# -------------------------------------------
# Footer
st.markdown("---")
st.markdown("Developed by Hammed Akande. Transforming RFP responses using AI.")
