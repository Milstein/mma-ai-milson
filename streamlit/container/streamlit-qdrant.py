import streamlit as st
import requests
import os
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.text_splitter import CharacterTextSplitter
import httpx
import json
import asyncio
import urllib.parse

# Streamlit app title
st.title("Retrieval Augmented Generation based on a given pdf")

# Qdrant connection parameters
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
QDRANT_API_KEY = "36e0550c8a4e63c182ad"


# LLAMA Server connection parameters
LLAMA_HOST = "llama-service"
LLAMA_PORT = "8080"

@st.cache_resource
def load_and_process_pdfs():
    pdf_urls = [os.getenv("PDF_URL")]
    url_parts = urllib.parse.urlparse(os.getenv("PDF_URL"))
    path_query = url_parts.path
    path_filename = os.path.split(path_query)
    pdf_names = [os.path.basename(path_filename[1])]
    
    all_docs = []
    
    for url, name in zip(pdf_urls, pdf_names):
        if not os.path.exists(name):
            output_path = os.path.join("/tmp/", name)
            st.write(f"Downloading {name}...")
            res = requests.get(url)
            with open(output_path, 'wb') as file:
                file.write(res.content)
        
        st.write(f"Processing {name}...")
        loader = PyPDFLoader(output_path)
        docs = loader.load()
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=768, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)
        all_docs.extend(split_docs)
    
    st.write("Embedding documents...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    st.write("Connecting to Qdrant...")
    client = QdrantClient(
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        api_key=QDRANT_API_KEY
    )
    
    st.write("Creating vector store...")
    vector_store = Qdrant.from_documents(
        all_docs,
        embedding=embeddings,
        client=client,
        collection_name="lighthouse"
    )
    
    st.write("Processing complete!")
    return vector_store

# Function to build prompt
def build_prompt(question, topn_chunks: list[str]):
    prompt = "Instructions: Compose a concise answer to the query using the provided search results, no need to mention you found it in the resuts\n\n"
    prompt += "Search results:\n"
    for chunk in topn_chunks:
        prompt += f"[Document: {chunk[0].metadata.get('source', 'Unknown')}, Page: {chunk[0].metadata.get('page', 'Unknown')}]: " + chunk[0].page_content.replace("\n", " ") + "\n\n"
    prompt += f"Query: {question}\n\nAnswer: "
    return prompt

# Asynchronous function to get LLAMA response
async def get_llama_response(prompt):
    json_data = {
        'prompt': prompt,
        'temperature': 0.1,
        'n_predict': 200,
        'stream': True,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream('POST', f'http://{LLAMA_HOST}:{LLAMA_PORT}/completion', json=json_data) as response:
            full_response = ""
            async for chunk in response.aiter_bytes():
                try:
                    data = json.loads(chunk.decode('utf-8')[6:])
                    if data['stop'] is False:
                        full_response += data['content']
                except:
                    pass
    return full_response

# Load and process PDFs
with st.spinner("Loading and processing PDFs... This may take a few minutes."):
    vector_store = load_and_process_pdfs()

# User input
question = st.text_input("Enter your question about the pdf you picked:")

if question:
    # Perform similarity search
    docs = vector_store.similarity_search_with_score(question, k=3)
    
    # Build prompt
    prompt = build_prompt(question, docs)
    
    # Get LLAMA response
    with st.spinner("Generating answer..."):
        answer = asyncio.run(get_llama_response(prompt))
    
    # Display answer
    st.write("Answer:", answer)
