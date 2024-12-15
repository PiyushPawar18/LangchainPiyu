import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
import os
import requests

# Proxy Configuration
proxies = {
    "http": os.getenv("HTTP_PROXY"),
    "https": os.getenv("HTTPS_PROXY")
}

# Verify the proxy setup
try:
    response = requests.get("https://api.groq.com/health", proxies=proxies)
    if response.status_code == 200:
        print("Proxy setup is working!")
    else:
        print("Proxy setup failed. Check your configuration.")
except Exception as e:
    print(f"Error verifying proxy setup: {e}")

# Streamlit UI Setup
st.title("News Research Tool with Groq 🤖")
st.sidebar.title("Enter News Article URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# Set Groq API Key
groq_api_key = "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"  # Replace with your actual Groq API key
if not groq_api_key:
    st.error("Set your Groq API key in the code.")
    st.stop()

# Initialize Groq Client
try:
    client = Groq(api_key=groq_api_key)
except Exception as e:
    st.error(f"Error initializing Groq client: {e}")
    st.stop()

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Processing Logic
if process_url_clicked:
    try:
        # Step 1: Load data from URLs
        if not urls:
            st.error("Please provide at least one valid URL.")
            st.stop()

        st.info("Loading data from URLs...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        if not data:
            st.error("Failed to load data from the provided URLs. Please check the URLs.")
            st.stop()

        # Step 2: Split text into chunks
        st.info("Splitting text into smaller chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000
        )
        documents = text_splitter.split_documents(data)
        if not documents:
            st.error("No valid text chunks generated from the data. Check the content of the URLs.")
            st.stop()

        # Step 3: Generate embeddings for text chunks
        st.info("Generating embeddings for documents...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store = FAISS.from_documents(documents, embeddings)

        st.success("Embeddings generated and stored in FAISS index!")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# Query Logic
query = st.text_input("Ask a question about the articles:")
if query:
    try:
        # Ensure the vector store is available
        if st.session_state.vector_store is not None:
            st.info("Retrieving relevant documents...")
            relevant_docs = st.session_state.vector_store.similarity_search(query, k=5)
            if not relevant_docs:
                st.warning("No relevant documents found for the query. Try a different question.")
                st.stop()

            # Compile context from relevant documents
            context = "\n".join([doc.page_content for doc in relevant_docs if hasattr(doc, "page_content")])

            # Generate an answer using Groq
            st.info("Generating an answer with Groq...")
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {query}",
                    }
                ],
                model="llama3-8b-8192",  # Replace with your preferred Groq model
            )

            # Display the result
            st.header("Answer")
            st.write(response.choices[0].message.content)

            # Show sources
            st.subheader("Sources")
            for doc in relevant_docs:
                if hasattr(doc, "page_content"):
                    st.write(doc.page_content)
        else:
            st.warning("Please process URLs before querying.")
    except Exception as e:
        st.error(f"An error occurred during querying: {e}")
