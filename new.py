import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq
import os

# Streamlit UI Setup
st.title("News Research Tool with Groq 🤖")
st.sidebar.title("Enter News Article URLs")

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
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
        st.info("Loading data from URLs...")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Step 2: Split text into chunks
        st.info("Splitting text into smaller chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000
        )
        documents = text_splitter.split_documents(data)

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
        # Check if the vector store is available
        if st.session_state.vector_store is not None:
            st.info("Retrieving relevant documents...")

            # Step 4: Retrieve relevant documents
            relevant_docs = st.session_state.vector_store.similarity_search(query, k=5)
            context = "\n".join([doc.page_content for doc in relevant_docs])

            # Step 5: Use Groq for answering the query
            st.info("Generating an answer with Groq...")
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {query}",
                    }
                ],
                model="llama3-8b-8192",  # Replace with your preferred Groq model if needed
            )

            st.header("Answer")
            st.write(response.choices[0].message.content)

            st.subheader("Sources")
            for doc in relevant_docs:
                st.write(doc.page_content)
        else:
            st.warning("Please process URLs before querying.")
    except Exception as e:
        st.error(f"An error occurred during querying: {e}")
