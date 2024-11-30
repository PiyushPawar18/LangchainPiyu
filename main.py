import streamlit as st
import pickle
import os
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredURLLoader
from apikey import GROQ_API_KEY  # Ensure this file contains a valid API key
from groq import Groq
import httpx

proxies = {
    "http": "http://your-proxy.com:8080",
    "https": "http://your-proxy.com:8080",
}

# Streamlit UI Setup
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

# Example: Handling initialization with optional proxies
try:
    client = Groq(api_key=GROQ_API_KEY)
    # Optional: Assign custom httpx client with proxies if needed
    client.http_client = httpx.Client(proxies=proxies)
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}. Please ensure your API key and Groq library version are correct.")
    st.stop()

if process_url_clicked and urls:
    try:
        # Load data from URLs
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading... Started ✅")
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitting... Started ✅")
        docs = text_splitter.split_documents(data)
        texts = [doc.page_content for doc in docs]

        # Create embeddings using SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        main_placeholder.text("Embedding Vector Building... Completed ✅")

        # Create FAISS index and add embeddings
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        vectorstore = {
            "index": index,
            "texts": texts,
            "metadata": [doc.metadata for doc in docs]
        }

        # Save FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
        main_placeholder.text("FAISS Index Saved ✅")
        st.success("URLs processed successfully!")
    except Exception as e:
        st.error(f"Error processing URLs: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                index = vectorstore["index"]
                texts = vectorstore["texts"]
                metadata = vectorstore["metadata"]

                # Retrieve top 5 relevant documents
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
                query_embedding = model.encode([query])
                distances, indices = index.search(query_embedding, k=5)
                retrieved_docs = [texts[i] for i in indices[0]]

                # Construct context for Groq completion
                context = "\n".join(retrieved_docs)

                # Get answer from Groq
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Context: {context}\n\nQuestion: {query}",
                        }
                    ],
                    model="llama3-8b-8192"
                )
                result = chat_completion.choices[0].message.content

                st.header("Answer")
                st.write(result)

                # Display sources
                st.subheader("Sources:")
                for doc in retrieved_docs:
                    st.write(doc)
        else:
            st.error("FAISS index file not found. Please process URLs first.")
    except Exception as e:
        st.error(f"Error during query processing: {e}")
