import streamlit as st
import pickle
import os
import faiss
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from groq import Groq

# Retrieve Groq API Key securely
try:
    from apikey import GROQ_API_KEY 
    client = Groq(api_key=GROQ_API_KEY)
except ImportError:
    st.error("API Key file 'apikey.py' not found. Please ensure it is present and contains your Groq API Key.")
    st.stop()
except Exception as e:
    st.error(f"Failed to initialize Groq client: {e}")
    st.stop()

# Streamlit UI Setup
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Helper function to validate URLs
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

# Input URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url and not is_valid_url(url):
        st.sidebar.error(f"Invalid URL: {url}")
    urls.append(url)

# Process Button
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    # Validate URLs
    valid_urls = [url for url in urls if is_valid_url(url)]
    if not valid_urls:
        st.error("Please provide at least one valid URL!")
    else:
        try:
            with st.spinner("Processing URLs..."):
                # Load data from URLs
                loader = UnstructuredURLLoader(urls=valid_urls)
                data = loader.load()

                # Limit document size
                MAX_DOC_SIZE = 10_000
                for doc in data:
                    if len(doc.page_content) > MAX_DOC_SIZE:
                        doc.page_content = doc.page_content[:MAX_DOC_SIZE]

                # Split data into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators=['\n\n', '\n', '.', ','],
                    chunk_size=1000
                )
                docs = text_splitter.split_documents(data)
                texts = [doc.page_content for doc in docs]

                # Create embeddings using SentenceTransformer
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder="models")
                embeddings = model.encode(texts)

                # Create FAISS index and add embeddings
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                vectorstore = {
                    "index": index,
                    "texts": texts,
                    "metadata": [doc.metadata for doc in docs]
                }

                # Save the FAISS index to a pickle file
                if os.path.exists(file_path):
                    if st.sidebar.checkbox("Overwrite existing FAISS index?"):
                        with open(file_path, "wb") as f:
                            pickle.dump(vectorstore, f)
                else:
                    with open(file_path, "wb") as f:
                        pickle.dump(vectorstore, f)

                st.success("URLs processed and FAISS index created successfully!")
        except Exception as e:
            st.error(f"Error during processing: {e}")

# Query Section
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
                model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder="models")
                query_embedding = model.encode([query])
                distances, indices = index.search(query_embedding, k=5)
                retrieved_docs = [texts[i] for i in indices[0]]

                # Construct context for Groq completion
                MAX_CONTEXT_LEN = 2000
                context = "\n".join(retrieved_docs)
                if len(context) > MAX_CONTEXT_LEN:
                    context = context[:MAX_CONTEXT_LEN] + "..."

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
                with st.expander("See sources"):
                    for doc in retrieved_docs:
                        st.write(doc)
        else:
            st.warning("FAISS index file not found. Process URLs first!")
    except Exception as e:
        st.error(f"Error during question answering: {e}")
