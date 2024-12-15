import streamlit as st
from urllib.parse import urlparse
from langchain.document_loaders import UnstructuredURLLoader
from newspaper import Article
import traceback

# Helper function to validate URLs
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

# Helper function to fetch content using newspaper3k
def fetch_content_with_newspaper(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting content with newspaper3k: {e}"

# Streamlit UI Setup
st.title("News Research Tool 🗞️")
st.sidebar.title("Enter News Article URLs")

# Input URLs
urls = []
for i in range(3):  # Allow up to 3 URLs
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
    if url and not is_valid_url(url):
        st.sidebar.error(f"Invalid URL: {url}")
    urls.append(url)

# Process Button
process_urls_clicked = st.sidebar.button("Process URLs")
if process_urls_clicked:
    # Filter out empty or invalid URLs
    valid_urls = [url for url in urls if is_valid_url(url)]
    if not valid_urls:
        st.error("No valid URLs provided! Please enter at least one valid URL.")
    else:
        try:
            with st.spinner("Processing URLs..."):
                # Try loading data using UnstructuredURLLoader
                st.subheader("Attempting to Load Content with UnstructuredURLLoader")
                loader = UnstructuredURLLoader(urls=valid_urls)
                try:
                    data = loader.load()
                except Exception as e:
                    st.warning(f"UnstructuredURLLoader failed: {e}")
                    data = []

                # If UnstructuredURLLoader fails, fallback to newspaper3k
                if not data:
                    st.warning("UnstructuredURLLoader failed to extract content. Using newspaper3k fallback.")
                    for url in valid_urls:
                        content = fetch_content_with_newspaper(url)
                        if "Error" in content:
                            st.error(f"Failed to extract content from {url}: {content}")
                        else:
                            st.success(f"Content extracted from {url} using newspaper3k.")
                            st.text(content[:500])  # Display a preview of the content

                else:
                    # Display content loaded with UnstructuredURLLoader
                    st.success("Content loaded successfully with UnstructuredURLLoader!")
                    for idx, doc in enumerate(data, start=1):
                        st.subheader(f"Content {idx} from {doc.metadata.get('source', 'unknown source')}")
                        st.text(doc.page_content[:500])  # Show preview of content

        except Exception as e:
            st.error(f"An error occurred while processing URLs: {e}")
            st.error("Detailed Traceback:")
            st.text(traceback.format_exc())
