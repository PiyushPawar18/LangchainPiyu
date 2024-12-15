import streamlit as st
from urllib.parse import urlparse
from langchain.document_loaders import UnstructuredURLLoader
import requests

# Helper function to validate URLs
def is_valid_url(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

# Helper function to fetch raw HTML content for debugging
def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text[:500]  # Return first 500 characters for inspection
    except Exception as e:
        return f"Error fetching URL: {e}"

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
                # Debugging: Check raw HTML content
                for url in valid_urls:
                    raw_html = fetch_html(url)
                    if "Error" in raw_html:
                        st.warning(f"Failed to fetch raw HTML for {url}. {raw_html}")
                    else:
                        st.write(f"Raw HTML for {url} (Preview):")
                        st.code(raw_html)

                # Load data from valid URLs using UnstructuredURLLoader
                loader = UnstructuredURLLoader(urls=valid_urls)
                data = loader.load()

                # Check if any content was loaded
                if not data:
                    st.error("Failed to load content from the provided URLs.")
                else:
                    # Display a summary of the loaded content
                    st.success("Content loaded successfully!")
                    for doc in data:
                        st.subheader(f"Content from {doc.metadata.get('source', 'unknown source')}")
                        st.text(doc.page_content[:500])  # Show preview of content
        except Exception as e:
            st.error(f"An error occurred while processing URLs: {e}")
