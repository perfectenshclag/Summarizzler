# Import Required Modules
import os
import validators
import streamlit as st
from bs4 import BeautifulSoup
import requests
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-specdec")

# Streamlit App Configuration
st.set_page_config(page_title="Summarizzler: The Final Boss!", page_icon="🦜")
st.title("🦜 Summarizzler: The Final Boss")
st.subheader("Summarize or Query URL Content 🧙🏻")

# User Inputs
generic_url = st.text_input("Enter URL", label_visibility="visible")
operation = st.selectbox("Choose Operation:", ["Summarize Content", "Query Extracted Text"])

# Show the query input box only if "Query Extracted Text" is selected
query = None
if operation == "Query Extracted Text":
    query = st.text_input("Enter your query:", label_visibility="visible")

# Prompt Templates
summary_prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
summary_prompt = PromptTemplate(template=summary_prompt_template, input_variables=["text"])

query_prompt_template = """
Answer the following question based on the provided context and if not available in context answer yourself and specify that you've answered:
Question: {query}
Context: {context}
"""
query_prompt = PromptTemplate(template=query_prompt_template, input_variables=["query", "context"])

# Helper Functions
def extract_website_content(url):
    """Extracts text content from a URL."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # Extracting all text content from valid tags
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "li", "span"], string=True)
        text = "\n".join(element.get_text(strip=True) for element in text_elements)
        return text if text else "No valid content found on the webpage."
    except Exception as e:
        st.error("Error extracting website content.")
        st.exception(e)
        return ""

def create_vector_db(docs):
    """Creates a FAISS vector database from a list of documents."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    split_docs = text_splitter.split_documents(docs)
    vector_db = FAISS.from_documents(split_docs, embeddings)
    return vector_db
# Main Workflow
if st.button("Process URL"):
    if not validators.url(generic_url):
        st.error("Please enter a valid URL.")
    else:
        try:
            with st.spinner("Processing..."):
                progress_bar = st.progress(0)  # Initialize progress bar
                status_placeholder = st.empty()  # Placeholder for status updates
                
                # Step 1: URL content extraction
                progress_bar.progress(20)
                status_placeholder.markdown("**Step 1/4: Extracting website content...**")
                
                if "youtube.com" in generic_url:
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=False, language="English")
                        docs = loader.load()
                    except Exception as yt_error:
                        st.error("Failed to load YouTube video details. Please try another URL.")
                        st.exception(yt_error)
                        docs = []
                else:
                    extracted_text = extract_website_content(generic_url)
                    if extracted_text:
                        docs = [Document(page_content=extracted_text)]
                    else:
                        docs = []

                if not docs:
                    st.warning("No content available for processing.")
                    st.stop()

                progress_bar.progress(40)
                status_placeholder.markdown("**Step 2/4: Creating vector database...**")
                
                # Step 2: Create vector database
                vector_db = create_vector_db(docs)

                progress_bar.progress(70)
                if operation == "Summarize Content":
                    status_placeholder.markdown("**Step 3/4: Summarizing content...**")
                    
                    # Summarize content
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=summary_prompt)
                    output_summary = chain({"input_documents": docs})
                    
                    progress_bar.progress(100)
                    status_placeholder.markdown("**Done! Summary generated.**")
                    
                    st.success("Summary:")
                    st.write(output_summary["output_text"])

                elif operation == "Query Extracted Text" and query:
                    status_placeholder.markdown("**Step 3/4: Querying extracted text...**")
                    
                    # Query content
                    retriever = vector_db.as_retriever()
                    retrieved_docs = retriever.get_relevant_documents(query)

                    if not retrieved_docs:
                        st.warning("No relevant documents found for the query.")
                        st.stop()

                    progress_bar.progress(90)
                    status_placeholder.markdown("**Step 4/4: Generating query result...**")
                    
                    # Combine retrieved content into a single context
                    combined_context = "\n".join(doc.page_content for doc in retrieved_docs)

                    # Use the custom query prompt
                    query_input = query_prompt.format(query=query, context=combined_context)
                    result = llm.predict(query_input)
                    
                    progress_bar.progress(100)
                    status_placeholder.markdown("**Done! Query result generated.**")
                    
                    st.success("Query Result:")
                    st.write(result)
        except Exception as e:
            st.error("An unexpected error occurred.")
            st.exception(e)
