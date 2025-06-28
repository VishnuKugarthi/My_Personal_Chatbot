import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM # Note the change to OllamaLLM
from langchain.chains import RetrievalQA

# --- Configuration ---
DOCUMENT_PATH = "my_resume.txt" # Ensure this file exists in the same directory as app.py
LLM_MODEL = "phi4-mini:3.8b" # Make sure you've pulled this model with ollama

# --- Streamlit UI ---
st.set_page_config(page_title="My Personal LLM Demo on MacBook Air M2", 
                   page_icon=":robot_face:", 
                   layout="wide")
st.title("Local LLM Document Q&A on MacBook Air M2")
st.subheader(f"Powered by {LLM_MODEL} via Ollama")
st.markdown("---")
st.write("This application demonstrates Retrieval-Augmented Generation (RAG) by answering questions based solely on the loaded resume. Ask about my experience, skills, or projects!")

# Display document content (optional, but good for demo)
try:
    with open(DOCUMENT_PATH, 'r') as f:
        st.expander("View Loaded Document Content").code(f.read())
except FileNotFoundError:
    st.error(f"Error: '{DOCUMENT_PATH}' not found. Please create this file in the same directory as app.py and add your text (e.g., resume, project details).")
    st.stop() # Stop the app if the document is missing

# --- RAG Pipeline Setup (runs once when app starts) ---
@st.cache_resource
def setup_rag_pipeline():
    with st.spinner(f"Setting up LLM and RAG pipeline with {LLM_MODEL}..."):
        # 1. Load document
        loader = TextLoader(DOCUMENT_PATH)
        documents = loader.load()

        # 2. Split into chunks
        # Smaller chunk size for 8GB RAM, to keep embeddings manageable
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        texts = text_splitter.split_documents(documents)

        # 3. Create embeddings using Ollama (efficient on M2)
        # OllamaEmbeddings uses the specified LLM for generating embeddings.
        embeddings = OllamaEmbeddings(model=LLM_MODEL)

        # 4. Create vector store (ChromaDB runs locally in memory for this demo)
        vectorstore = Chroma.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()

        # 5. Initialize LLM (also via Ollama)
        llm = OllamaLLM(model=LLM_MODEL)

        # 6. Create RetrievalQA chain
        # "stuff" chain type puts all retrieved docs into the prompt. Good for small docs.
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        st.success("RAG pipeline setup complete!")
        return qa_chain

qa_chain = setup_rag_pipeline()

# --- User Interaction ---
st.markdown("---")
user_query = st.text_input("Ask a question about the document:")

if user_query:
    with st.spinner("Generating answer..."):
        try:
            response = qa_chain.invoke({"query": user_query})
            st.write("**Answer:**")
            st.info(response["result"])
        except Exception as e:
            st.error(f"An error occurred: {e}. Make sure Ollama is running and the model '{LLM_MODEL}' is pulled.")
            st.warning("If the model is not responding, try restarting Ollama or your Streamlit app.")

st.markdown("---")
st.caption("This demo showcases local LLM inference and RAG on Apple Silicon. [Learn more about LangChain](https://www.langchain.com/) and [Ollama](https://ollama.com/).")