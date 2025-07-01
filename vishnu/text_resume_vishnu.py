import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# --- Configuration ---
# IMPORTANT: Ensure 'my_resume.txt' exists in the same directory as this app.py file
# and contains the plain text content of your resume.
DOCUMENT_PATH = "my_resume.txt"
# Make sure you've pulled this model with Ollama (e.g., ollama pull llama3.2:latest)
# LLM_MODEL = "llama3.2:latest" 
# LLM_MODEL = "qwen3:0.6b"
LLM_MODEL = "llama3.2:latest"  

# Dedicated model for generating embeddings (MUST support embeddings)
# nomic-embed-text is highly recommended for this purpose
EMBEDDING_MODEL = "nomic-embed-text"

# --- Streamlit UI ---
st.set_page_config(page_title="Vishnu's personal assistant - LLM Demo on MacBook Air M2",
                   page_icon=":robot_face:",
                   layout="wide")
st.title("Hi, I'm Vishnu's personal assistant powered by Local LLMs")
st.subheader("You can ask questions about my work experience, skills and projects that I have worked on.")
st.write(f"Powered by {LLM_MODEL} via Ollama")
st.markdown("---")
st.write("This application demonstrates Retrieval-Augmented Generation (RAG) by answering questions based solely on the loaded resume. Ask about my experience, skills, or projects!")

# Display document content (optional, but good for demo and debugging)
try:
    with open(DOCUMENT_PATH, 'r') as f:
        st.expander("View Loaded Document Content").code(f.read())
except FileNotFoundError:
    st.error(f"Error: '{DOCUMENT_PATH}' not found. Please create this file in the same directory as app.py and add your resume text.")
    st.stop() # Stop the app if the document is missing to prevent further errors

# --- RAG Pipeline Setup (runs once when app starts) ---
@st.cache_resource
def setup_rag_pipeline():
    """
    Sets up the RAG pipeline components: document loading, splitting,
    embedding, vector store creation, and LLM initialization.
    This function is cached to run only once.
    """
    with st.spinner(f"Setting up LLM ({LLM_MODEL}) and RAG pipeline with Embeddings ({EMBEDDING_MODEL})..."):
        # 1. Load document from the specified path
        loader = TextLoader(DOCUMENT_PATH)
        documents = loader.load()
        st.write(f"Loaded {len(documents)} document(s).")

        # 2. Split documents into smaller, manageable chunks
        # Chunk size and overlap are crucial for effective retrieval and LLM context.
        # Smaller chunks (e.g., 400 characters) are good for detailed questions and
        # for models with limited context windows, especially on systems with less RAM.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        texts = text_splitter.split_documents(documents)
        
        st.write(f"Split document into {len(texts)} chunks.")

        # 3. Create embeddings for the text chunks using Ollama
        # OllamaEmbeddings uses the specified LLM to convert text chunks into numerical vectors.
        # These vectors are used to find relevant chunks when a query is made.
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL) 

        st.write(f"{EMBEDDING_MODEL} Embeddings initialized.")

        # 4. Create a vector store (ChromaDB) from the chunks and their embeddings
        # ChromaDB stores these text chunks and their embeddings, enabling efficient
        # similarity search later. It runs locally in memory for this demo.
        vectorstore = Chroma.from_documents(texts, embeddings)

        st.write("Vector store created with ChromaDB.")

        # 5. Create a retriever to fetch relevant documents from the vector store
        # The retriever fetches the most similar text chunks based on a user's query.
        retriever = vectorstore.as_retriever()

        st.write("Retriever configured.")

        # 6. Initialize the Large Language Model (LLM) via Ollama
        # This is the model that will generate answers based on the retrieved context.
        llm = OllamaLLM(model=LLM_MODEL)
        st.write(f"LLM '{LLM_MODEL}' initialized.")

        # 7. Create the RetrievalQA chain
        # This chain connects the retriever and the LLM. When a query comes in:
        #   a) Retriever finds relevant text chunks.
        #   b) These chunks are "stuffed" (concatenated) into the LLM's prompt.
        #   c) The LLM generates an answer based on the query and the provided context.
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        st.success("RAG pipeline setup complete!")
        return qa_chain

# Setup the RAG pipeline when the app starts or is rerun
qa_chain = setup_rag_pipeline()

# --- User Interaction ---
st.markdown("---")
# Text input for the user to type their question
user_query = st.text_input("Ask a question about Vishnu:", placeholder="e.g., What are Vishnu's skills, work experience and projects?", key="user_query")

# Process the query when the user submits one
if user_query:
    with st.spinner("Generating answer..."):
        try:
            # Invoke the QA chain with the user's query
            response = qa_chain.invoke({"query": user_query})
            st.write("**Answer:**")
            # Display the answer from the LLM
            st.info(response["result"])
        except Exception as e:
            # Error handling for issues with Ollama or model loading
            st.error(f"An error occurred: {e}. Make sure Ollama is running and the model '{LLM_MODEL}' is pulled.")
            st.warning("If the model is not responding, try restarting Ollama or your Streamlit app.")

st.markdown("---")
st.caption("This demo showcases local LLM inference and RAG on Apple Silicon. [Learn more about Vishnu](https://vitk.in/).")
