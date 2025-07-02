import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate # New import for better prompt engineering

# --- Configuration ---
# IMPORTANT: Ensure 'my_resume.txt' exists in the same directory as this app.py file
# and contains the plain text content of your resume.
DOCUMENT_PATH = "my_resume.txt"

# Model for answering questions (LLM for generation)
# Consider models like "llama3:8b" (larger, better quality), "mistral:7b" (good balance),
# or "tinyllama:latest" / "qwen:0.5b-chat" for very small footprint.
# Let's try a slightly larger, more capable model for better answers, if your M2 can handle it.
# If "llama3.2:1b" is still struggling, consider "tinyllama:latest" or "qwen:0.5b-chat".
LLM_MODEL = "llama3.2:3b" # Changed to 3B for potentially better answers

# Dedicated model for generating embeddings (MUST support embeddings)
# nomic-embed-text is highly recommended for this purpose and is very efficient.
EMBEDDING_MODEL = "nomic-embed-text"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Vishnu's Personal Assistant - LLM Demo",
                   page_icon=":robot_face:",
                   layout="centered",
                   initial_sidebar_state="collapsed")

st.title("Hi, I'm Vishnu's Personal Assistant 🤖")
st.subheader("Powered by Local LLMs via Ollama")
st.write("Ask me anything about Vishnu's **work experience, skills, or projects**.")
st.write(f"Powered by **{LLM_MODEL}** (for answers) and **{EMBEDDING_MODEL}** (for understanding)")
st.markdown("---")

# Display document content in an expander for transparency
try:
    with open(DOCUMENT_PATH, 'r') as f:
        st.expander("View Loaded Resume Content").code(f.read())
except FileNotFoundError:
    st.error(f"Error: '{DOCUMENT_PATH}' not found. Please create this file in the same directory as app.py and add Vishnu's resume text.")
    st.stop()

st.markdown("---")

# --- RAG Pipeline Setup (runs once when app starts) ---
@st.cache_resource
def setup_rag_pipeline():
    """
    Sets up the RAG pipeline components: document loading, splitting,
    embedding, vector store creation, and LLM initialization.
    This function is cached to run only once to optimize performance.
    """
    with st.spinner(f"Setting up LLM ({LLM_MODEL}) and RAG pipeline with Embeddings ({EMBEDDING_MODEL})..."):
        # 1. Load document from the specified path
        loader = TextLoader(DOCUMENT_PATH)
        documents = loader.load()

        # 2. Split documents into smaller, manageable chunks
        # Increased chunk_size slightly for more context, adjusted overlap accordingly
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # 3. Create embeddings for the text chunks using the DEDICATED Embedding model
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # 4. Create a vector store (ChromaDB) from the chunks and their embeddings
        vectorstore = Chroma.from_documents(texts, embeddings)

        # 5. Create a retriever to fetch relevant documents from the vector store
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 relevant chunks

        # 6. Initialize the Large Language Model (LLM) for text generation
        llm = OllamaLLM(model=LLM_MODEL)

        # 7. Create a custom prompt template for the RAG chain
        # This guides the LLM to answer based *only* on the provided context.
        template = """You are Vishnu's personal assistant. Use the following context to answer questions about Vishnu.
        If you don't know the answer based on the provided context, politely state that the information is not available in the resume.
        Do not make up answers.

        Context: {context}

        Question: {question}
        Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # 8. Create RetrievalQA chain with the custom prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False, # Set to True if you want to see which chunks were used
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        st.success("RAG pipeline setup complete! Ready to chat.")
        return qa_chain

# Setup the RAG pipeline when the app starts or is rerun
qa_chain = setup_rag_pipeline()

# --- Chat History Management ---
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add an initial greeting from the assistant
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm Vishnu's personal assistant. How can I help you today?"})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Response Generation ---
# Accept user input
# Added a more inviting placeholder
if prompt := st.chat_input("Ask me about Vishnu's experience, skills, or projects...", key="chat_input"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Invoke the QA chain with the user's query
                response = qa_chain.invoke({"query": prompt})
                assistant_response = response["result"]
                st.markdown(assistant_response)
            except Exception as e:
                # Improved error message for the user
                assistant_response = (
                    f"Oops! An error occurred: `{e}`. "
                    "This might be due to Ollama not running, or the models "
                    f"('{LLM_MODEL}' and '{EMBEDDING_MODEL}') not being pulled correctly. "
                    "Please check your terminal for more details and ensure Ollama is active."
                )
                st.error(assistant_response)
                st.warning("If the model is not responding, try restarting Ollama or your Streamlit app.")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

st.markdown("---")
st.caption("This demo showcases local LLM inference and RAG on Apple Silicon. [Learn more about Vishnu](https://vitk.in/).")
