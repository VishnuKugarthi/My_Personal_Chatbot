# Vishnu's Personal Chatbot (RAG Demo with Local LLMs)

This project is a Streamlit web application that demonstrates Retrieval-Augmented Generation (RAG) using a local Large Language Model (LLM) via Ollama. The chatbot answers questions about Vishnu's work experience, skills, and projects by referencing the content of a resume file.

## Features
- **Local LLM Inference**: Uses Ollama to run LLMs like `llama3.2:latest` or `phi4-mini:3.8b` on your MacBook Air M2 (Apple Silicon).
- **Retrieval-Augmented Generation (RAG)**: Answers are generated based only on the content of `my_resume.txt`.
- **Modern Streamlit UI**: Simple, interactive web interface for asking questions.
- **ChromaDB Vector Store**: Efficient local similarity search for relevant resume chunks.

## How It Works
1. Loads your resume from `my_resume.txt`.
2. Splits the resume into manageable text chunks.
3. Embeds the chunks using Ollama's LLM embeddings.
4. Stores embeddings in a local ChromaDB vector store.
5. When you ask a question, retrieves relevant chunks and generates an answer using the LLM.

## Setup Instructions

### 1. Prerequisites
- **Python 3.9+** (recommended: 3.10 or 3.11)
- **Ollama** installed and running locally ([get it here](https://ollama.com/))
- **Apple Silicon Mac** (optimized for M1/M2/M3, but works on Intel Macs too)

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Resume
- Create a file named `my_resume.txt` in the project directory.
- Paste your resume content (plain text) into this file.

### 4. Pull the LLM Model with Ollama
- By default, the app uses `llama3.2:latest`. You can change this in `app.py`.
- To pull the model:
  ```bash
  ollama pull llama3.2:latest
  # or for phi4-mini
  # ollama pull phi4-mini:3.8b
  ```

### 5. Run the App
```bash
streamlit run app.py
```

Open the provided local URL in your browser to interact with the chatbot.

## File Structure
- `app.py` — Main Streamlit application
- `my_resume.txt` — Your resume (plain text)
- `requirements.txt` — Python dependencies
- `terminal-commands.txt` — (Optional) Useful terminal commands

## Troubleshooting
- **Ollama not running?** Start it with `ollama serve`.
- **Model not found?** Make sure you have pulled the model with `ollama pull <model-name>`.
- **Resume file missing?** Ensure `my_resume.txt` exists in the project directory.

## Credits
- Built with [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), and [Ollama](https://ollama.com/).

---
Demo by VISHNU TEJA KUGARTHI. 
[My Website](https://vitk.in/)
