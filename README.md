# Personal Resume Chatbot (RAG Demo with Local LLMs)

This project is a Streamlit web application that demonstrates Retrieval-Augmented Generation (RAG) using a local Large Language Model (LLM) via Ollama. The chatbot answers questions about your work experience, skills, and projects by referencing the content of your resume file. It features a modern chat interface and maintains chat history for a more interactive experience.

## Features
- **Local LLM Inference**: Uses Ollama to run LLMs like `llama3.2:3b` (default) or any compatible model on your Mac (Apple Silicon or Intel).
- **Dedicated Embedding Model**: Uses `nomic-embed-text` for efficient and accurate text embeddings.
- **Retrieval-Augmented Generation (RAG)**: Answers are generated based only on the content of `my_resume.txt`.
- **Modern Streamlit Chat UI**: Interactive chat interface with chat history and assistant/user roles.
- **ChromaDB Vector Store**: Efficient local similarity search for relevant resume chunks.
- **Customizable Prompt**: The assistant answers strictly based on your resume content and does not hallucinate.

## How It Works
1. Loads your resume from `my_resume.txt`.
2. Splits the resume into manageable text chunks.
3. Embeds the chunks using a dedicated embedding model via Ollama.
4. Stores embeddings in a local ChromaDB vector store.
5. When you ask a question, retrieves relevant chunks and generates an answer using the LLM, guided by a custom prompt.
6. Maintains chat history for a conversational experience.

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

### 4. Pull the LLM and Embedding Models with Ollama
- By default, the app uses `llama3.2:3b` for answers and `nomic-embed-text` for embeddings. You can change these in `app.py`.
- To pull the models:
  ```bash
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
  # or use any other supported models
  ```

### 5. Run the App
```bash
streamlit run app.py
```

Open the provided local URL in your browser to interact with the chatbot.


## Demo

https://github.com/VishnuKugarthi/My_Personal_Chatbot/blob/main/RAG%20Local%20LLM%20Demo.m4v

<details>
<summary>Click to view embedded video (GitHub may not preview .m4v inline, but you can download or open in a new tab)</summary>

<video src="https://github.com/VishnuKugarthi/My_Personal_Chatbot/raw/main/RAG%20Local%20LLM%20Demo.m4v" controls width="600"></video>

</details>

## File Structure
- `app.py` — Main Streamlit application
- `my_resume.txt` — Your resume (plain text)
- `requirements.txt` — Python dependencies
- `terminal-commands.txt` — (Optional) Useful terminal commands

## Troubleshooting
- **Ollama not running?** Start it with `ollama serve`.
- **Model not found?** Make sure you have pulled both the LLM and embedding models with `ollama pull <model-name>`.
- **Resume file missing?** Ensure `my_resume.txt` exists in the project directory.
- **Chat not working?** Check that Ollama is running and both models are available.

## Credits
- Built with [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), [ChromaDB](https://www.trychroma.com/), and [Ollama](https://ollama.com/).

---
Demo by Vishnu Teja Kugarthi.

`vishnutejaap@gmail.com`

[Let's connect.](https://vitk.in/)
