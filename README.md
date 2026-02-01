# MVP Project: GitHub Code Documentation Assistant

A Streamlit application that lets you chat with the contents of any GitHub repository. 
It uses FAISS for vector storage, Ollama for embeddings and language model, and PydanticAI as the agent framework to handle retrieval and question-answering.

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [How It Works](#how-it-works)
* [Customization](#customization)



---

## Features

* **GitHub Repo Cloning**: Clone any public GitHub repository and extract text from code and documentation files.
* **Text Chunking**: Split large text into manageable chunks for efficient vector storage.
* **Vector Search**: Build a FAISS index of embeddings to enable semantic similarity search.
* **PydanticAI Agent**: Leverage PydanticAI’s agent-and-tool architecture to handle retrieval and LLM calls.
* **Ollama Integration**: Use Ollama’s local LLM and embedding models for both embeddings and chat completions.
* **Streamlit UI**: Interactive web interface for loading repos and chatting with contents.

---

## Prerequisites

* **Python 3.10+**
* **Ollama** installed and running locally (default HTTP endpoint `http://localhost:11434`)
* Git installed on your system

---

## Installation

1. **Clone this repository**

   ```bash
   git clone 
   cd github-repo-GitHub_CodeDocAssist
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   venv/Scripts/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

1. **Run Ollama daemon**

   Ensure Ollama is up and running. By default it listens on `http://localhost:11434`.

2. **Model names**

   * Embedding model: `all-minilm:33m`
   * LLM model: `tinyllama`

   You can change these in `app.py` when initializing `OllamaEmbeddings` and `OpenAIModel`.

---

## Usage

Launch the Streamlit app:

```bash
streamlit run app.py
```

1. Open the URL shown in the console (usually `http://localhost:8501`).
2. Enter a **GitHub repository URL** in the sidebar (e.g., `https://github.com/mjmanoj2004/chat-with-docs`).
3. Select file extensions to include (default: `.py, .md, .txt, .js, .html, .css, .json`).
4. Click **Load Repository** to clone, process, and index the repo.
5. Once loaded, ask questions in the chat interface about the repository’s contents.

---


## How It Works

1. **Clone & Extract**

   * `get_repo_text` clones the repository into a temporary directory.
   * Walks through files with allowed extensions and concatenates their contents.

2. **Chunking**

   * `split_text` splits the combined text into overlapping chunks of \~1000 characters.

3. **Vector Store**

   * `create_vectorstore` builds a FAISS index using OllamaEmbeddings on each chunk.

4. **Agent & Retrieval**

   * `initialize_agent` creates a PydanticAI `Agent` with an `retrieve` tool that returns the top-4 similar chunks.
   * When a user query comes in, PydanticAI handles tool invocation (retrieval) and crafts the prompt for Ollama to answer.

5. **Chat UI**

   * Streamlit displays previous messages and handles user input via `st.chat_input`.
   * Responses from the agent are streamed back into the chat interface.

## Customization

* **Chunk Size & Overlap**: Adjust `chunk_size` and `overlap` in `split_text`.
* **Number of Docs**: Change the `k` parameter in `vectorstore.similarity_search` within the `retrieve` tool.
* **Models**: Swap out `OllamaEmbeddings` or `OpenAIModel` parameters for different model sizes or temperatures.
* **Additional Tools**: Add more `@agent.tool` functions to perform extra tasks (e.g., code search, summary, translation).
---
a. Quick setup instructions

Install Python 3.10+ and Git.
Install dependencies: pip install -r requirements.txt
Ensure Ollama is running locally (for LLM/embeddings).
Run: streamlit run gitchat_app.py
Enter a GitHub repo URL in the sidebar and click "Load Repository" to start chatting.

b. Architecture overview
Streamlit UI for user interaction.
GitHub repo is cloned and relevant files are extracted.
Text is chunked and embedded using Ollama (local LLM).
Chunks are stored in a FAISS vector database.
User questions are answered by retrieving relevant chunks and using a PydanticAI agent (with Ollama as the LLM backend).

c. Productionization & scalability
Containerize the app (Docker).
Use a managed vector DB (e.g., Pinecone, Weaviate) instead of local FAISS.
Deploy LLM/embedding model as a scalable API (e.g., on AWS Sagemaker, GCP Vertex AI, or Azure OpenAI).
Use cloud storage for repo data.
Add authentication, monitoring, and autoscaling.
Deploy Streamlit or a web frontend on a managed service (e.g., AWS ECS, GCP Cloud Run).

d. RAG/LLM approach & decisions
Retrieval-Augmented Generation (RAG) is used: user queries retrieve top-k relevant code/doc chunks via FAISS, then LLM generates answers.
LLM: Ollama (local, open-source, model: llama3.2).
Embeddings: Ollama all-minilm:33m.
Vector DB: FAISS (local, for demo; would use managed DB in prod).
Orchestration: PydanticAI agent wraps retrieval and LLM.
Prompt: System prompt enforces grounding in repo content, concise answers, and citation of files/snippets.
Guardrails: Prompt-based (no code execution, factuality emphasis).
Observability: Minimal in demo; would add logging, tracing, and analytics in production.

e. Key technical decisions
Chose local LLM/embeddings for privacy and cost.
Used FAISS for simplicity and speed in local/dev.
Streamlit for rapid prototyping and UI.
PydanticAI agent for modular, tool-augmented LLM interaction.
Focused on extensibility (easy to swap vector DB, LLM, or UI).

f. Engineering standards
Modular code (functions for each step).
Type hints and docstrings.
Error handling for repo cloning and file reading.
Used open-source, well-maintained libraries.
Skipped: full test coverage, CI/CD, advanced logging, and security hardening (for demo).

g. Use of AI tools in development
Used LLMs (like Copilot) for code generation, refactoring, and docstring writing.
Used AI for brainstorming architecture and prompt design.
Validated code and prompts iteratively with LLM feedback.

h. What I'd do differently with more time
Add tests and CI/CD.
Support more file types and larger repos (streaming, async).
Add user authentication and rate limiting.
Improve UI/UX (file tree, code highlighting, chat history).
Add analytics and observability.
Support multi-LLM backends and cloud deployment out of the box.


