# Study Assistant Agent

An AI-powered study assistant that reads your PDFs and helps you learn through three modes: chat, flashcards, and quizzes. Built with LangGraph and RAG (Retrieval-Augmented Generation).

> **Note:** This is a first version of the project. Features and improvements will be added over time.

## Features

- **Chat** — Ask questions about your study material and get explanations based on the documents
- **Flashcard** — Generate Q&A flashcards on any topic from your PDFs (saved persistently)
- **Quiz** — Create multiple-choice quizzes to test your knowledge (saved persistently)

## Tech Stack

- **LangGraph** — Agent orchestration and state management
- **LangChain** — Tool framework and document processing
- **Groq API** — LLM inference (Llama 3.3 70B)
- **Ollama** — Local embeddings (nomic-embed-text) and optional local LLM
- **ChromaDB** — Vector store for document retrieval

## Model Options

The project supports multiple LLM backends. You can switch between them in `nodes.py` and `tools.py`:

- **Groq (recommended)** — `llama-3.3-70b-versatile` offers the best quality and speed. Requires a free Groq API key.
- **Ollama (local)** — `qwen2.5:7b` or `llama3.1:8b` for offline use. Slower and slightly lower quality, but no API limits.

For the best experience, use Groq. Ollama is a good fallback when you hit Groq's daily rate limit.

## Project Structure

```
├── main.py            # Entry point — graph construction and conversation loop
├── state.py           # State definitions (AgentState, StudyHistory)
├── tools.py           # Tool definitions with InjectedState
├── nodes.py           # Graph nodes (router, LLM caller, continuation check)
├── vectorstore.py     # PDF loading, chunking, and ChromaDB setup
├── documents/         # Place your PDFs here
├── chroma_db/         # Vector store (auto-generated)
├── flashcards/        # Saved flashcards (auto-generated)
└── quizzes/           # Saved quizzes (auto-generated)
```

## Setup

1. Clone the repo and install dependencies:
```bash
git clone https://github.com/paolofania01/study-assistant-agent.git
cd study-assistant-agent
pip install -r requirements.txt
```

2. Pull the embedding model (required):
```bash
ollama pull nomic-embed-text
```

3. Choose your LLM backend:

   **For Groq (recommended):** create a `.env` file with your Groq API key:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

   **For Ollama (local):** pull the model:
   ```bash
   ollama pull qwen2.5:7b
   ```

4. Add your PDFs to the `documents/` folder and run:
```bash
python main.py
```

## Usage Examples

```
What is your question: Explain the concept of virtual memory
What is your question: Create flashcards on scheduling algorithms
What is your question: Quiz me on disk management
```

Type `exit` or `quit` to end the session.

## Planned Improvements

- Interactive quiz verification with score tracking
- Persistent study history across sessions
- Telegram bot integration
- Web interface with Streamlit
