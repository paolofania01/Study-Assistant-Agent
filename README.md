# Study Assistant Agent

An AI-powered study assistant that reads your PDFs and helps you learn through three modes: chat, flashcards, and quizzes. Available both as a terminal application and a Telegram bot. Built with LangGraph and RAG (Retrieval-Augmented Generation).

> **Note:** This is a first version of the project. Features and improvements will be added over time.

## Features

- **Chat** — Ask questions about your study material and get explanations based on the documents
- **Flashcard** — Generate Q&A flashcards on any topic, automatically saved for later review
- **Quiz** — Create multiple-choice quizzes to test your knowledge, saved persistently
- **Telegram Bot** — Access the assistant from your phone via Telegram (chat mode)
- **Study History Tracking** — Automatically tracks covered topics to avoid repetition

## Tech Stack

- **LangGraph** — Agent orchestration and state management
- **LangChain** — Tool framework and document processing
- **Groq API** — LLM inference (Llama 3.3 70B)
- **Ollama** — Local embeddings (nomic-embed-text) and optional local LLM
- **ChromaDB** — Vector store for document retrieval
- **python-telegram-bot** — Telegram bot interface

## Model Options

The project supports multiple LLM backends. You can switch between them in `nodes.py` and `tools.py`:

- **Groq (recommended)** — `llama-3.3-70b-versatile` offers the best quality and speed. Requires a free Groq API key.
- **Ollama (local)** — `qwen2.5:7b` or `llama3.1:8b` for offline use. Slower and slightly lower quality, but no API limits.

For the best experience, use Groq. Ollama is a good fallback when you hit Groq's daily rate limit.

## Project Structure

```
├── main.py            # Terminal entry point — graph construction and conversation loop
├── telegram_bot.py    # Telegram bot interface
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

4. Add your PDFs to the `documents/` folder.

## Usage

### Terminal Mode

Run the agent in terminal:
```bash
python main.py
```

Example commands:
```
What is your question: Explain virtual memory
What is your question: Create flashcards on scheduling algorithms
What is your question: Quiz me on disk management
```

Type `exit` or `quit` to end the session.

### Telegram Bot Mode

1. Create a bot on Telegram via [@BotFather](https://t.me/BotFather) and get your token
2. Add it to your `.env`:
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   ```
3. Run the bot:
   ```bash
   python telegram_bot.py
   ```
4. Open Telegram, search for your bot, and start chatting!

Currently, the Telegram bot supports chat mode only. Flashcard and quiz modes are available in terminal mode.

## Planned Improvements

- Flashcard and quiz modes on Telegram
- Interactive quiz verification with score tracking
- PDF upload via Telegram
- Multi-user support with isolated documents
- Web interface with Streamlit
