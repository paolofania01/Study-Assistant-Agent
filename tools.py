# tools.py — Defines the tools available to the LLM agent.
# Each tool uses InjectedState to access the graph's state (study history, etc.)
# and the shared retriever to search through the study documents.

from typing import Annotated
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState
from vectorstore import get_retriever
import json
import os
from langchain_groq import ChatGroq

# Created once and shared across all tools to avoid reloading PDFs on every call
retriever = get_retriever()


generation_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# generation_llm = ChatOllama(model="llama3.1:8b", temperature=0)
# generation_llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# We save all flashcards in a single JSON file for simplicity. Each flashcard has a "question" and "answer" field.
FLASHCARDS_FILE = "flashcards/flashcards.json"

QUIZZES_FILE = "quizzes/quizzes.json"


def load_all_flashcards() -> list:
    """Load all saved flashcards from the single JSON file."""
    if not os.path.exists("flashcards"):
        os.makedirs("flashcards")
    
    if not os.path.exists(FLASHCARDS_FILE):
        return []
    
    try:
        with open(FLASHCARDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading flashcards: {e}")
        return []


def parse_flashcards(text: str) -> list:
    """Parse LLM output into a list of flashcard dicts.
    Expected format:
    Q: <question>
    A: <answer>
    """
    flashcards = []
    lines = text.strip().split("\n")
    
    current_question = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("Q:"):
            current_question = line[2:].strip()  
        elif line.startswith("A:") and current_question:
            answer = line[2:].strip()
            flashcards.append({
                "question": current_question,
                "answer": answer
            })
            current_question = None
    
    return flashcards


def save_all_flashcards(flashcards: list):
    """Save all flashcards to the single JSON file."""
    if not os.path.exists("flashcards"):
        os.makedirs("flashcards")
    
    try:
        with open(FLASHCARDS_FILE, "w", encoding="utf-8") as f:
            json.dump(flashcards, f, ensure_ascii=False, indent=2)
        print(f"[Flashcards saved] {FLASHCARDS_FILE} ({len(flashcards)} total)")
    except Exception as e:
        print(f"Error saving flashcards: {e}")
        
def load_all_quizzes() -> list:
    """Load all saved quizzes from the single JSON file."""
    if not os.path.exists("quizzes"):
        os.makedirs("quizzes")
    
    if not os.path.exists(QUIZZES_FILE):
        return []
    
    try:
        with open(QUIZZES_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
        return json.loads(content)
    except Exception as e:
        print(f"Error loading quizzes: {e}")
        return []


def save_all_quizzes(quizzes: list):
    """Save all quizzes to the single JSON file."""
    if not os.path.exists("quizzes"):
        os.makedirs("quizzes")
    
    try:
        with open(QUIZZES_FILE, "w", encoding="utf-8") as f:
            json.dump(quizzes, f, ensure_ascii=False, indent=2)
        print(f"[Quizzes saved] {QUIZZES_FILE} ({len(quizzes)} total)")
    except Exception as e:
        print(f"Error saving quizzes: {e}")

def parse_quiz(text: str, topic: str) -> list:
    """Parse LLM quiz output into a list of quiz dicts."""
    import re
    
    quizzes = []
    # Split the text by questions (Q1:, Q2:, etc.)
    blocks = re.split(r"Q\d+:", text)
    
    for block in blocks[1:]:  # skip the first empty split
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        
        if len(lines) < 5:
            continue
        
        question = lines[0]
        options = {}
        correct = None
        
        for line in lines[1:]:
            if line.startswith(("A)", "B)", "C)", "D)")):
                letter = line[0]
                options[letter] = line[2:].strip()
            elif "correct" in line.lower():
            # Search for the letter after "answer:" to avoid matching the C in "Correct"
                match = re.search(r"answer\s*:\s*([A-D])", line, re.IGNORECASE)
                if match:
                    correct = match.group(1)
        
        if question and len(options) == 4 and correct:
            quizzes.append({
                "topic": topic,
                "question": question,
                "options": options,
                "correct": correct
            })
    
    return quizzes
       
@tool
def retriever_tool(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Searches the study documents for relevant information. 
    Use this tool to answer questions about the study material."""
    
    docs = retriever.invoke(query)
    
    if not docs:
        return "I found no relevant information in the documents about the study material"

    results = []
    
    # Format each chunk with its source file for traceability
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_file", "unknown")
        results.append(f"[Source: {source}] Document {i+1}:\n{doc.page_content}")
        
    return "\n\n".join(results)
    
@tool
def flashcard_generator(topic: str, state: Annotated[dict, InjectedState]) -> str:
    """Generate flashcards for a given topic.
    The tool creates question-answer pairs based on the study documents
    and avoids repeating already covered topics from the study history.
    Flashcards are accumulated in a single JSON file."""
    
    docs = retriever.invoke(topic)
    
    if not docs:
        return "I found no relevant information in the documents about the study material"
    
    # Load existing flashcards to avoid duplicates
    existing_flashcards = load_all_flashcards()

    context = "\n".join([doc.page_content for doc in docs])
    
    existing_text = "\n".join([f"Q: {fc['question']}\nA: {fc['answer']}" for fc in existing_flashcards])
                              
    prompt = f"""You are an expert study assistant that creates high-quality flashcards.

        Generate flashcards based ONLY on the provided context.

        MAIN TOPIC: {topic}

        CONTEXT:
        {context}

        ALREADY GENERATED FLASHCARDS (do NOT repeat these):
        {existing_text if existing_text else "None yet"}

        INSTRUCTIONS:
        - Generate NEW flashcards on topics not yet covered above
        - Format each flashcard exactly as:
        Q: <question>
        A: <answer>

        - Generate 5 to 10 flashcards
        - Do not include information not present in the context
        """
    
    # The tool calls the LLM directly (without tool binding)
    response = generation_llm.invoke(prompt)
    generated_text = response.content
    
    # Parse the response and save
    new_flashcards = parse_flashcards(generated_text)
    save_all_flashcards(existing_flashcards + new_flashcards)
    
    return f"""FLASHCARDS GENERATED AND SAVED to {FLASHCARDS_FILE}

            {generated_text}

            TASK COMPLETE. Show these flashcards to the user. Do NOT generate more flashcards."""
            
@tool
def quiz_generator(topic: str, state: Annotated[dict, InjectedState]) -> str:
    """Generate multiple-choice quiz questions for a given topic.
    Quizzes are accumulated in a single JSON file for later review."""
    
    docs = retriever.invoke(topic)
    
    if not docs:
        return "I found no relevant information in the documents about the study material"
    
    # Read weak topics to focus the quiz on areas where the student struggles
    history = state["study_history"]
    weak = history.get("weak_topics", [])
    
    # Load existing quizzes to avoid duplicates
    existing_quizzes = load_all_quizzes()
    existing_questions = "\n".join([f"- {q['question']}" for q in existing_quizzes])
    
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are an expert tutor creating high-quality multiple-choice quizzes.

    Your task is to generate a quiz based ONLY on the provided context.

    CONTEXT:
    {context}

    STUDENT PROFILE:
    - Weak topics: {weak}

    ALREADY GENERATED QUESTIONS (do NOT repeat these):
    {existing_questions if existing_questions else "None yet"}

    INSTRUCTIONS:

    1. Generate 5 multiple-choice questions about the topic: "{topic}".

    2. Each question must:
        - Be clear, specific, and unambiguous
        - Test understanding, not just memorization

    3. For each question, provide exactly 4 options:
        - One correct answer
        - Three incorrect but plausible distractors

    4. If weak topics are provided, focus more on those areas.

    5. Output format MUST be exactly:

    Q1: <question>
    A) ...
    B) ...
    C) ...
    D) ...
    Correct answer: <letter>

    Q2: ...
        ...

    6. Do NOT include explanations.
    7. Do NOT use information outside the provided context.

    Generate the quiz now."""
    
    # The tool calls the LLM directly (without tool binding)
    response = generation_llm.invoke(prompt)
    generated_text = response.content
    
    # Parse the response and save (JSON keeps the correct answers)
    new_quizzes = parse_quiz(generated_text, topic)
    save_all_quizzes(existing_quizzes + new_quizzes)
    
    # Filter out "Correct answer" lines before showing to the user
    filtered_lines = [
        line for line in generated_text.split("\n") 
        if not line.strip().lower().startswith("correct answer")
    ]
    display_text = "\n".join(filtered_lines)
    
    return f"""QUIZ GENERATED AND SAVED to {QUIZZES_FILE}

    {display_text}

    TASK COMPLETE. Show this quiz to the user. Do NOT reveal correct answers. Do NOT generate more quizzes."""
