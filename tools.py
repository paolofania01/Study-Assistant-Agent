# tools.py — Defines the tools available to the LLM agent.
# Each tool uses InjectedState to access the graph's state (study history, etc.)
# and the shared retriever to search through the study documents.

from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from vectorstore import get_retriever

# Created once and shared across all tools to avoid reloading PDFs on every call
retriever = get_retriever()

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
    and avoids repeating already covered topics from the study history."""
    
    docs = retriever.invoke(topic)
    
    if not docs:
        return "I found no relevant information in the documents about the study material"
    
    # Read study history to avoid generating flashcards on already covered topics
    history = state["study_history"]
    covered = history.get("topics_covered", [])

    context = "\n".join([doc.page_content for doc in docs])
    
    # The prompt instructs the LLM to generate flashcards based ONLY on the retrieved context
    # to minimize hallucinations
    prompt = f"""
    You are an expert study assistant that creates high-quality flashcards.

    Your task is to generate flashcards based ONLY on the provided context.

    MAIN TOPIC:
    {topic}

    CONTEXT:
    {context}

    ALREADY COVERED TOPICS:
    {covered}

    INSTRUCTIONS:
    - Generate clear and concise flashcards (question-answer format)
    - Focus on concepts relevant to the MAIN TOPIC
    - Avoid repeating topics that are already covered
    - Each question should test understanding, not just memorization
    - Answers should be precise but complete
    - Do not include information not present in the context

    OUTPUT FORMAT:
    Q: <question>
    A: <answer>

    Generate 5 to 10 flashcards.
    """
    return prompt

@tool
def quiz_generator(topic: str, state: Annotated[dict, InjectedState]) -> str:
    """Generate multiple-choice quiz questions for a given topic.
    The tool focuses more on weak topics from the student's study history
    to reinforce areas that need improvement."""
    
    docs = retriever.invoke(topic)
    
    if not docs:
        return "I found no relevant information in the documents about the study material"
    
    # Read weak topics to focus the quiz on areas where the student struggles
    history = state["study_history"]
    weak = history.get("weak_topics", [])
    
    context = "\n".join([doc.page_content for doc in docs])
    
    prompt = f"""You are an expert tutor creating high-quality multiple-choice quizzes.

    Your task is to generate a quiz based ONLY on the provided context.

    CONTEXT:
    {context}

    STUDENT PROFILE:
    - Weak topics: {weak}

    INSTRUCTIONS:

    1. Generate 5 multiple-choice questions about the topic: "{topic}".

    2. Each question must:
        - Be clear, specific, and unambiguous
        - Test understanding, not just memorization (include reasoning when possible)

    3. For each question, provide exactly 4 options:
        - One correct answer
        - Three incorrect but plausible distractors (avoid obviously wrong answers)

    4. Make distractors:
        - Conceptually close to the correct answer
        - Based on common mistakes or misconceptions

    5. If weak topics are provided:
        - Focus more on those areas
        - Increase difficulty slightly on those concepts

    6. Do NOT include explanations.

    7. Output format MUST be:

    Q1: <question>
    A) ...
    B) ...
    C) ...
    D) ...
    Correct answer: <letter>

    Q2: ...
        ...

    8. Do NOT use information outside the provided context.

    9. Avoid repetition and ensure diversity in questions.

    Generate the quiz now."""
    
    return prompt
