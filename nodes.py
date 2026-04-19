# nodes.py — Defines the graph nodes: router, LLM caller, and continuation check.
# The router determines the mode (chat/flashcard/quiz) based on the user's message.
# The LLM node calls the model with the system prompt and current mode context.

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from state import AgentState
from tools import retriever_tool, flashcard_generator, quiz_generator

# model = ChatOllama(model="llama3.1:8b", temperature=0)
model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# model = ChatOllama(model="qwen2.5:7b", temperature=0)

tools = [retriever_tool, flashcard_generator, quiz_generator]

# Dictionary to look up tools by name when executing tool calls
tools_dict = {t.name: t for t in tools}

# Bind tools to the model so the LLM knows which tools are available
llm = model.bind_tools(tools)

system_prompt = """
    You are an intelligent study assistant designed to help students learn effectively.

    You operate in three main modes:
    1. CHAT → use retriever_tool
    2. FLASHCARD → use flashcard_generator
    3. QUIZ → use quiz_generator

    Your task is to understand the user's intent and ALWAYS use the most appropriate available tool.

    Guidelines:

    - CHAT mode:
        RULE: You MUST call retriever_tool for EVERY user question, no exceptions.
        NEVER answer from your own knowledge.
        NEVER answer based on what you remember.
        ALWAYS call retriever_tool first, then use ONLY the retrieved context to answer.
        If the retrieved context does not contain the answer, reply: "Non ho trovato informazioni nei documenti su questo argomento."

    - FLASHCARD mode:
        Use flashcard_generator ONCE per user request.
        When the tool returns flashcards, display them to the user and STOP.
        Do NOT call flashcard_generator multiple times for the same request.

    - QUIZ mode:
        Use quiz_generator ONCE per user request.
        When the tool returns quiz questions, display them to the user and STOP.
        Do NOT call quiz_generator multiple times for the same request.

    Important rules:

    - You MUST always select and use the correct tool for the requested mode.
    - If the user's intent is not explicit, infer it from context.
    - Maintain a clear, educational, and student-friendly tone.
    - Do not just provide answers—guide the student through the reasoning process when appropriate.
    - Adapt the difficulty based on the user's level when possible.

    Goal:
    Help the student understand, memorize, and test their knowledge as effectively as possible. """


def router(state: AgentState):
    """Determines the current mode based on keywords in the user's message."""
    
    flash_list = ["flashcard", "flash", "card"]
    quiz_list = ["quiz", "domanda", "test"]
    
    message = state["messages"][-1].content.lower()
    
    if any(word in message for word in flash_list):
        mode = "flashcard"
    elif any(word in message for word in quiz_list):
        mode = "quiz"
    else:
        mode = "chat"

    return {"current_mode": mode}


def call_llm(state: AgentState):
    """Calls the LLM with the full conversation history and the system prompt."""
    
    # Copy messages to avoid modifying the original state
    messages = list(state['messages'])
    
    # Append the current mode to the system prompt so the LLM knows which tool to use
    mode = state.get("current_mode", "chat")
    context_info = f"\nCurrent mode: {mode}"
    
    messages = [SystemMessage(content=system_prompt + context_info)] + messages
    
    # The LLM response can be either a direct text answer or a tool call
    message = llm.invoke(messages)
    
    return {'messages': [message]}


def should_continue(state: AgentState):
    """Checks if the LLM's response contains tool calls.
    Returns True if tools need to be executed, False if the response is final."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
