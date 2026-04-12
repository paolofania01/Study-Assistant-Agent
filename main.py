# main.py — Entry point of the Study Assistant Agent.
# Builds the LangGraph graph, connects all nodes, and runs the main conversation loop.

from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import router, should_continue, call_llm
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from tools import retriever_tool, flashcard_generator, quiz_generator

tools = [retriever_tool, flashcard_generator, quiz_generator]

# We use ToolNode instead of a custom execute_tools function because our tools use InjectedState.
# ToolNode automatically injects the graph's state into tools that require it.
tool_node = ToolNode(tools)

# --- Graph Construction ---
graph = StateGraph(AgentState)

graph.add_node("router", router)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", tool_node)

# If the LLM made tool calls, execute them. Otherwise, the response is final.
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", 
     False: END}
)

# router → llm → (tools → llm)* → END
graph.add_edge("router", "llm")
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("router")

study_agent = graph.compile()


def studing_agent():
    print("\n=== STUDY AGENT ===")
    
    # Initialize study history — persists across the session
    study_history = {
        "topics_covered": [],
        "correct_answers": 0,
        "wrong_answers": 0,
        "weak_topics": []
    }

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        messages = [HumanMessage(content=user_input)]

        result = study_agent.invoke({
            "messages": messages,
            "study_history": study_history,
            "current_mode": "chat"
        })

        # Track the topic for study history
        study_history["topics_covered"].append(user_input[:50])
        
        print("\n=== ANSWER ===")
        print(result['messages'][-1].content)


studing_agent()