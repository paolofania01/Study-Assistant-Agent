# state.py — Defines the agent's state structure.
# Contains the data structures that flow through the graph:
# messages, study history, and current mode.

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add as add_messages

class StudyHistory(TypedDict):
    
    topics_covered: list[str]   # topics already studied
    correct_answers: int        # total correct answers
    wrong_answers: int          # total incorrect answers
    weak_topics: list[str]      # topics where you make the most mistakes
    
class AgentState(TypedDict):
    
    # add_messages ensures that new messages are appended to the existing list
    # instead of overwriting it
    messages: Annotated[Sequence[BaseMessage], add_messages]
    study_history: StudyHistory
    current_mode: str                   # "chat", "flashcard", "quiz" 