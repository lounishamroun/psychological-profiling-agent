"""
State schema for the interrogation graph.
"""

from typing import TypedDict, Annotated
import operator


class InterrogationState(TypedDict):
    # Case & suspect data (loaded once at the start, never changes)
    case_data: dict
    suspect_profile: dict

    # Conversation (grows each turn via operator.add)
    conversation_history: Annotated[list, operator.add]

    # RAG results (replaced each turn with fresh retrieval)
    retrieved_context: list

    # Current turn data (overwritten each turn)
    last_question: str
    last_answer: str

    # Profiler output for the current turn (overwritten each turn)
    profiler_output: dict

    # Full profiler history (grows each turn via operator.add)
    profiler_history: Annotated[list, operator.add]

    # Turn tracking
    turn: int
    max_turns: int

    # Final report (written once at the end)
    final_report: str
