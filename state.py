import operator
from typing import Any, Annotated
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Stato condiviso tra tutti i nodi del grafo.
    """
    user_input: str                                        # Input dell'utente (gioco o richiesta suggerimento)
    reasoning_trace: Annotated[list[dict], operator.add]   # Trace strutturata Thought/Action/Observation
    tool_outputs: dict[str, Any]                           # Output grezzi dei tool {tool_name: [results]}
    kg_context: str                                        # Summaries dal Knowledge Graph
    planning_information: dict[str, Any]                   # Piano editoriale dal planner
    research_summary: str                                  # Estrazione strutturata dal summarizer
    draft_post: str                                        # Bozza della review
    human_feedback: str                                    # Feedback dell'utente
    quality_passed: bool                                   # Risultato quality check
    revision_count: int                                    # Contatore revisioni automatiche
