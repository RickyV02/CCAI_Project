from typing import TypedDict

class AgentState(TypedDict):
    """
    Rappresenta lo stato del grafo.
    """
    user_input: str #Input di partenza dell'utente
    reasoning_trace: list[str] # Ogni volta che un nodo del grafo viene eseguito, aggiungiamo una stringa che rappresenta l'azione eseguita e il risultato ottenuto (Explainability)
    tool_outputs: dict[str, any] # Per ogni tool utilizzato, memorizziamo il suo output
    kg_summaries: str
    planning_information: dict[str, any] # Risultati del planner node
    draft_post: str #L'articolo generato fino a quel momento
    human_feedback: str # Feedback dell'utente sulla bozza
    post_type: str # Tipo di post da generare
    post_history: list[dict] # Storico dei post generati
