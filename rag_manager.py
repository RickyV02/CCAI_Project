class RAGManager:
    """
    Gestore del sistema K-RAG (Mock)
    """
    def __init__(self):
        pass

    def retrieve(self, query: str) -> str:
        return f"Mock RAG Document related to {query}"
