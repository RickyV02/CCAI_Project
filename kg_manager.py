class KGManager:
    """
    Gestore del Knowledge Graph (Mock).
    """
    def __init__(self):
        pass

    def query(self, query_str: str) -> str:
        return f"Mock KG Result for: {query_str}"

    def update(self, data: str) -> bool:
        print(f"[KGManager] Updater: {data[:50]}...")
        return True
