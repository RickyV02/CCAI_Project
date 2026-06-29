from tools import search_tool
from ml_manager import MLEvaluator

ml = MLEvaluator()

def stress_test_search(query: str):
    print(f"\n--- TEST: {query} ---")

    risultati = search_tool.invoke(query)

    print(f"\nRisultato finale del tool: {risultati}")

if __name__ == "__main__":
    test_queries = [
        "acquista collector edition Hollow Knight Silksong prezzo",
        "offerte Black Friday PS5 giochi",
        "recensione Elden Ring e dove comprarlo al miglior prezzo"
    ]

    for q in test_queries:
        stress_test_search(q)
        print("\n" + "="*50)
