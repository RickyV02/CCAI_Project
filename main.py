import os
import uuid
from dotenv import load_dotenv
from agent_graph import graph
from langgraph.types import Command

def main():
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("ATTENZIONE: Variabile GROQ_API_KEY non trovata nel file .env")

    print("====================================")
    print(" Avvio Agentic AI - Blogger Videoludico")
    print("====================================\n")

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "user_input": "Vogliamo fare un nuovo articolo su Elden Ring. Controlla cosa abbiamo già pubblicato in passato e proponi un format diverso per non ripeterci.",
        "reasoning_trace": [],
        "tool_outputs": {},
        "kg_summaries": "",
        "planning_information": {},
        "draft_post": "",
        "human_feedback": "",
        "post_type": "",
        "post_history": []
    }

    print("Esecuzione in corso...\n")

    # Prima esecuzione — si ferma prima di human_review
    try:
        result = graph.invoke(initial_state, config)
    except Exception as e:
        print(f"Errore durante l'esecuzione del grafo: {e}")
        return

    # Loop feedback
    while True:
        draft = result.get('draft_post', '')
        print("\n====================================")
        print(" BOZZA GENERATA — RIVEDI IL POST")
        print("====================================")
        print(draft)
        print("\n====================================")
        print("\nCosa vuoi fare?")
        print("[approve] Approva il post")
        print("[reject]  Rifiuta e rigenera completamente")
        print("[altro]   Scrivi le tue modifiche")

        feedback = input("\nIl tuo feedback: ").strip()

        if not feedback:
            print("Feedback vuoto, riprova.")
            continue

        if feedback.lower() == "approve":
            # Riprendi con approve → va a kg_updater → fine
            final_result = graph.invoke(
                Command(resume={"human_feedback": "approve"}),
                config
            )
            print("\n====================================")
            print(" Post approvato e salvato!")
            print("====================================")
            print("\nTraccia del ragionamento:")
            for step in final_result.get('reasoning_trace', []):
                print(f" -> {step}")
            break
        else:
            # Feedback testuale o reject → writer riscrive
            result = graph.invoke(
                Command(resume={"human_feedback": feedback}),
                config
            )

if __name__ == "__main__":
    main()
