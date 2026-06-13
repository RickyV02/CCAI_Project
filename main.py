import config
import os
import uuid
from agent_graph import graph
from langgraph.types import Command
from tools import kg_manager

def validate_env():
    """Verifica che tutte le variabili d'ambiente necessarie siano presenti."""
    required = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        print(f"  Variabili d'ambiente mancanti: {', '.join(missing)}")
        print("   Controlla il file .env nella directory del progetto.")
        return False

    optional = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    missing_optional = [var for var in optional if not os.getenv(var)]
    if missing_optional:
        print(f"  Variabili Neo4j non trovate: {', '.join(missing_optional)}")
        print("   Il Knowledge Graph potrebbe non funzionare correttamente.\n")

    return True


def print_reasoning_trace(trace: list[dict]):
    """Stampa la reasoning trace ReAct in formato strutturato."""
    print("\n" + "=" * 50)
    print(" 🧠 TRACCIA DEL RAGIONAMENTO (ReAct)")
    print("=" * 50)

    for i, step in enumerate(trace):
        if isinstance(step, dict):
            node = step.get('node', '?')
            thought = step.get('thought', '')
            action = step.get('action', '')
            observation = step.get('observation', '')

            print(f"\n--- Step {i+1} [{node}] ---")
            print(f"  💭 THOUGHT:     {thought}")
            print(f"  ⚡ ACTION:      {action}")
            if observation:
                print(f"  👁️  OBSERVATION: {observation[:200]}{'...' if len(str(observation)) > 200 else ''}")
        else:
            print(f"  → {step}")


def print_menu():
    """Stampa il menu delle opzioni per il feedback umano."""
    print("\n" + "=" * 50)
    print(" Cosa vuoi fare?")
    print("=" * 50)
    print("  [approve]          ✅ Approva e salva nel KG e RAG")
    print("  [cerca più info]   🔍 La review è generica, cerca più dettagli")
    print("  [cambia gioco]     🔄 Voglio recensire un altro gioco")
    print("  [testo libero]     ✏️  Scrivi le modifiche che vuoi alla review")
    print()


def main():
    if not validate_env():
        return

    print("=" * 50)
    print(" 🎮 Agentic AI — Blogger Videoludico")
    print(" 📝 Modalità: Solo Recensioni")
    print("=" * 50)

    # Il main interroga Neo4j SOLO PER MOSTRARE L'INTERFACCIA GRAFICA TESTUALE
    print("\n⏳ Controllo memoria di sistema in Neo4j...")
    active_plan = kg_manager.get_active_plan_status()

    if active_plan and active_plan["next_game"]:
        print("\n 📅 PIANO EDITORIALE IN CORSO:")
        for item in active_plan["status"]:
            check = "✅" if item["done"] else "⏳"
            print(f"  {check} {item['game']}")
        print(f"\n💡 Prossimo articolo in coda: '{active_plan['next_game']}'")
        print("👉 (Puoi scrivermi 'ok procedi', chiedermi un altro gioco bonus, o 'suggerisci un nuovo piano')")
    else:
        if active_plan:
            print("\n✅ Hai completato tutti i giochi del piano precedente!")
        print("\n👉 Cosa facciamo oggi? (Scrivi un gioco o chiedimi un suggerimento)")

    user_input = input("\n👤 Tu: ").strip()

    if not user_input and active_plan and active_plan["next_game"]:
        user_input = "ok, procedi"
    elif not user_input:
        print("Input vuoto. Uscita.")
        return

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state = {
        "user_input": user_input,
        "reasoning_trace": [],
        "tool_outputs": {},
        "kg_context": "",
        "planning_information": {},
        "research_summary": "",
        "draft_post": "",
        "human_feedback": "",
        "quality_passed": False,
        "revision_count": 0,
    }

    print("\n⏳ Esecuzione in corso...\n")

    try:
        result = graph.invoke(initial_state, config)
    except Exception as e:
        print(f"❌ Errore durante l'esecuzione del grafo: {e}")
        import traceback
        traceback.print_exc()
        return

    while True:
        snapshot = graph.get_state(config)

        if not snapshot.next:
            print("\n✅ Post approvato e salvato nei Database (Neo4j + ChromaDB)!")
            print_reasoning_trace(result.get('reasoning_trace', []))
            break

        if snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    for intr in task.interrupts:
                        interrupt_value = intr.value
                        if isinstance(interrupt_value, dict):
                            interrupt_type = interrupt_value.get("type", "")

                            if interrupt_type == "topic_suggestion":
                                print("\n" + "=" * 50)
                                print(" 💡 SUGGERIMENTO DELL'AGENTE")
                                print("=" * 50)
                                print(interrupt_value.get("message", ""))
                                print()
                                user_response = input("👤 La tua risposta: ").strip()
                                if not user_response:
                                    user_response = "Procedi pure con la proposta"
                                result = graph.invoke(
                                    Command(resume=user_response), config
                                )
                                continue

                            elif interrupt_type == "existing_review_warning":
                                print("\n" + "=" * 50)
                                print(" ⚠️  ATTENZIONE — REVIEW GIÀ ESISTENTE")
                                print("=" * 50)
                                print(interrupt_value.get("message", ""))
                                print()
                                user_response = input("👤 La tua risposta: ").strip()
                                if not user_response:
                                    user_response = "Procedi pure con la proposta"
                                result = graph.invoke(
                                    Command(resume=user_response), config
                                )
                                continue

                        draft = result.get('draft_post', '') if isinstance(result, dict) else ''

                        if not draft:
                            current_state = snapshot.values
                            draft = current_state.get('draft_post', '')

                        if draft:
                            print("\n" + "=" * 50)
                            print(" 📝 BOZZA REVIEW — RIVEDI IL POST")
                            print("=" * 50)
                            print(draft)

                        print_menu()
                        feedback = input("👤 Il tuo feedback: ").strip()

                        if not feedback:
                            feedback = "La recensione è perfetta, approvo!"
                            continue

                        result = graph.invoke(
                            Command(resume=feedback), config
                        )


if __name__ == "__main__":
    main()
