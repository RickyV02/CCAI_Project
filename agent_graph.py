import os
import json
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, Any, Literal

from state import AgentState
from schemas import (
    PlannerIntent, PlannerOutput, GameResearchExtraction,
    QualityVerdict, PostEntities, FeedbackRouting, PlanApprovalRouting
)
from helpers import create_react_entry, format_extraction_for_writer, truncate_text
from tools import (
    search_tool, rag_retrieval_tool, knowledge_graph_tool,
    deep_read_article, youtube_transcript_fetcher,
    kg_manager, rag_manager
)

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.5"))
POST_LENGTH_GUIDANCE = os.getenv("POST_LENGTH_GUIDANCE", "1000 parole circa")  # Indicazione di lunghezza per il writer
MAX_QUALITY_RETRIES = int(os.getenv("MAX_QUALITY_RETRIES", "1"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
MAX_RESEARCHER_ITERATIONS = int(os.getenv("MAX_RESEARCHER_ITERATIONS", "5")) # Previene ricorsione infinita durante la fase di ricerca (LangGraph ha un suo limite nativo, ma lo abbassiamo noi a mano per evitare loop e consumo di token API eccessivo)
# Potremmo anche dare dei limiti nel system prompt del researcher, ma preferisco lasciare più libertà all'LLM e intervenire solo se vediamo che tende ad abusare di iterazioni o tool calls (e sprecare token API)

# LLM principale
llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# ============================================================
# NODO 1: PLANNER
# ============================================================
def planner_node(state: AgentState) -> Dict[str, Any]:
    """Pianifica la review: gestisce sia input specifico che richiesta di suggerimento."""
    print("--- [PlannerNode] Analisi del topic e pianificazione ---")

    user_in = state['user_input'].strip()
    reasoning = []

    # Step 1: Capire l'intento dell'utente
    intent_messages = [
        ("system", "Sei un assistente AI. Il tuo compito è classificare l'intento dell'utente.\n"
                   "Scegli 'suggest' se l'utente chiede un piano editoriale o un suggerimento.\n"
                   "Scegli 'specific' SOLO E UNICAMENTE se l'utente digita il nome di un videogioco specifico da recensire."),
        ("user", f"L'utente dice: '{user_in}'. Ha specificato un gioco preciso da recensire o vuole un suggerimento?")
    ]
    intent_llm = llm.with_structured_output(PlannerIntent)
    intent = intent_llm.invoke(intent_messages)

    planner_llm = llm.with_structured_output(PlannerOutput)

    # ==========================================
    # BINARIO A: MODALITÀ SUGGERIMENTO
    # ==========================================
    if intent.mode == "suggest":
        all_games_kg = kg_manager.query_all_games()
        reasoning.append(create_react_entry("planner", "L'utente vuole un suggerimento, interrogo il KG per trovare i giochi meno coperti", "kg_manager.query_all_games()", truncate_text(str(all_games_kg), 300)))

        suggest_messages = [
            ("system", "Sei un Editorial Director per un blog di videogiochi. "
                       "Devi generare una SEQUENZA di 3 prossime recensioni. Giustifica l'ordine logico. Infine, estrai il PRIMO gioco della sequenza per scriverlo OGGI.\n"
                       "🚨 REGOLA SCELTA GIOCHI: Cerca giochi nel Knowledge Graph che soddisfino la richiesta. SE NON CI SONO abbastanza giochi adatti nel KG, usa la tua conoscenza generale per proporre altri titoli famosi reali. Se le informazioni del KG non sono sufficienti per soddisfare la richiesta, usa sempre la tua conoscenza generale (ad esempio, se un gioco nel KG non ha un genere associato, consiglialo SOLO SE TU NON CONOSCI ALTERNATIVE MIGLIORE E CERTE CHE RISPETTANO LA RICHIESTA).\n"
                       "🚨 RIDONDANZA: Scegli giochi che rispettino la richiesta dell'utente, ma se un gioco è già stato recensito, evitalo per evitare la ridondanza (se non trovi giochi che rispettino la richiesta dal KG, usa sempre la tua conoscenza generale, suggerisci giochi già recensiti SOLO SE non c'è nessun'altro gioco che rispetti la richiesta).\n"
                       "🚨 REGOLA SULL'ANGOLO: Se il gioco NON è mai stato recensito nel blog, l'angolo DEVE essere 'Recensione Completa e Generale'. Se è già stato recensito, scegli un angolo inedito.\n"
                       "🚨 REGOLA JSON: NON usare MAI il carattere backslash (\\) per fare l'escape di apostrofi."),
            ("user", f"L'utente ha fatto questa richiesta specifica: '{user_in}'. Tieni OBBLIGATORIAMENTE conto di questa richiesta per scegliere i giochi.\n\nEcco tutti i giochi nel Knowledge Graph e le review già scritte:\n{all_games_kg}")
        ]
        plan_result = planner_llm.invoke(suggest_messages)

        reasoning.append(create_react_entry(
            "planner",
            f"RAGIONAMENTO DEL DIRETTORE:\n{plan_result.reasoning_process}",
            "llm.with_structured_output(PlannerOutput)",
            f"Sequenza generata: {plan_result.sequence_of_posts}"
        ))

        user_decision = interrupt({
            "type": "topic_suggestion",
            "sequence": plan_result.sequence_of_posts,
            "suggestion": plan_result.suggested_game,
            "angle": plan_result.review_angle,
            "justification": plan_result.justification,
            "message": f"🗓️ CALENDARIO EDITORIALE PROPOSTO:\n"
                       f"Sequenza: {', '.join(plan_result.sequence_of_posts)}\n"
                       f"Motivo: {plan_result.justification}\n\n"
                       f"💡 OGGI RECENSIREMO: '{plan_result.suggested_game}' (Focus: {plan_result.review_angle})\n\n"
                       f"👉 Rispondi 'ok' per confermare tutto.\n"
                       f"👉 Altrimenti, scrivi le tue modifiche (es. 'Fai prima Hades')."
        })

        user_response = str(user_decision).strip()
        approval = llm.with_structured_output(PlanApprovalRouting).invoke([
            ("system", "Classifica la risposta dell'utente alla proposta del calendario editoriale. Se accetta/conferma, scegli 'approve'. Se chiede modifiche, scegli 'modify'."),
            ("user", f"Risposta utente: '{user_response}'")
        ])

        if approval.decision == "modify":
            feedback_messages = [
                ("system", "Sei un Editorial Director. Modifica il tuo piano editoriale precedente seguendo ALLA LETTERA le nuove istruzioni dell'utente. 🚨 REGOLA JSON: NON usare MAI il carattere backslash (\\) per fare l'escape di apostrofi."),
                ("user", f"Richiesta originale: '{user_in}'\nPiano precedente:\nSequenza: {plan_result.sequence_of_posts}\nGioco di oggi: {plan_result.suggested_game}\n\nFeedback utente: '{user_response}'")
            ]
            plan_result = planner_llm.invoke(feedback_messages)
            reasoning.append(create_react_entry("planner", f"RAGIONAMENTO POST-FEEDBACK:\n{plan_result.reasoning_process}", "Modifica Inline", f"Nuovo topic: {plan_result.suggested_game}"))

            print("\n" + "=" * 50)
            print(" 🗓️ NUOVO PIANO EDITORIALE (Modificato)")
            print("=" * 50)
            print(f"Sequenza: {', '.join(plan_result.sequence_of_posts)}")
            print(f"💡 OGGI RECENSIREMO: '{plan_result.suggested_game}' (Focus: {plan_result.review_angle})\n")
        else:
            reasoning.append(create_react_entry("planner", "Piano confermato dall'utente", "LLM Routing", f"Decisione: {approval.decision}"))

        topic = plan_result.suggested_game

    # ==========================================
    # BINARIO B: MODALITÀ GIOCO SPECIFICO
    # ==========================================
    else:
        topic = intent.game_name or user_in
        existing_reviews = kg_manager.check_existing_reviews(topic)

        reasoning.append(create_react_entry(
            "planner", f"L'utente vuole recensire '{topic}'. Interrogo il KG.",
            f"kg_manager.check_existing_reviews('{topic}')", truncate_text(str(existing_reviews), 300)
        ))

        existing_str = str(existing_reviews)
        has_existing = (
            existing_str and "Nessun" not in existing_str and "Errore" not in existing_str and
            '"Numero_Review": 0' not in existing_str and "'text': '[]'" not in existing_str
        )

        if has_existing:
            user_decision = interrupt({
                "type": "existing_review_warning",
                "existing_reviews": existing_str,
                "message": f"⚠️ Esistono già review su '{topic}':\n{existing_reviews}\n\n"
                           f"Vuoi continuare con un angolo diverso? Rispondi 'ok' o scrivi un nuovo gioco."
            })
            user_response = str(user_decision).strip()

            approval = llm.with_structured_output(PlanApprovalRouting).invoke([
                ("system", "L'utente è stato avvisato che esiste già una recensione per il gioco. Classifica la sua risposta. Scegli 'approve' se vuole procedere comunque. Scegli 'modify' se decide di CAMBIARE gioco."),
                ("user", f"Risposta utente: '{user_response}'")
            ])

            if approval.decision == "modify" and approval.new_game:
                topic = approval.new_game
                existing_reviews = kg_manager.check_existing_reviews(topic)
                existing_str = str(existing_reviews)
                has_existing = (
                    existing_str and
                    "Nessun" not in existing_str and
                    "Errore" not in existing_str and
                    '"Numero_Review": 0' not in existing_str and
                    "'text': '[]'" not in existing_str and
                    '"text": "[]"' not in existing_str
                )

                reasoning.append(create_react_entry(
                "planner", f"Review esistente gestita via LLM. Utente ha risposto: '{user_response}'",
                "Routing Semantico", f"Topic finale: {topic}"
                ))

                print(f"\n💡 CAMBIO GIOCO ACCETTATO. Oggi recensiremo: '{topic}'\n")

        angle_instruction = (
            "Il gioco ha GIÀ delle recensioni passate nel nostro database. Scegli un ANGOLO INEDITO (es. focus su una meccanica, una boss fight)."
            if has_existing else
            "Questa è la PRIMA recensione per questo gioco. L'angolo DEVE essere 'Recensione Completa e Generale'. Il piano deve coprire lore/storia, gameplay, comparto tecnico e tutte le altre informazioni utili che trovi."
        )

        plan_result = planner_llm.invoke([
            ("system", f"Sei un Editorial Director per un blog di videogiochi.\n{angle_instruction}"),
            ("user", f"Topic: '{topic}'\nReview esistenti:\n{existing_reviews}\nGenera un piano editoriale.")
        ])

        reasoning.append(create_react_entry("planner", f"RAGIONAMENTO DEL DIRETTORE:\n{plan_result.reasoning_process}", "llm.with_structured_output(PlannerOutput)", f"Piano generato: angolo '{plan_result.review_angle}'"))

    kg_context = kg_manager.query(topic)

    return {
        "reasoning_trace": reasoning,
        "planning_information": {
            "plan": getattr(plan_result, 'plan', ''),
            "sequence": getattr(plan_result, 'sequence_of_posts', []),
            "review_angle": getattr(plan_result, 'review_angle', ''),
            "justification": getattr(plan_result, 'justification', ''),
            "topic": topic
        },
        "kg_context": str(kg_context),
        "user_input": topic
    }

# ============================================================
# NODO 2: RESEARCHER
# ============================================================
def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Ricerca informazioni in 2 fasi: deterministica (web+KG obbligatori) + agentica (ReAct manuale)."""
    print("--- [ResearcherNode] Ricerca informazioni ---")

    topic = state['user_input']
    kg_context = state.get('kg_context', '')
    feedback = state.get('human_feedback', '')
    reasoning = []
    tool_outputs = {}

    # ═══════════════════════════════════════════
    # FASE 1: DETERMINISTICA (sempre eseguita)
    # ═══════════════════════════════════════════

    # 1A: Web search OBBLIGATORIA
    search_query = f"{topic} recensione"
    print(f"    [Fase 1] Ricerca web: '{search_query}'")
    web_result = search_tool.invoke({"query": search_query})
    tool_outputs["search_tool"] = [str(web_result)]

    # 1B: KG query per K-RAG
    print(f"    [Fase 1] Query KG per entità K-RAG: '{topic}'")
    kg_entities = kg_manager.get_entities_for_krag(topic)
    tool_outputs["knowledge_graph_tool"] = [str(kg_entities)]
    reasoning.append(create_react_entry(
        "researcher", "Interrogo il KG per entità da usare come query RAG espanse (K-RAG)",
        f"kg_manager.get_entities_for_krag('{topic}')", truncate_text(str(kg_entities), 300)
    ))

    # 1C: RAG retrieval con query espanse dal KG (K-RAG)
    krag_queries = _build_krag_queries(topic, str(kg_entities))
    for q in krag_queries:
        print(f"    [Fase 1] K-RAG query: '{q}'")
        rag_result = rag_retrieval_tool.invoke({"query": q})
        tool_outputs.setdefault("rag_retrieval_tool", []).append(str(rag_result))
        reasoning.append(create_react_entry(
            "researcher", f"Query RAG espansa dal KG (K-RAG): '{q}'",
            f"rag_retrieval_tool('{q}')", truncate_text(str(rag_result), 200)
        ))

    # ═══════════════════════════════════════════
    # FASE 2: AGENTICA (Custom ReAct Loop)
    # ═══════════════════════════════════════════
    print("    [Fase 2] ReAct loop manuale per approfondimento...")
    optional_tools = [search_tool, rag_retrieval_tool, deep_read_article, youtube_transcript_fetcher, knowledge_graph_tool]
    llm_with_tools = llm.bind_tools(optional_tools)

    system_prompt = (
        f"Sei un instancabile ricercatore per un blog di videogiochi.\n"
        f"Devi approfondire '{topic}'. Contesto KG: {truncate_text(kg_context, 500)}\n"
        f"FEEDBACK DELL'UTENTE: '{feedback}'. Concentrati nel cercare informazioni per soddisfare questa richiesta!\n\n"
        f"🚨 REGOLA CHIAMATA TOOL: Invoca i tool usando ESATTAMENTE e SOLO il loro nome (es. 'search_tool'). È severamente vietato concatenare o fondere gli argomenti JSON direttamente nel nome del tool.\n"
        f"🚨 DIRETTIVA FONDAMENTALE SULLA RICERCA E I TOOL:\n"
        f"1. Cerca info solo sul GIOCO BASE ('{topic}'). Scarta URL su DLC, Mod, Remaster o Spinoff.\n"
        f"2. La semplice ricerca web ti dà solo dei riassunti. Quindi devi usarla se hai bisogno di una panoramica generale o di trovare nuove informazioni. Quando trovi un URL interessante, DEVI leggere l'articolo completo usando il tool `deep_read_article`. Cerca di approfondire SOLO articoli di testate giornalistiche, recensioni, wiki o guide (es. IGN, Fandom, Wikipedia, Everyeye, Multiplayer).\n"
        f"3. Passa al tool l'URL che hai trovato. Inizia leggendo i primi paragrafi (es. offset=0, limit=5). Se l'articolo è lungo e ti servono altre info, richiama il tool aumentando l'offset.\n"
        f"4. Se hai dubbi su personaggi, boss o meccaniche già noti nel nostro database, usa `knowledge_graph_tool` per un fact-checking.\n"
        f"5. Se trovi un video interessante, usa `youtube_transcript_fetcher` per estrarre la trascrizione e cercare info rilevanti al suo interno.\n\n"
        f"PENSA AD ALTA VOCE: Prima di usare qualsiasi tool, scrivi sempre una frase spiegando quale informazione ti manca e PERCHÉ stai per usare quel tool."
        f"Esplora finché non hai una comprensione profonda della lore e del gameplay, poi fermati e fornisci un riassunto testuale."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Inizia la ricerca di approfondimento per {topic}."}
    ]

    tool_map = {t.name: t for t in optional_tools}

    for i in range(MAX_RESEARCHER_ITERATIONS):
        try:
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                final_thought = response.content.strip() if response.content else "Ho raccolto abbastanza informazioni, termino la ricerca."
                reasoning.append(create_react_entry(
                    "researcher", "Decisione finale", "STOP", final_thought
                ))
                break  # Fine ricerca, l'LLM ha deciso di fermarsi

            actual_thought = response.content.strip() if response.content else f"Analizzo il contesto e decido di usare {tool_name}"

            for tc in response.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]

                action_str = f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)[:200]})"

                reasoning.append(create_react_entry(
                    "researcher",
                    actual_thought,
                    action_str,
                    ""
                ))

                tool_result = ""
                try:
                    tool_func = tool_map[tool_name]
                    tool_result = str(tool_func.invoke(tool_args))
                except Exception as e:
                    tool_result = f"Errore tool {tool_name}: {e}"

                if reasoning and not reasoning[-1]["observation"]:
                    reasoning[-1]["observation"] = truncate_text(tool_result, 300)

                tool_outputs.setdefault(tool_name, []).append(tool_result)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tool_name, "content": tool_result})

        except Exception as e:
            reasoning.append(create_react_entry(
                "researcher", f"Errore nel ReAct loop: {e}",
                "llm_with_tools.invoke()", str(e)
            ))
            break

    reasoning.append(create_react_entry(
        "researcher", "Ricerca completata (Fase 1 + Fase 2)",
        "STOP", f"Tool usati: {list(tool_outputs.keys())}"
    ))

    return {"reasoning_trace": reasoning, "tool_outputs": tool_outputs}


def _build_krag_queries(topic: str, kg_entities_str: str) -> list[str]:
    """Costruisce query RAG espanse dalle entità del KG (pattern K-RAG)."""
    queries = [topic]  # Query base sempre presente

    # Estrai entità dal risultato KG
    kg_lower = kg_entities_str.lower()
    # Cerca pattern comuni nei risultati KG
    if "boss" in kg_lower:
        queries.append(f"{topic} boss difficoltà")
    if "meccaniche" in kg_lower or "mechanic" in kg_lower:
        queries.append(f"{topic} meccaniche gameplay")
    if "similar" in kg_lower or "simili" in kg_lower:
        queries.append(f"{topic} giochi simili confronto")

    return queries[:4]  # Max 4 query per non sovraccaricare


# ============================================================
# NODO 3: SUMMARIZER
# ============================================================
def summarizer_node(state: AgentState) -> Dict[str, Any]:
    print("--- [SummarizerNode] Estrazione strutturata e verifica fonti ---")

    tool_outputs = state.get('tool_outputs', {})
    topic = state['user_input']
    kg_context = state.get('kg_context', '')

    all_research = []
    for tool_name, results in tool_outputs.items():
        if isinstance(results, list):
            for r in results:
                if r and str(r).strip():
                    all_research.append(f"[{tool_name}]: {r}")
        elif results and str(results).strip():
            all_research.append(f"[{tool_name}]: {results}")

    research_text = truncate_text("\n\n".join(all_research), 15000)

    system_prompt = (
        f"Sei un analista di ricerca enciclopedico. Il tuo compito è leggere i dati forniti ed ESTRARRE ogni singolo dettaglio rilevante.\n"
        f"🚨 REGOLA: NON FARE RIASSUNTI BREVI. Mantieni ogni singolo dettaglio specifico sul gioco, aneddoto sulla trama e modalità di gameplay, ecc... che trovi nel testo originale.\n"
        f"Filtra solo la spazzatura web (menu, cookie, pubblicità). Genera paragrafi ricchi di dettagli, non elenchi puntati o riassunti brevi.\n"
        f"Valuta la credibilità delle fonti: 'alta' (testate note, es. IGN, Everyeye, Multiplayer.it, GameSpot, GameInformer), 'media' (blog), 'bassa' (sconosciute)."
        f"Se fonti diverse dicono cose opposte, fidati di quella con credibilità più alta.\n\n"
        f"Se ci sono informazioni inutili e/o fuorvinati e/o fuori contesto, segnalalo compilando l'apposito campo 'fact_check_notes'. Allo stesso modo, se trovi contraddizioni tra le fonti web e/o con il KG, evidenziale chiaramente sempre nella sezione 'fact_check_notes' alla fine dell'estrazione.\n\n"
        f"Se non ci sono discrepanze, lascia il campo 'fact_check_notes' vuoto.\n\n"
        f"CONTESTO KNOWLEDGE GRAPH (usalo per fact-checking):\n{truncate_text(kg_context, 1000)}"
    )

    user_prompt = f"Estrai tutte le informazioni dal seguente materiale di ricerca su '{topic}':\n\n{research_text}"

    messages = [
        ("system", system_prompt),
        ("user", user_prompt)
    ]

    extraction_llm = llm.with_structured_output(GameResearchExtraction)
    try:
        extraction = extraction_llm.invoke(messages)
    except Exception as e:
        print(f"    Errore estrazione strutturata: {e}")
        extraction = GameResearchExtraction(
            lore_and_story_details="Informazioni non disponibili a causa di un errore di estrazione.",
            gameplay_and_mechanics_deep_dive="Nessun dato estratto.",
            sources=[]
        )

    research_summary = format_extraction_for_writer(extraction)

    reasoning = [create_react_entry(
        "summarizer",
        f"Estratte info strutturate usando System/User messages.",
        "llm.with_structured_output(GameResearchExtraction)",
        truncate_text(research_summary, 300)
    )]

    return {"reasoning_trace": reasoning, "research_summary": research_summary}


# ============================================================
# NODO 4: WRITER
# ============================================================
def writer_node(state: AgentState) -> Dict[str, Any]:
    print("--- [WriterNode] Redazione della bozza ---")

    topic = state['user_input']
    research_summary = state.get('research_summary', '')
    kg_context = state.get('kg_context', '')
    plan = state.get('planning_information', {}).get('plan', '')
    review_angle = state.get('planning_information', {}).get('review_angle', '')
    human_feedback = state.get('human_feedback', '').strip()
    
    recent_posts = kg_manager.get_recent_posts(limit=3)

    system_prompt = f"""Sei un blogger esperto di videogiochi.
Devi scrivere una RECENSIONE di '{topic}'.
Focus editoriale suggerito: {review_angle}.

REGOLA ANTI-ALLUCINAZIONE E LIMITI SUL CONTENUTO:
- Parla ESCLUSIVAMENTE di '{topic}'.
- Scrivi usando ESCLUSIVAMENTE le informazioni presenti nel materiale di ricerca fornito sotto. Non inventare nulla.

LUNGHEZZA DELL'ARTICOLO:
L'articolo deve rispettare questa indicazione di lunghezza: {POST_LENGTH_GUIDANCE}. Per raggiungere questo obiettivo senza allungare il brodo, sviluppa i paragrafi in ESTREMA profondità, analizzando minuziosamente la lore, ogni singola meccanica e l'atmosfera.

RICCHEZZA DEI DETTAGLI (PER IL DATABASE DEL BLOG):
- Se li trovi tra le informazioni fornite, devi menzionare in modo naturale nel testo: il Genere del gioco, lo Studio di sviluppo, l'Anno di uscita e le Piattaforme disponibili.
- Arricchisci la recensione citando i nomi specifici di Boss, Personaggi chiave e Meccaniche di gioco scoperti durante la ricerca.
- 🚨 ATTENZIONE: Inserisci questi dati SOLO se sono presenti nel 'MATERIALE DI RICERCA' o nel 'CONTESTO DEL GIOCO'. Se un dato manca, NON inventarlo per nessun motivo.

CITAZIONI CROSS-POST E STILE BLOG:
- Usa il tono tipico di un blog. Se appropriato, nell'introduzione puoi salutare i lettori e fare riferimento agli ultimi articoli pubblicati (elencati in 'ULTIMI POST SUL BLOG') dicendo ad esempio: "Dopo avervi parlato di [Gioco Precedente], oggi ci dedicheremo... (ovviamente questo è solo un esempio!, cerca di essere creativo ma preciso in ciò che dici rispetto alla verità del blog e delle fonti)".
- Usa il 'CONTESTO DEL GIOCO' per menzionare giochi simili o vecchi capitoli dello stesso studio in modo naturale.

CITAZIONE FONTI ESTERNE:
Cita le fonti ALLA FINE dell'articolo in un'apposita sezione. Non citare nomi di siti o giornalisti nel mezzo del discorso.

MATERIALE DI RICERCA (Ricco di dettagli e lore):
{research_summary}

CONTESTO DEL GIOCO E GIOCHI SIMILI (Dal Knowledge Graph):
{kg_context}

ULTIMI POST SUL BLOG (Per introduzione e continuity):
{recent_posts}

PIANO EDITORIALE DA SEGUIRE:
{plan}
"""

    messages = [("system", system_prompt)]

    if human_feedback and human_feedback.lower() not in ["", "approve"]:
        user_message = (
            f"🚨 REVISIONE EDITORIALE OBBLIGATORIA 🚨\n"
            f"Il Direttore Responsabile ha rifiutato la bozza precedente e ti ha dato questo ordine diretto:\n"
            f"\"{human_feedback}\"\n\n"
            f"Riscrivi l'INTERA recensione obbedendo ciecamente a questo feedback. "
            f"Se ti ha chiesto una LINGUA DIVERSA (es. Inglese), l'intero testo generato DEVE essere in quella lingua. "
            f"Se ti ha chiesto di approfondire degli argomenti, fallo usando i dati di ricerca. "
            f"Formatta l'output in Markdown."
        )
    else:
        user_message = f"Scrivi la recensione completa e dettagliata di '{topic}' rispettando le regole."

    messages.append(("user", user_message))

    response = llm.invoke(messages)

    reasoning = [create_react_entry(
        "writer", f"Bozza review scritta per '{topic}'. Feedback umano integrato via User Message.",
        "llm.invoke(messages)", f"Lunghezza: {len(response.content.split())} parole"
    )]

    return {"reasoning_trace": reasoning, "draft_post": response.content}


# ============================================================
# NODO 5: QUALITY CHECK
# ============================================================
def quality_check_node(state: AgentState) -> Command[Literal["human_review", "writer"]]:
    """Valuta la qualità della review con LLM judge. Usa Command per routing atomico."""
    print("--- [QualityCheckNode] Valutazione qualità ---")

    draft = state.get('draft_post', '')
    revision_count = state.get('revision_count', 0)
    word_count = len(draft.split())

    system_prompt = (
        "Sei un Senior Editor spietato di un blog di videogiochi.\n"
        "Valuta se la bozza della recensione rispetta questi standard minimi:\n"
        "1. Contiene dettagli specifici (nomi boss, meccaniche, aree)?\n"
        "2. Cita le fonti in modo naturale?\n"
        "3. Ha un hook iniziale coinvolgente?\n"
        "4. Ha una conclusione con opinione netta?\n"
        "5. È strutturata in Markdown con sezioni?\n"
        "6. Lunghezza adeguata (almeno 500 parole)?\n"
        "Se manca anche un solo elemento, bocciala (passed=False).\n"
        "🚨 REGOLA JSON STRICТ: Il campo 'missing_elements' DEVE SEMPRE ESSERE UN ARRAY (lista), anche se manca un solo elemento (es. [\"Lunghezza\"]). Non usare mai una stringa semplice."
        "🚨 REGOLA FORMATO OUTPUT: Non 'incartare' la risposta. È SEVERAMENTE VIETATO usare tag XML (es. <function=QualityVerdict>) o blocchi markdown. Fornisci ESCLUSIVAMENTE gli argomenti grezzi."
    )
    user_prompt = f"Questa bozza è lunga ESATTAMENTE {word_count} parole.\nValuta attentamente questa bozza di recensione:\n\n{draft}"

    judge_llm = llm.with_structured_output(QualityVerdict)
    try:
        verdict = judge_llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
    except Exception as e:
        print(f"    Errore quality check: {e}")
        verdict = QualityVerdict(
            reasoning_process="Valutazione bypassata a causa di un errore dell'API.",
            passed=True,
            reason=f"Quality check fallito per errore: {e}"
        )

    reasoning = [create_react_entry(
        "quality_check",
        f"RAGIONAMENTO DEL GIUDICE:\n{verdict.reasoning_process}",
        f"Verdetto finale: {'PASS' if verdict.passed else 'FAIL'}",
        f"Elementi mancanti: {verdict.missing_elements}" if verdict.missing_elements else "Nessun elemento mancante"
    )]

    if verdict.passed or revision_count >= MAX_QUALITY_RETRIES:
        return Command(
            update={"quality_passed": True, "reasoning_trace": reasoning},
            goto="human_review"
        )
    else:
        return Command(
            update={
                "quality_passed": False,
                "revision_count": revision_count + 1,
                "reasoning_trace": reasoning,
                "human_feedback": f"REVISIONE AUTOMATICA: {verdict.reason}. Elementi mancanti: {', '.join(verdict.missing_elements)}"
            },
            goto="writer"
        )


# ============================================================
# NODO 6: HUMAN REVIEW
# ============================================================
def human_review_node(state: AgentState) -> Command[Literal["memory_updater", "writer", "researcher", "planner"]]:
    """Presenta la review all'utente e gestisce il feedback classificandolo con un LLM."""
    print("--- [HumanReviewNode] In attesa di approvazione umana ---")

    feedback_payload = interrupt("In attesa di feedback...")
    feedback = str(feedback_payload).strip()

    # Routing via LLM
    system_prompt = (
        "Sei un sistema di routing che classifica il feedback di un utente sulla recensione di un videogioco.\n"
        "Scegli UNA tra queste decisioni:\n"
        "- 'need_research': L'utente vuole più dettagli (es. 'parlami più dei boss', 'manca la lore').\n"
        "- 'change_topic': L'utente vuole scartare il gioco e recensirne un altro (es. 'cambia gioco', 'facciamo Zelda').\n"
        "- 'rewrite': L'utente vuole solo correzioni di stile o lunghezza (es. 'falla più corta', 'usa un tono più epico').\n"
        "- 'approve': L'utente fa complimenti senza chiedere modifiche (es. 'ottimo lavoro')."
    )

    user_prompt = f"FEEDBACK UTENTE: {feedback}"

    routing_llm = llm.with_structured_output(FeedbackRouting)
    try:
        decision = routing_llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
    except Exception as e:
        print(f"    [Warning] Errore LLM nel routing: {e}. Fallback su 'rewrite'.")
        decision = FeedbackRouting(decision="rewrite", reasoning=f"Fallback per errore LLM: {e}")

    reasoning = [create_react_entry(
        "human_review", f"Feedback analizzato: '{truncate_text(feedback, 100)}'. Decisione: {decision.decision}",
        "llm.with_structured_output(FeedbackRouting)", f"Motivo: {decision.reasoning}"
    )]

    dest_map = {
        "need_research": "researcher",
        "change_topic": "planner",
        "rewrite": "writer",
        "approve": "memory_updater"
    }

    # Se l'LLM allucina una decisione, il fallback è "writer"
    goto_node = dest_map.get(decision.decision, "writer")

    update_data = {
        "human_feedback": feedback,
        "reasoning_trace": reasoning
    }

    # Se l'utente vuole cambiare gioco, resettiamo l'input per il planner
    if goto_node == "planner":
        update_data["user_input"] = feedback

    return Command(update=update_data, goto=goto_node)


# ============================================================
# NODO 7: MEMORY UPDATER
# ============================================================
def memory_updater_node(state: AgentState) -> Dict[str, Any]:
    """Aggiorna KG e RAG con le entità estratte dalla review approvata."""
    print("--- [MemoryUpdaterNode] Aggiornamento Memoria (KG + RAG) ---")

    draft = state.get('draft_post', '')
    research_summary = state.get('research_summary', '')
    plan_info = state.get('planning_information', {})

    system_prompt = (
        "Sei un analista dati senior. Il tuo compito è estrarre TUTTI i dati fattuali da una recensione e dai suoi appunti di ricerca per popolare un Knowledge Graph enciclopedico.\n"
        "🚨 REGOLA COMPLETEZZA: Usa gli appunti di ricerca per riempire tutti i buchi tecnici (Anno di uscita, Piattaforme, Studio di sviluppo, Genere, Personaggi) anche se la recensione non li cita esplicitamente.\n"
        "🚨 REGOLA ONTOLOGICA: Distingui rigorosamente i TITOLI dei giochi dai GENERI. 'Soulslike', 'Action RPG', 'Open World', 'Roguelike', 'Metroidvania', 'Survival Horror' sono GENERI e non giochi.\n"
        "🚨 REGOLA DI NORMALIZZAZIONE: Quando estrai il nome di un gioco, uno studio o un genere, scrivilo SEMPRE in Title Case (iniziali maiuscole, es. 'Survival Horror')."
    )
    user_prompt = f"APPUNTI DI RICERCA:\n{research_summary}\n\nRECENSIONE FINALE:\n{draft}\n\nEstrai tutte le entità chiave per il database (Generi, Studi, Piattaforme, Anno, Boss, Meccaniche, Personaggi, Giochi Simili, Opinioni, Fonti)."

    entity_llm = llm.with_structured_output(PostEntities)
    try:
        entities = entity_llm.invoke([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
    except Exception as e:
        print(f"    Errore estrazione entità: {e}")
        entities = PostEntities(
            main_topic=state.get('user_input', 'Sconosciuto'),
            post_title=f"Review di {state.get('user_input', 'Sconosciuto')}"
        )

    reasoning = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unique_title = f"{entities.post_title} [{timestamp}]"

    # Aggiorna KG
    update_success = kg_manager.update(
        post_title=unique_title,
        topic=entities.main_topic,
        review_angle=entities.review_angle or plan_info.get('review_angle', ''),
        bosses=entities.bosses,
        mechanics=entities.mechanics,
        claims=entities.claims,
        sources=entities.sources,
        similar_games=entities.similar_games,
        genres=entities.genres,
        studios=entities.studios,
        platforms=entities.platforms,
        characters=entities.characters,
        release_year=entities.release_year
    )

    if update_success:
        reasoning.append(create_react_entry(
            "memory_updater",
            f"KG aggiornato: {len(entities.bosses)} boss, {len(entities.mechanics)} meccaniche, "
            f"{len(entities.claims)} claims, {len(entities.sources)} fonti",
            "kg_manager.update()", "Successo"
        ))
    else:
        reasoning.append(create_react_entry(
            "memory_updater", "Errore aggiornamento KG",
            "kg_manager.update()", "Fallito"
        ))

    # Salva il draft approvato nel RAG (solo il draft, non ri-salva i tool_outputs)
    if draft.strip():
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(draft)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source_url": f"internal_blog/{entities.main_topic.replace(' ', '_').lower()}",
                    "source_name": entities.post_title,
                    "source_type": "blog_post",
                    "topic": entities.main_topic,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        added = rag_manager.add_documents(documents)
        reasoning.append(create_react_entry(
            "memory_updater", f"RAG aggiornato con {added} chunk del post approvato",
            "rag_manager.add_documents()", f"Topic: {entities.main_topic}"
        ))

    return {"reasoning_trace": reasoning}


# ============================================================
# BUILD GRAPH
# ============================================================
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("summarizer", summarizer_node)
builder.add_node("writer", writer_node)
builder.add_node("quality_check", quality_check_node)
builder.add_node("human_review", human_review_node)
builder.add_node("memory_updater", memory_updater_node)

# Entry point
builder.set_entry_point("planner")

# Edge deterministici
builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "summarizer")
builder.add_edge("summarizer", "writer")
builder.add_edge("writer", "quality_check")
# quality_check → Command (human_review o writer)
# human_review → Command (memory_updater, writer, researcher, o planner)
builder.add_edge("memory_updater", END)

# Compile con checkpointer per interrupt (DA USARE QUANDO SI AVVIA CON MAIN.PY)
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Compile senza checkpointer per test rapidi (DA USARE DURANTE LO SVILUPPO e/o TEST CON DEV)
#graph = builder.compile()
