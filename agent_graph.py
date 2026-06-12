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
    PlannerIntent, SuggestPlannerOutput, SpecificPlannerOutput, GameResearchExtraction,
    QualityVerdict, PostEntities, FeedbackRouting, PlanApprovalRouting
)
from helpers import create_react_entry, format_extraction_for_writer, truncate_text, format_blacklist_for_llm, format_catalog_for_llm, format_kg_context, format_existing_reviews, format_similarity_catalog, format_krag_entities, format_recent_posts_for_writer
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

_INTENT_CACHE = {}
_PLAN_CACHE = {}

# ============================================================
# NODO 1: PLANNER
# ============================================================
def planner_node(state: AgentState) -> Dict[str, Any]:
    """Pianifica la review: gestisce sia input specifico che richiesta di suggerimento."""
    print("--- [PlannerNode] Analisi del topic e pianificazione ---")

    user_in = state['user_input'].strip()
    reasoning = []

    # Step 1: Capire l'intento dell'utente e leggere la memoria di sistema
    active_plan = kg_manager.get_active_plan_status()
    system_context = ""

    # Se c'è un piano, "iniettiamo" la consapevolezza nel prompt dell'LLM
    if active_plan and active_plan.get("next_game"):
        next_g = active_plan["next_game"]
        system_context = (
            f"\n\n🚨 CONTESTO DI SISTEMA: Nel database c'è un piano editoriale attivo. "
            f"Il prossimo gioco in coda da recensire è '{next_g}'. "
            f"Se l'utente accetta, acconsente o fa affermazioni generiche di conferma (es. 'ok', 'vai', 'procedi', 'continua il piano'), oppure semplicemente preme invio (input vuoto)"
            f"DEVI scegliere 'specific' ed estrarre come game_name ESATTAMENTE '{next_g}'."
        )
    else:
        system_context = (
            f"\n\n🚨 CONTESTO DI SISTEMA: ATTENZIONE, al momento NON c'è nessun piano editoriale in sospeso. "
            f"Se l'utente usa frasi generiche come 'ok', 'vai avanti', 'procedi' SENZA specificare fisicamente il nome di un gioco, "
            f"DEVI scegliere 'suggest' in modo da creare in automatico un nuovo piano editoriale da zero."
        )

    intent_messages = [
        ("system", "Sei un assistente AI. Il tuo compito è classificare l'intento dell'utente.\n"
                   "Scegli 'suggest' se l'utente chiede un NUOVO piano editoriale o un suggerimento.\n"
                   "Scegli 'specific' se l'utente indica un videogioco preciso (es. 'Facciamo Zelda'), OPPURE se accetta di procedere con il piano attivo." + system_context),
        ("user", f"L'utente dice: '{user_in}'. Ha specificato un gioco, confermato il piano, o vuole un suggerimento?")
    ]
    if user_in in _INTENT_CACHE:
        intent = _INTENT_CACHE[user_in]
    else:
        intent_llm = llm.with_structured_output(PlannerIntent)
        intent = intent_llm.invoke(intent_messages)
        _INTENT_CACHE[user_in] = intent

    # ==========================================
    # BINARIO A: MODALITÀ SUGGERIMENTO
    # ==========================================
    if intent.mode == "suggest":
        planner_llm = llm.with_structured_output(SuggestPlannerOutput)

        all_games_kg = kg_manager.query_all_games()
        blacklist_raw = kg_manager.get_recent_posts(limit=100)

        reasoning.append(create_react_entry("planner", "L'utente vuole un suggerimento, interrogo il KG per trovare i giochi meno coperti", "kg_manager.query_all_games()", truncate_text(str(all_games_kg), 300)))

        suggest_messages = [
            ("system", "Sei un Editorial Director per un blog di videogiochi. "
                       "Devi generare una SEQUENZA di 3 prossime recensioni. Giustifica l'ordine logico. Infine, estrai il PRIMO gioco della sequenza per scriverlo OGGI. 🚨 ATTENZIONE: LA BLACKLIST È LA LEGGE SUPREMA. Se l'utente ti chiede esplicitamente di inserire un gioco che si trova nella '⛔ BLACKLIST' o che ha '⛔ GIÀ RECENSITO' nel 'CATALOGO', DEVI IGNORARE QUELLA SPECIFICA RICHIESTA (fornirai poi nel campo apposito della risposta JSON la spiegazione) dell'utente e inserire un gioco nuovo al suo posto. Non ci sono eccezioni, neanche se l'utente insiste.\n\n"
                       "🚨 REGOLA 1 (LA TUA PRIORITÀ ASSOLUTA - I DIVIETI): Leggi attentamente la '⛔ BLACKLIST' e il 'CATALOGO GIOCHI'. I giochi presenti nella Blacklist, o che all'interno del Catalogo hanno '⛔ GIÀ RECENSITO' sono 'BRUCIATI' e già recensiti. È SEVERAMENTE VIETATO proporli o inserirli nella sequenza. Non ci sono eccezioni.\n\n"
                       "🚨 REGOLA 2 (LA SCELTA E I GENERI): Analizza la richiesta dell'utente. Scegli i giochi attingendo dai titoli ancora liberi nel 'CATALOGO GIOCHI', per aiutare a diversificare i contenuti del blog. Se non ci sono giochi pertinenti, ATTINGI ALLA TUA CONOSCENZA VIDEOLUDICA GENERALE. Sei PIENAMENTE AUTORIZZATO a usare la tua memoria interna, MA ATTENZIONE: devono comunque sottostare alla BLACKLIST E AL CATALOGO, CIOE NON DEVONO AVERE '⛔ GIÀ RECENSITO'. Se un gioco ti viene in mente ma è già recensito, scartalo e pensane un altro!\n"
                       "🚨 IL CATALOGO NON È UNIVERSALE (FACT-CHECKING OBBLIGATORIO): Il database che ti forniamo è limitato e potrebbe avere lacune o errori. DEVI SEMPRE usare la tua conoscenza interna per fare fact-checking. Se un gioco nel catalogo ha un genere o un anno sbagliato, omettilo o correggilo mentalmente. Se l'utente chiede criteri specifici (es. 'horror del 2022') e nel catalogo non c'è nulla, o ci sono solo informazioni parziali, ATTINGI DIRETTAMENTE ALLA TUA MEMORIA INTERNA. Sei PIENAMENTE AUTORIZZATO a proporre giochi non presenti nel catalogo, purché non siano '⛔ GIÀ RECENSITO' o in Blacklist.\n"
                       "Sii un purista dei generi videoludici: rispetta categoricamente le differenze tra i sottogeneri (es. non confondere JRPG giapponesi con RPG occidentali, oppure gli 'Horror' con i 'Survival Horror'). SE NON CONOSCI IL GENERE DI UN GIOCO NEL KG, USA LA TUA CONOSCENZA PERSONALE PER DETERMINARLO (se non sei sicuro, lascia perdere quel gioco come consiglio e non metterlo nel piano editoriale).\n\n"
                       "🚨 REGOLA 3 (IL FILTRO MENTALE): Prima di confermare la sequenza, fai un check incrociato con la Fase 1. Se hai pensato a un gioco famosissimo ma è già stato recensito, SCARTALO IMMEDIATAMENTE e trovane un altro (anche meno famoso/di nicchia) per sostituirlo.\n\n"
                       "🚨 REGOLA SULL'ANGOLO: Poiché sceglierai obbligatoriamente giochi mai trattati, l'angolo DEVE essere 'Recensione Completa e Generale'.\n\n"
                       "🚨 REGOLA NOMI CANONICI: Quando l'utente nomina un gioco (es. FF6, Silent Hill 2 Remake), tu devi mentalmente tradurlo nel suo nome CANONICO COMPLETO UFFICIALE e confrontarlo in modo ESATTO e LETTERALE con i nomi presenti in Blacklist e Catalogo. Fai estrema attenzione ai numeri romani o arabi (es. Final Fantasy 6 NON è Final Fantasy VIII, per cui se non c'è un match esatto tra quello proposto e quello nel catalogo, il gioco proposto è '✅ LIBERO')."
                       "🚨 REGOLA FORMATO (CRITICA): Agisci come un'API dati pura. Restituisci ESCLUSIVAMENTE l'oggetto JSON. NON incapsulare MAI la risposta in alcun tipo di tag, parentesi angolare (< >) o blocco markdown (```). Rispondi con il JSON nudo e crudo. Usa apostrofi normali, nessun backslash (\)."),
            ("user", f"L'utente ha fatto questa richiesta specifica: '{user_in}'.\n\n"
                     f"⛔ BLACKLIST DEI GIOCHI GIÀ RECENSITI (VIETATO USARLI):\n{format_blacklist_for_llm(blacklist_raw)}\n\n"
                     f"🎮 CATALOGO GIOCHI (Knowledge Graph):\n{format_catalog_for_llm(all_games_kg)}")
        ]
        if user_in in _PLAN_CACHE:
            plan_result = _PLAN_CACHE[user_in]
        else:
            plan_result = planner_llm.invoke(suggest_messages)
            _PLAN_CACHE[user_in] = plan_result

        reasoning.append(create_react_entry(
            "planner",
            f"RAGIONAMENTO DEL DIRETTORE:\n{plan_result.reasoning_process}",
            "llm.with_structured_output(SuggestPlannerOutput)",
            f"Sequenza generata: {plan_result.sequence_of_posts}"
        ))

        user_decision = interrupt({
            "type": "topic_suggestion",
            "sequence": plan_result.sequence_of_posts,
            "suggestion": plan_result.suggested_game,
            "angle": plan_result.review_angle,
            "justification": plan_result.justification,
            "message": f"🗓️ CALENDARIO EDITORIALE PROPOSTO:\n"
                       f"Analisi/Note: {plan_result.feedback_analysis}\n"
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
                ("system", "Sei un Editorial Director. Modifica il tuo piano editoriale precedente seguendo ALLA LETTERA le nuove istruzioni dell'utente. 🚨 ATTENZIONE: LA BLACKLIST È LA LEGGE SUPREMA. Se l'utente ti chiede esplicitamente di inserire un gioco che si trova nella '⛔ BLACKLIST' o che ha '⛔ GIÀ RECENSITO' nel 'CATALOGO', DEVI IGNORARE QUELLA SPECIFICA RICHIESTA (fornirai poi nel campo apposito della risposta JSON la spiegazione) dell'utente e inserire un gioco nuovo al suo posto. Non ci sono eccezioni, neanche se l'utente insiste.\n\n"
                           "🚨 GESTIONE CAMBIO TEMA (SOVRASCRITTURA): Il 'Feedback Utente' ha la PRIORITÀ ASSOLUTA sulla 'Richiesta originale'. Se l'utente nel feedback ti chiede giochi di un genere diverso (es. prima voleva Horror, ora ti chiede Hades 2 o Final Fantasy), l'utente HA CAMBIATO IDEA. DEVI ignorare e cancellare i vecchi vincoli di genere/anno e adattare il piano alle nuove richieste! Non costringere i nuovi giochi dentro i vecchi generi, devi essere FLESSIBILE e adattarti a quello che ti chiede l'utente interpretandolo correttamente.\n\n"
                           "🚨 CONTROLLO NOMI E NUMERI (CRITICO): Prima di scartare un gioco perché credi sia nella '⛔ BLACKLIST', controlla lettera per lettera. I NUMERI SONO IMPORTANTI. 'Final Fantasy 6' (o VI) è DIVERSO da 'Final Fantasy Viii'. 'Silent Hill 2' è DIVERSO da 'Silent Hill'. Se non è un match esatto al 100%, il gioco è '✅ LIBERO' e devi usarlo!\n\n"
                           "🚨 REGOLA 1 (LA TUA PRIORITÀ ASSOLUTA - I DIVIETI): Leggi attentamente la '⛔ BLACKLIST' e il 'CATALOGO GIOCHI'. I giochi presenti nella Blacklist, o che all'interno del Catalogo hanno '⛔ GIÀ RECENSITO' sono 'BRUCIATI' e già recensiti. È SEVERAMENTE VIETATO proporli o inserirli nella sequenza. Non ci sono eccezioni.\n\n"
                           "Se l'utente ti chiede esplicitamente di inserire un gioco già recensito o presente nella '⛔ BLACKLIST', DEVI SCARTARE SOLO QUEL NOME e avvisare l'utente. NON CERCARE RIMPIAZZI o sostituti per il gioco vietato! Se il piano precedente andava bene per il resto dei criteri, confermalo inalterato. INVECE, se l'utente oltre a un gioco vietato ha chiesto di CAMBIARE COMPLETAMENTE i criteri (es. 'basta horror, facciamo sparatutto'), allora obbedisci alla nuova direttiva ignorando il gioco vietato.\n\n"
                           "🚨 REGOLA 2 (LA SCELTA E I GENERI): Analizza la richiesta dell'utente. Scegli i giochi attingendo dai titoli ancora liberi nel 'CATALOGO GIOCHI', per aiutare a diversificare i contenuti del blog. Se non ci sono giochi pertinenti, ATTINGI ALLA TUA CONOSCENZA VIDEOLUDICA GENERALE. Sei PIENAMENTE AUTORIZZATO a usare la tua memoria interna, MA ATTENZIONE: devono comunque sottostare alla BLACKLIST E AL CATALOGO, CIOE NON DEVONO AVERE '⛔ GIÀ RECENSITO'. Se un gioco ti viene in mente ma è già recensito, scartalo e pensane un altro!\n"
                           "🚨 IL CATALOGO NON È UNIVERSALE (FACT-CHECKING OBBLIGATORIO): Il database che ti forniamo è limitato e potrebbe avere lacune o errori. DEVI SEMPRE usare la tua conoscenza interna per fare fact-checking. Se un gioco nel catalogo ha un genere o un anno sbagliato, omettilo o correggilo mentalmente. Se l'utente chiede criteri specifici (es. 'horror del 2022') e nel catalogo non c'è nulla, o ci sono solo informazioni parziali, ATTINGI DIRETTAMENTE ALLA TUA MEMORIA INTERNA. Sei PIENAMENTE AUTORIZZATO a proporre giochi non presenti nel catalogo, purché non siano '⛔ GIÀ RECENSITO' o in Blacklist.\n"
                           "Sii un purista dei generi videoludici: rispetta categoricamente le differenze tra i sottogeneri (es. non confondere JRPG giapponesi con RPG occidentali, oppure gli 'Horror' con i 'Survival Horror'). SE NON CONOSCI IL GENERE DI UN GIOCO NEL KG, USA LA TUA CONOSCENZA PERSONALE PER DETERMINARLO (se non sei sicuro, lascia perdere quel gioco come consiglio e non metterlo nel piano editoriale).\n\n"
                           "🚨 REGOLA 3 (IL FILTRO MENTALE): Prima di confermare la sequenza, fai un check incrociato con la Fase 1. Se hai pensato a un gioco famosissimo ma è già stato recensito, SCARTALO IMMEDIATAMENTE e trovane un altro (anche meno famoso/di nicchia) per sostituirlo.\n\n"
                           "🚨 REGOLA SULL'ANGOLO: Poiché sceglierai obbligatoriamente giochi mai trattati, l'angolo DEVE essere 'Recensione Completa e Generale'.\n\n"
                           "🚨 REGOLA ANTI-RIPETIZIONE (CRITICA): Guarda i giochi elencati nel 'Piano precedente'. Se il feedback dell'utente ti chiede di CAMBIARE DEI GIOCHI o ne vuole ALTRI, quei titoli sono temporaneamente banditi. È SEVERAMENTE VIETATO riproporre gli stessi giochi che hai appena suggerito (a meno che l'utente non ti dica qualcosa del tipo 'il primo gioco lascialo però gli altri due cambiali', in quel caso solo gli ultimi 2 sono da modificare necessariamente rispetto a prima)!\n\n"
                           "🚨 REGOLA NOMI CANONICI: Quando l'utente nomina un gioco (es. FF6, Silent Hill 2 Remake), tu devi mentalmente tradurlo nel suo nome CANONICO COMPLETO UFFICIALE e confrontarlo in modo ESATTO e LETTERALE con i nomi presenti in Blacklist e Catalogo. Fai estrema attenzione ai numeri romani o arabi (es. Final Fantasy 6 NON è Final Fantasy VIII, per cui se non c'è un match esatto tra quello proposto e quello nel catalogo, il gioco proposto è '✅ LIBERO')."
                           "🚨 REGOLA FORMATO (CRITICA): Agisci come un'API dati pura. Restituisci ESCLUSIVAMENTE l'oggetto JSON. NON incapsulare MAI la risposta in alcun tipo di tag, parentesi angolare (< >) o blocco markdown (```). Rispondi con il JSON nudo e crudo. Usa apostrofi normali, nessun backslash (\)."),
                ("user", f"Richiesta originale: '{user_in}'\nPiano precedente:\nSequenza: {plan_result.sequence_of_posts}\nGioco di oggi: {plan_result.suggested_game}\n\n"
                         f"💬 FEEDBACK UTENTE (PRIORITARIO RISPETTO ALLA RICHIESTA ORIGINALE): '{user_response}'\n\n"
                         f"⛔ BLACKLIST DEI GIOCHI GIÀ RECENSITI (VIETATO USARLI):\n{format_blacklist_for_llm(blacklist_raw)}\n\n"
                         f"🎮 CATALOGO GIOCHI (Knowledge Graph):\n{format_catalog_for_llm(all_games_kg)}")
            ]
            plan_result = planner_llm.invoke(feedback_messages)
            reasoning.append(create_react_entry("planner", f"RAGIONAMENTO POST-FEEDBACK:\n{plan_result.reasoning_process}", "Modifica Inline", f"Nuovo topic: {plan_result.suggested_game}"))

            print("\n" + "=" * 50)
            print(" 🗓️ NUOVO PIANO EDITORIALE (Modificato)")
            print("=" * 50)
            print(f"Analisi/Note: {plan_result.feedback_analysis}")
            print(f"Sequenza: {', '.join(plan_result.sequence_of_posts)}")
            print(f"💡 OGGI RECENSIREMO: '{plan_result.suggested_game}' (Focus: {plan_result.review_angle})\n")
        else:
            reasoning.append(create_react_entry("planner", "Piano confermato dall'utente", "LLM Routing", f"Decisione: {approval.decision}"))

        topic = plan_result.suggested_game

        try:
            kg_manager.save_active_plan(plan_result.sequence_of_posts)
            reasoning.append(create_react_entry(
                "planner", "Piano editoriale salvato in memoria persistente (Neo4j)",
                "kg_manager.save_active_plan()", str(plan_result.sequence_of_posts)
            ))
        except Exception as e:
            print(f"[Warning] Impossibile salvare il piano nel KG: {e}")

    # ==========================================
    # BINARIO B: MODALITÀ GIOCO SPECIFICO
    # ==========================================
    else:
        topic = intent.game_name or user_in
        existing_reviews = format_existing_reviews(kg_manager.check_existing_reviews(topic))

        reasoning.append(create_react_entry(
            "planner", f"L'utente vuole recensire '{topic}'. Interrogo il KG.",
            f"kg_manager.check_existing_reviews('{topic}')", truncate_text(str(existing_reviews), 300)
        ))

        existing_str = str(existing_reviews)
        has_existing = (
            existing_str and
            "Nessuna recensione esistente" not in existing_str and
            "Errore" not in existing_str
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
                    "Nessuna recensione esistente" not in existing_str and
                    "Errore" not in existing_str
                )

                reasoning.append(create_react_entry(
                "planner", f"Review esistente gestita via LLM. Utente ha risposto: '{user_response}'",
                "Routing Semantico", f"Topic finale: {topic}"
                ))

                print(f"\n💡 CAMBIO GIOCO ACCETTATO. Oggi recensiremo: '{topic}'\n")

        angle_instruction = (
            "Il gioco ha GIÀ delle recensioni passate nel nostro database. Scegli un ANGOLO INEDITO CHE NON SIA STATO TRATTATO FINORA (es. focus su una meccanica, una boss fight)."
            if has_existing else
            "Questa è la PRIMA recensione per questo gioco. L'angolo DEVE essere 'Recensione Completa e Generale'. Il piano deve coprire lore/storia, gameplay, comparto tecnico e tutte le altre informazioni utili che trovi."
        )

        user_prompt_content = f"Topic: '{topic}'\nReview esistenti:\n{existing_reviews}\n"

        if has_existing and 'user_response' in locals():
            user_prompt_content += f"\n💬 FEEDBACK DELL'UTENTE: '{user_response}'\n"
            user_prompt_content += (
                "🚨 REGOLA SULL'ANGOLO: Analizza il feedback dell'utente. "
                "Se contiene SOLO una conferma generica (es. 'ok', 'va bene', 'procedi', 'sì'), "
                "ignora il feedback e inventa tu un angolo INEDITO, CONTROLLANDO che non sia già stato trattato.\n"
                "Se invece l'utente ha richiesto un focus specifico (es. 'parliamo della trama', 'solo i boss', 'ok, fai la grafica'), "
                "DEVI ASSOLUTAMENTE impostare il campo 'review_angle' su quella specifica richiesta!\n"
            )

        user_prompt_content += "\nGenera un piano editoriale."

        planner_llm = llm.with_structured_output(SpecificPlannerOutput)

        plan_result = planner_llm.invoke([
            ("system", f"Sei un Editorial Director per un blog di videogiochi.\n{angle_instruction}\n🚨 REGOLA FORMATO (CRITICA): Agisci come un'API dati pura. Restituisci ESCLUSIVAMENTE l'oggetto JSON. NON incapsulare MAI la risposta in alcun tipo di tag, parentesi angolare (< >) o blocco markdown (```). Rispondi con il JSON nudo e crudo. Usa apostrofi normali, nessun backslash (\)."),
            ("user", user_prompt_content)
        ])

        reasoning.append(create_react_entry("planner", f"RAGIONAMENTO DEL DIRETTORE:\n{plan_result.reasoning_process}", "llm.with_structured_output(SpecificPlannerOutput)", f"Piano generato: angolo '{plan_result.review_angle}'"))

    kg_context = format_kg_context(kg_manager.query(topic))

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
    review_angle = state.get('planning_information', {}).get('review_angle', 'Recensione Generale')
    reasoning = []
    tool_outputs = state.get('tool_outputs', {})

    # ═══════════════════════════════════════════
    # FASE 1: DETERMINISTICA (sempre eseguita)
    # ═══════════════════════════════════════════

    if not feedback:
        # 1A: Web search OBBLIGATORIA
        search_query = f"{topic} recensione"
        print(f"    [Fase 1] Ricerca web: '{search_query}'")
        web_result = search_tool.invoke({"query": search_query})
        tool_outputs["search_tool"] = [str(web_result)]

        # 1B: KG query per K-RAG
        print(f"    [Fase 1] Query KG per entità K-RAG: '{topic}'")
        kg_entities = format_krag_entities(kg_manager.get_entities_for_krag(topic))
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
    else:
        print("Feedback umano presente, salto la fase deterministica e passo direttamente alla fase agentica per approfondire in modo mirato secondo il feedback.")

    # ═══════════════════════════════════════════
    # FASE 2: AGENTICA (Custom ReAct Loop)
    # ═══════════════════════════════════════════
    print("    [Fase 2] ReAct loop manuale per approfondimento...")
    optional_tools = [search_tool, rag_retrieval_tool, deep_read_article, youtube_transcript_fetcher, knowledge_graph_tool]
    llm_with_tools = llm.bind_tools(optional_tools)

    already_known = ""
    previous_summary = state.get('research_summary', '').strip()
    if previous_summary:
        already_known = (
            f"🚨 SINTESI DELLE INFORMAZIONI GIÀ RACCOLTE FINORA:\n"
            f"{previous_summary}\n\n"
            f"NON usare i tool per cercare di nuovo le informazioni scritte qui sopra! "
            f"Concentrati ESCLUSIVAMENTE sulle mancanze o sul nuovo FEEDBACK DELL'UTENTE."
        )

    system_prompt = (
        f"Sei un instancabile e meticoloso ricercatore per un blog di videogiochi.\n"
        f"Devi cercare informazioni enciclopediche e critiche su '{topic}'. Contesto dal KG: {truncate_text(kg_context, 500)}\n"
        f"{already_known}\n"
        f"🚨 FOCUS DELLA RICERCA: L'angolo editoriale è '{review_angle}'.\n"
        f"Se è un angolo specifico, orienta le tue query (web e RAG) su quel singolo aspetto; se è generale, copri tutti gli aspetti del gioco.\n"
        f"FEEDBACK DELL'UTENTE: '{feedback}'. Concentrati nel cercare informazioni per soddisfare questa richiesta!\n\n"

        f"🚨 CHECKLIST E FOCUS DELLA RICERCA: Non usare l'azione 'STOP' finché non hai trovato, usando più tool, queste informazioni essenziali (salvo esaurimento iterazioni)\n"
        f"Il tuo obiettivo principale è soddisfare questo Focus Editoriale: '{review_angle}'.\n"
        f"- SE IL FOCUS È GENERALE (es. 'Recensione Completa e Generale'): Non usare 'STOP' finché non hai trovato: 1. Trama generale e contesto narrativo, 2. Gameplay e meccaniche principali, 3. Dati tecnici (Anno, piattaforme, sviluppatore).\n"
        f"- SE IL FOCUS È SPECIFICO (es. 'Sistema di combattimento'): La tua priorità ASSOLUTA è trovare informazioni iper-dettagliate su '{review_angle}'. IGNORA i punti della checklist se non c'entrano nulla con il tuo focus (es., se il focus è la storia, ignora elementi come il gameplay o i combattimenti)! Trova solo i Dati Tecnici di base per inquadrare il gioco, e poi sprofonda nella ricerca del tuo argomento specifico.\n"
        f"- REGOLA DELLE DUE FONTI: DEVI approfondire ALMENO DUE FONTI DISTINTE per avere prospettive diverse. Puoi farlo leggendo due articoli web diversi (usando 'deep_read_article') OPPURE leggendo un articolo e analizzando la trascrizione di un video saggio (usando 'youtube_transcript_fetcher' e poi 'deep_read_article'). Non basta leggere lo stesso articolo in più parti usando l'offset! Questo ti serve per avere più prospettive sul topic!\n"
        f"- Se hai letto con la deep read almeno due URL diversi, poi decidi tu quale approfondire aumentando gli offset per leggere il resto dell'articolo.\n"
        f"- Se le ricerche iniziali (Tavily/RAG) non bastano, INVENTA NUOVE QUERY mirate (es. se il focus è la lore, cerca 'Silent Hill spiegazione finale' o 'Silent Hill simbolismi').\n\n"

        f"🚨 DIRETTIVE TECNICHE SUI TOOL (DA RISPETTARE RIGOROSAMENTE):\n"
        f"- GIOCO BASE: Cerca info solo su '{topic}'. Scarta DLC, Mod o Spinoff.\n"
        f"- VALUTAZIONE FONTI: Prima di usare 'deep_read_article', leggi lo snippet del 'search_tool'. Se lo snippet contiene parole che ti fanno pensare a siti che includano informazioni INUTILI (per esempio un sito di compravendita di videogiochi) IGNORA QUEL LINK. Non sprecare iterazioni a leggerlo. Usa piuttosto il tuo ragionamento per fare una nuova query più specifica (es. 'Silent Hill recensione trama')\n"
        f"- LETTURA PROFONDA: La semplice ricerca web dà solo riassunti. Quindi devi usarla se hai bisogno di una panoramica generale o di trovare nuove informazioni. Se un URL giornalistico è promettente (es. IGN, Wikipedia, Everyeye), USA SUBITO 'deep_read_article' per leggerlo (es. offset=0, limit=5). Inizia leggendo i primi paragrafi (es. offset=0, limit=5). Se l'articolo è lungo e ti servono altre info, richiama il tool aumentando l'offset. IN ALTERNATIVA, le video-recensioni o i video-saggi su YouTube sono considerati fonti ECCELLENTI e di altissima qualità, quindi usa 'youtube_transcript_fetcher' per estrarre la trascrizione e poi la 'deep_read_article' per leggerne il contenuto passando l'URL del video (esattamente come per i siti web)!.\n"
        f"- USO DEL RAG: Il tool 'rag_retrieval_tool' non serve solo per cercare il titolo del gioco. Puoi e DEVI usarlo passandogli DOMANDE DISCORSIVE specifiche per approfondire la tua conoscenza del gioco (es. 'Come funziona il sistema di cura?', 'Chi è il boss finale?'). Il RAG ti risponderà pescando dai chunk salvati!\n"
        f"- VIDEO YOUTUBE: Per trovare video, fai una query con 'search_tool' aggiungendo la parola chiave (es. 'Silent Hill recensione youtube' o 'Elden ring lore youtube video'). Quando trovi un URL YouTube nei risultati, usa IMMEDIATAMENTE 'youtube_transcript_fetcher' su quell'URL per generare la trascrizione E POI, SUBITO DOPO, usa 'deep_read_article' passandogli lo STESSO URL di YouTube (offset=0, limit=5) per leggerne i paragrafi come se fosse un normale articolo web. Se la trascrizione è lunga e ti servono altre info, richiama il tool aumentando l'offset.\n"
        f"- CHIAMATA SINTATTICA TOOL: Usa esclusivamente l'interfaccia nativa invisibile per chiamare i tool. Il tuo output testuale deve contenere SOLO il tuo ragionamento in linguaggio naturale e poi invoca i tool usando ESATTAMENTE e SOLO il loro nome (es. 'search_tool'). È severamente vietato concatenare o fondere gli argomenti JSON direttamente nel nome del tool e/o scrivere tag XML.\n"
        f"- KNOWLEDGE GRAPH: Usa 'knowledge_graph_tool' per dubbi o fact-checking veloce sugli aspetti del topic.\n\n"
        f"- REGOLA APPROFONDIMENTO DELLE DUE FONTI: Non usare l'azione 'STOP' se hai esplorato un solo dominio web. Anche se il primo sito (es. Wikipedia) ti ha dato tutte le informazioni, DEVI usare 'deep_read_article' su un secondo URL di una testata giornalistica per avere un parere critico prima di terminare la ricerca.\n"
        f"- ISOLAMENTO DELLA SAGA E GIOCO ESATTO: Cerca info ESCLUSIVAMENTE sul gioco '{topic}'. SCARTA CATEGORICAMENTE link o info su prequel, sequel (es. Silent Hill 2), film omonimi o capitoli futuri (es. Silent Hill f). Nelle tue query (Web e RAG), usa parole chiave per disambiguare (es. 'Silent Hill 1999 PS1 trama').\n"

        f"⏳ ATTENZIONE: Hai un limite rigido di ITERAZIONI. Cerca di essere chirurgico e completare la Checklist il più velocemente possibile prima di esaurirle.\n"
        f"PENSA AD ALTA VOCE: Prima di usare qualsiasi tool, scrivi sempre una frase spiegando quale informazione ti manca e PERCHÉ stai per usare quel tool."
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

            actual_thought = response.content.strip() if response.content else f"Analizzo il contesto e decido di usare {response.tool_calls[0]['name']}"

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

    research_text = truncate_text("\n\n".join(all_research), 15000) # Limite token per l'estrazione, preferisco tagliare qui che rischiare di superare il limite durante l'estrazione e perdere tutto il contesto

    system_prompt = (
        f"Sei un analista di ricerca enciclopedico. Il tuo compito è leggere i dati forniti ed ESTRARRE ogni singolo dettaglio rilevante.\n"
        f"🚨 REGOLA: NON FARE RIASSUNTI BREVI. Mantieni ogni singolo dettaglio specifico sul gioco, aneddoto sulla trama e modalità di gameplay, ecc... che trovi nel testo originale.\n"
        "🚨 REGOLA ZERO-ALLUCINAZIONI SULLE FONTI: Estrai ESCLUSIVAMENTE gli URL e i nomi delle fonti che sono scritti esplicitamente nel testo che ricevi. È SEVERAMENTE VIETATO inventare URL o testate giornalistiche che non ti sono state fornite basandoti sulla tua memoria interna.\n"
        "🚨 REGOLA DEGLI SCARTI (FACT CHECKING): Se leggi testi provenienti da store, listini prezzi o fonti non pertinenti, filtrali. MA DEVI OBBLIGATORIAMENTE dichiarare che li hai scartati nel campo 'fact_check_notes' spiegando il perché (es. 'Scartata fonte X perché è un e-commerce') L'e-commerce è solo un esempio per aiutarti a comprende il task.\n"
        f"- ATTENZIONE ALLA CONTAMINAZIONE DA FRANCHISE: Se il testo fornito contiene informazioni su sequel, prequel, remake, film o altri capitoli della saga (es. citazioni a 'Silent Hill f', 'Silent Hill 2', o mostri iconici di altri capitoli come 'Pyramid Head'), ELIMINALI COMPLETAMENTE dall'estrazione. Salva solo i dati del capitolo originale e dichiara lo scarto nelle 'fact_check_notes'.\n"
        f"Filtra la spazzatura web (menu, cookie, pubblicità). Genera paragrafi ricchi di dettagli, non elenchi puntati o riassunti brevi.\n"
        f"Valuta la credibilità delle fonti: 'alta' (testate note, es. IGN, Everyeye, Multiplayer.it, GameSpot, GameInformer), 'media' (blog), 'bassa' (sconosciute)."
        f"Se fonti diverse dicono cose opposte, fidati di quella con credibilità più alta.\n\n"
        f"Se ci sono informazioni inutili e/o fuorvinati e/o fuori contesto (ad esempio, informazioni non rilevanti perché c'è una fonte che non c'entra nulla con il gioco e/o con le recensioni e che contiene informazioni totalmente fuori contesto), segnalalo compilando l'apposito campo 'fact_check_notes' (le fonti che citi in questo campo vanno riportate con il loro URL, se lo hanno). Allo stesso modo, se trovi contraddizioni tra le fonti web e/o con il KG, evidenziale chiaramente sempre nella sezione 'fact_check_notes' alla fine dell'estrazione.\n\n"
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

    recent_posts = format_recent_posts_for_writer(kg_manager.get_recent_posts(limit=3))

    system_prompt = f"""Sei un blogger esperto di videogiochi.
Devi scrivere un ARTICOLO su '{topic}'.

🚨 ADERENZA TOTALE AL FOCUS (MASSIMA PRIORITÀ):
Focus editoriale OBBLIGATORIO: {review_angle}.
- Se il focus è specifico (es. 'Meccaniche di combattimento', 'Analisi della trama'), l'80% dell'intero articolo DEVE essere dedicato a scavare in profondità in quel singolo aspetto. Usa il restante 20% per dare una veloce infarinatura generale. Non fare una recensione classica!
- Se il focus è 'Recensione Completa e Generale', allora fai un'analisi classica a 360 gradi coprendo tutti gli aspetti (una recensione tradizionale).

🚨 ZERO FILLER (DIVIETO DI PARAGRAFI VUOTI):
Se dal materiale di ricerca non emergono informazioni su un certo aspetto (es. grafica, audio, o una certa meccanica), NON scriverlo. È SEVERAMENTE VIETATO scrivere frasi come "Purtroppo non abbiamo informazioni su...". Semplicemente, lascia perdere quell'aspetto e concentrati solo su ciò che è emerso dalla ricerca!
È SEVERAMENTE VIETATO inserire note di scuse alla fine dell'articolo (es. "Nota: le informazioni sono limitate...").
È SEVERAMENTE VIETATO usare parole dubbiose e ipotetiche ripetitive come "Sembra che", "Potrebbe essere". Scrivi in modo assertivo basandoti SOLO su quello che sai per certo, devi essere sicuro e professionale sulla base delle informazioni che hai!

REGOLA ANTI-ALLUCINAZIONE E LIMITI SUL CONTENUTO:
- Parla ESCLUSIVAMENTE di '{topic}'.
- Scrivi usando ESCLUSIVAMENTE le informazioni presenti nel materiale di ricerca fornito sotto. Non inventare nulla.
- ATTENZIONE AI GENERI VIDEOLUDICI: Sii un purista dei generi videoludici: rispetta categoricamente le differenze tra i sottogeneri e attieniti a quelli eventualmente presenti nelle fonti, o se deducibili da essi (es. non confondere JRPG giapponesi con RPG occidentali, oppure gli 'Horror' con i 'Survival Horror').
- ISOLAMENTO DELLA SAGA: Se '{topic}' è il primo capitolo di una serie, è SEVERAMENTE VIETATO citare mostri, personaggi, eventi o recensioni appartenenti a sequel, film o remake futuri (es. niente Pyramid Head in Silent Hill 1). Stesso discorso vale per un gioco X, se X è il terzo capitolo, tu devi parlare e usare informazioni solo di X, non di capitoli precedenti e/o futuri.
- ATTENZIONE ALLA CONTAMINAZIONE DA FRANCHISE: Se il testo fornito contiene informazioni su sequel, prequel, remake, film o altri capitoli della saga (es. citazioni a 'Silent Hill f', 'Silent Hill 2', o mostri iconici di altri capitoli come 'Pyramid Head'), NON USARLI perché fuori tema rispetto alla recensione del gioco!

LUNGHEZZA DELL'ARTICOLO:
L'articolo deve rispettare questa indicazione di lunghezza: {POST_LENGTH_GUIDANCE}. Per raggiungere questo obiettivo senza allungare il brodo, sviluppa i paragrafi in ESTREMA profondità, analizzando minuziosamente la lore, ogni singola meccanica e l'atmosfera.

RICCHEZZA DEI DETTAGLI (PER IL DATABASE DEL BLOG):
- Se li trovi tra le informazioni fornite, devi menzionare in modo naturale nel testo: il Genere del gioco, lo Studio di sviluppo, l'Anno di uscita e le Piattaforme disponibili.
- Arricchisci la recensione citando i nomi specifici di Boss, Personaggi chiave e Meccaniche di gioco scoperti durante la ricerca.
- 🚨 ATTENZIONE: Inserisci questi dati SOLO se sono presenti nel 'MATERIALE DI RICERCA' o nel 'CONTESTO DEL GIOCO'. Se un dato manca, NON inventarlo per nessun motivo.

CITAZIONI CROSS-POST E STILE BLOG:
- TITOLO ACCATTIVANTE (H1): La primissima riga del testo deve essere un Titolo (H1) giornalistico, creativo e a effetto (es. "Lies of P: Il lato oscuro della fiaba di Collodi"). È SEVERAMENTE VIETATO usare titoli banali come "Benvenuti nel mondo di..." o incollare la dicitura asettica "una recensione completa e generale". Sii un vero copywriter!
- Usa il tono tipico di un blog. Se appropriato, nell'introduzione puoi salutare i lettori e fare riferimento agli ultimi articoli pubblicati (elencati in 'ULTIMI POST SUL BLOG') dicendo ad esempio: "Dopo avervi parlato di [Gioco Precedente], oggi ci dedicheremo... (ovviamente questo è solo un esempio!, cerca di essere creativo ma preciso in ciò che dici rispetto alla verità del blog e delle fonti)".
- Usa il 'CONTESTO DEL GIOCO' per menzionare giochi simili o vecchi capitoli dello stesso studio in modo naturale.
- PARAGONI VIDEOLUDICI (OPZIONALE E CONDIZIONALE): Inserisci paragoni con altri videogiochi SOLO SE hanno un reale senso critico ed editoriale (es. per ovvie somiglianze di gameplay, atmosfera o genere). Puoi usare i giochi presenti nel 'CONTESTO DEL GIOCO' o nel 'MATERIALE DI RICERCA'. 🚨 REGOLA AUREA: Se i giochi a tua disposizione non c'entrano nulla con il topic (es. paragonare uno sparatutto a un survival horror in modo illogico), NON FARE NESSUN PARAGONE. La naturalezza del testo viene prima di tutto.

CITAZIONE FONTI ESTERNE:
Cita le fonti ALLA FINE dell'articolo in un'apposita sezione. Non citare nomi di siti o giornalisti nel mezzo del discorso. NON INSERIRE LA CREDIBILITA' DELLA FONTE QUANDO LE RIPORTI.

MATERIALE DI RICERCA (Ricco di dettagli e lore):
{research_summary}

CONTESTO DEL GIOCO E GIOCHI SIMILI (Dal Knowledge Graph):
{kg_context}

ULTIMI POST SUL BLOG (Per introduzione e continuity):
{recent_posts}

PIANO EDITORIALE DA SEGUIRE (come riferimento):
{plan}

🚨 ZERO SPOILER SUL PIANO EDITORIALE:
È SEVERAMENTE VIETATO menzionare, riassumere o incollare il "PIANO EDITORIALE" o annunciare i "prossimi articoli" all'interno del testo. Il piano ti è fornito solo come briefing interno per strutturare l'articolo, QUINDI DEVI USARLO ASSOLUTAMENTE COME TRACCIA O RIFERIMENTO PER LA RECENSIONE, MA NON CITARLO ALLA FINE O RIPORTARLO. Chiudi sempre la recensione in modo naturale o con le fonti, senza mai fare spoiler sui giochi futuri del blog.
"""

    messages = [("system", system_prompt)]

    if human_feedback and human_feedback.lower() not in ["", "approve"]:
        user_message = (
            f"🚨 REVISIONE EDITORIALE OBBLIGATORIA 🚨\n"
            f"Il Direttore Responsabile ha rifiutato la bozza precedente e ti ha dato questo ordine diretto:\n"
            f"\"{human_feedback}\"\n\n"
            f"Riscrivi l'INTERA recensione obbedendo ciecamente a questo feedback. "
            f"Se ti ha chiesto un angolo diverso/un focus specifico, DEVI OBBLIGATORIAMENTE rispettare quella richiesta e focalizzarti pesantemente su quell'aspetto, anche se va contro il focus editoriale originale. Viceversa, se non è stato detto nulla in merito a focus editoriali/angoli dell'articolo, attieniti al piano editoriale/focus che hai/sai già. "
            f"Se ti ha chiesto una LINGUA DIVERSA (es. Inglese), l'intero testo generato DEVE essere in quella lingua. "
            f"Se ti ha chiesto di approfondire degli argomenti, fallo usando i dati di ricerca. "
            f"Formatta l'output in Markdown."
        )
    else:
        user_message = f"Scrivi l'articolo su '{topic}'. Ricorda la regola fondamentale: focalizzati pesantemente su '{review_angle}' rispettando il piano editoriale."

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
        "🚨 REGOLA FORMATO OUTPUT: Non 'incartare' la risposta. È SEVERAMENTE VIETATO usare tag XML o blocchi markdown. Fornisci ESCLUSIVAMENTE gli argomenti grezzi."
        "🚨 REGOLA APOSTROFI: NON usare MAI il backslash (\\) per fare l'escape degli apostrofi nei testi. Scrivi in modo normale (es. un'opinione) altrimenti il parser esplode."
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
        old_feedback = state.get('human_feedback', '').strip()
        auto_revision_msg = f"REVISIONE AUTOMATICA: {verdict.reason}. Elementi mancanti: {', '.join(verdict.missing_elements)}"
        combined_feedback = f"{old_feedback}\n\n🚨 {auto_revision_msg}" if old_feedback else auto_revision_msg

        return Command(
            update={
                "quality_passed": False,
                "revision_count": revision_count + 1,
                "reasoning_trace": reasoning,
                "human_feedback": combined_feedback
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
    topic = state['user_input']

    # Routing via LLM
    system_prompt = (
        "Sei un sistema di routing che classifica il feedback di un utente sulla recensione di un videogioco.\n"
        f"Il topic dell'articolo prodotto è '{topic}'.\n"
        "Scegli UNA tra queste decisioni:\n"
        "- 'need_research': L'utente vuole più dettagli, ma sullo stesso topic dell'articolo (es. 'parlami più dei boss', 'manca la lore', 'approfondiamo questa parte', 'focus sul gameplay'; senza specificare un nuovo topic/gioco).\n"
        "- 'change_topic': L'utente vuole scartare il gioco e recensirne un altro rispetto a quello che era il topic dell'articolo (es. 'cambia gioco', 'facciamo Zelda', 'recensiamo Elden Ring con focus sul gameplay'; quindi sta cambiando topic).\n"
        "- 'rewrite': L'utente vuole solo correzioni di stile o lunghezza (es. 'falla più corta', 'usa un tono più epico') sullo stesso topic dell'articolo.\n"
        "- 'approve': L'utente fa complimenti senza chiedere modifiche e/o fornire indicazioni specifiche (es. 'ottimo lavoro')."

        "\nREGOLE DI PRIORITÀ:\n"
        "1. Se il feedback menziona un videogioco diverso dal topic corrente, scegli SEMPRE 'change_topic'.\n"
        "2. Anche se l'utente specifica focus, approfondimenti o aspetti da trattare del nuovo gioco, SE il topic è diverso allora la decisione resta 'change_topic'.\n"
        "3. 'need_research' può essere scelto SOLO se il feedback riguarda il topic corrente (oppure viene chiesto un approfondimento senza nominare un nuovo topic).\n"
        "4. 'rewrite' può essere scelto SOLO se il feedback riguarda il testo già scritto sul topic corrente e non viene nominato un nuovo topic (vedi esempi sotto).\n"
        "5. 'approve' può essere scelto SOLO se non sono presenti richieste o istruzioni.\n"

        "ESEMPI:\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: recensiamo Silent Hill F\n"
        "Decisione: change_topic\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: recensiamo Silent Hill F con focus sulla storia e lore\n"
        "Decisione: change_topic\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: passiamo a Elden Ring\n"
        "Decisione: change_topic\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: approfondisci la lore\n"
        "Decisione: need_research\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: parlami di più dei boss\n"
        "Decisione: need_research\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: rendila più corta\n"
        "Decisione: rewrite\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: usa un tono più professionale\n"
        "Decisione: rewrite\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: ottimo lavoro\n"
        "Decisione: approve\n\n"

        "Topic corrente: Dark Souls\n"
        "Feedback: perfetto così\n"
        "Decisione: approve"
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

    # Se l'utente vuole cambiare gioco, dobbiamo fare un HARD RESET della memoria globale!
    if goto_node == "planner":
        update_data["user_input"] = feedback
        update_data["human_feedback"] = ""         # Svuotiamo il feedback così il Ricercatore farà la Fase 1!
        update_data["tool_outputs"] = {}           # Svuotiamo la memoria di ricerca del gioco precedente
        update_data["research_summary"] = ""       # Svuotiamo i riassunti vecchi
        update_data["draft_post"] = ""             # Cancelliamo la bozza vecchia
        update_data["planning_information"] = {}   # Resettiamo il piano

        _INTENT_CACHE.clear()
        _PLAN_CACHE.clear()

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

    similarity_catalog = format_similarity_catalog(kg_manager.get_catalog_for_similarity())

    system_prompt = (
        "Sei un analista dati senior. Il tuo compito è estrarre TUTTI i dati fattuali da una recensione e dai suoi appunti di ricerca per popolare un Knowledge Graph enciclopedico.\n"
        "🚨 REGOLA COMPLETEZZA: Usa gli appunti di ricerca per riempire tutti i buchi tecnici (Anno di uscita, Piattaforme, Studio di sviluppo, Genere, Personaggi) anche se la recensione non li cita esplicitamente.\n"
        "🚨 REGOLA ONTOLOGICA: Distingui rigorosamente i TITOLI dei giochi dai GENERI. 'Soulslike', 'Action RPG', 'Open World', 'Roguelike', 'Metroidvania', 'Survival Horror' sono GENERI e non giochi.\n"
        "🚨 REGOLA DI NORMALIZZAZIONE: Quando estrai il nome di un gioco, uno studio o un genere, scrivilo SEMPRE in Title Case (iniziali maiuscole, es. 'Survival Horror')."
        "🚨 REGOLA FONDAMENTALE SUI GIOCHI SIMILI: Non farti ingannare dall'introduzione dell'articolo. I giochi citati nei saluti iniziali come 'post precedenti' NON E' DETTO CHE SIANO giochi simili. Estrai solo i veri paragoni videoludici."
        "🚨 REGOLA PER L'ANGOLO: Analizza il TITOLO e il testo della 'RECENSIONE FINALE'.\n"
            "- Se l'articolo si concentra CHIARAMENTE su un aspetto specifico (lo capisci dal titolo, es. 'La Storia di...', 'Meccaniche di...'), scrivi il focus tematico nuovo (es. 'Analisi della Storia').\n"
            "- Se invece l'articolo è generale e tocca tutto, ricopia ESATTAMENTE il valore di 'ANGOLO ORIGINALE'.\n"
            "- NON inserire MAI generi del gioco (es. 'Survival Horror') in questo campo.\n"
        "🚨 REGOLA DELL'ARRAY VUOTO: Se l'articolo NON fa paragoni videoludici reali e diretti tra le meccaniche o la lore, devi TASSATIVAMENTE lasciare l'array 'similar_games' VUOTO []. Non inserire MAI i giochi citati nell'introduzione come recensioni passate, A MENO CHE NON SIANO DAVVERO POI GIOCHI SIMILI (lo capisci dal contesto della recensione generale). Meglio un array vuoto che un dato falso."
        "🚨 DEDUZIONE GIOCHI SIMILI:\n"
        "1. Cerca nel testo della recensione se l'autore ha citato esplicitamente altri giochi come paragoni DI SOMIGLIANZA e aggiungili all'elenco.\n"
        "2. IN AGGIUNTA ai giochi del testo, GUARDA TUTTO IL CATALOGO DELLE SOMIGLIANZE. Confronta i 'Generi' e le 'Meccaniche' del topic con quelli del catalogo. Se c'è una forte sovrapposizione (es. entrambi sono 'Survival Horror' o usano 'Stealth') o appartengono allo stesso ramo di genere/sviluppatore (o alla stessa serie/saga/franchise), inserisci i titoli dal catalogo nell'array 'similar_games', aggiungi anche questi titoli all'array 'similar_games'.\n"
        "3. 🚨 REGOLA DELL'ARRAY VUOTO: Se nel catalogo non c'è NULLA di logicamente paragonabile per meccaniche o genere, devi TASSATIVAMENTE lasciare l'array 'similar_games' VUOTO []. Non inserire MAI i giochi citati nell'introduzione come saluti o recensioni passate. Meglio un array vuoto che un paragone falso."
        "🚨 REGOLA ONTOLOGICA SUI TITOLI: Distingui rigorosamente i NOMI PROPRI dei giochi (es. 'Bloodborne', 'Persona 5') dalle CATEGORIE DESCRITTIVE o GENERI. "
        "Termini come 'Soulslike', 'Action RPG', 'Social Link-Heavy Games', 'Open World' o 'Story-driven' sono generi o meccaniche, NON sono titoli di videogiochi! Se incontri una categoria descrittiva, ignorala o inseriscila nei 'Generi' o 'Meccaniche', MAI nei 'Giochi Simili'.\n"
    )

    user_prompt = (
        f"ANGOLO ORIGINALE: '{plan_info.get('review_angle', 'Generico')}'\n"
        f"APPUNTI DI RICERCA:\n{research_summary}\n\n"
        f"RECENSIONE FINALE:\n{draft}\n\n"
        f"CATALOGO DELLE SOMIGLIANZE (Usa Generi e Meccaniche per dedurre i link):\n{truncate_text(str(similarity_catalog), 15000)}\n\n"
        f"Estrai tutte le entità chiave per il database (Generi, Studi, Piattaforme, Anno, Boss, Meccaniche, Personaggi, Giochi Simili, Opinioni, Fonti)."
    )

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
        review_angle=entities.review_angle,
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
