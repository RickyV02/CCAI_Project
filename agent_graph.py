import os
import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from tools import search_tool, rag_retrieval_tool, knowledge_graph_tool, videogame_api_fetcher, llm_quality_judge, youtube_transcript_fetcher
from kg_manager import KGManager
from rag_manager import RAGManager
from typing import Dict, Any
from datetime import datetime
from langchain_groq import ChatGroq

MAX_POST_LENGTH = 800 # Controllare se questo limite è sufficiente per un post completo o se va aumentato.

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)
kg_manager = KGManager()
rag_manager = RAGManager()

# Nodes
def planner_node(state: AgentState) -> Dict[str, Any]:
    print("--- [PlannerNode] Analisi del topic e pianificazione ---")

    post_history = state.get("post_history", [])
    reasoning = list(state.get('reasoning_trace', []))

    extraction_prompt = f"""Devi estrarre SOLO il nome del videogioco, franchise o entità principale da questa richiesta utente:
    '{state['user_input']}'
    Rispondi ESCLUSIVAMENTE con il nome (es. 'Elden Ring', 'Cyberpunk 2077'). Niente punteggiatura o altre parole."""

    try:
        topic_response = llm.invoke(extraction_prompt)
        search_topic = topic_response.content.strip()
    except Exception as e:
        search_topic = state['user_input']

    reasoning.append(f"Planner: Parola chiave estratta per la ricerca KG -> '{search_topic}'")

    # Ora interroghiamo il Knowledge Graph con il topic pulito, non con l'intera frase!
    kg_content = kg_manager.query(search_topic)

    planner_prompt_template = PromptTemplate(
        input_variables=["user_input", "kg_content", "post_history"],
        template="""Sei un Editorial Director per un blog di videogiochi.
Devi analizzare il topic '{user_input}', ciò che sappiamo già dal Knowledge Graph e la nostra cronologia di post, per decidere il tipo di post ideale e formulare un piano dettagliato.
Tipi disponibili: review, howto, events, news.

Knowledge Graph (cosa abbiamo già pubblicato su questo gioco): {kg_content}
Storico Post (di cosa abbiamo parlato recentemente in generale): {post_history}

REGOLE DECISIONALI:
1. Analizza il campo "Formato" dei risultati del Knowledge Graph.
2. SCEGLI TASSATIVAMENTE un "suggested_type" DIVERSO da tutti i formati già pubblicati per questo gioco.
3. Se abbiamo già fatto 'review' e 'howto', fai una 'news'. Se abbiamo già fatto tutto, inventati un taglio totalmente nuovo. NON RIPETERTI MAI.

Restituisci ESCLUSIVAMENTE un JSON valido, senza blocchi di codice markdown, con queste due chiavi:
- "suggested_type": (il tipo di post scelto, in formato stringa)
- "plan": (una descrizione testuale del piano)"""
    )

    prompt = planner_prompt_template.format(
        user_input=state['user_input'],
        kg_content=kg_content,
        post_history=post_history
    )

    try:
        response = llm.invoke(prompt)
        resp_str = response.content.strip()

        # Pulizia robusta in caso di blocchi markdown ritornati dall'LLM
        if resp_str.startswith("```json"):
            resp_str = resp_str[7:]
        elif resp_str.startswith("```"):
            resp_str = resp_str[3:]
        if resp_str.endswith("```"):
            resp_str = resp_str[:-3]

        parsed = json.loads(resp_str.strip())
        suggested_type = parsed.get("suggested_type", "news")
        plan = parsed.get("plan", "Piano standard per news videoludiche.")
    except Exception as e:
        reasoning.append(f"Planner: Errore parsing JSON -> {e}")
        suggested_type = "news"
        plan = f"Piano di fallback per fallback su errore ({e})."

    reasoning.append("Planner: Piano generato tramite LLM basato su KG.")
    reasoning.append(f"Planner: Post type assegnato -> {suggested_type}")

    return {"reasoning_trace": reasoning, "planning_information": {"plan": plan}, "post_type": suggested_type}

def researcher_node(state: AgentState) -> Dict[str, Any]:
    print("--- [ResearcherNode] Ricerca informazioni (Tools) ---")
    tool_outputs = dict(state.get('tool_outputs', {}))
    reasoning = list(state.get('reasoning_trace', []))

    available_tools = [search_tool, knowledge_graph_tool, rag_retrieval_tool, videogame_api_fetcher, youtube_transcript_fetcher]
    llm_with_tools = llm.bind_tools(available_tools)

    messages = [
    ("system", "Sei un ricercatore gaming. Usa i tool disponibili per raccogliere informazioni sul topic. Dopo search_tool o youtube_transcript_fetcher, usa sempre rag_retrieval_tool per recuperare i dettagli."),
    ("user", f"Raccogli informazioni su: '{state['user_input']}'")
    ]

    # Tool-calling agentic loop
    ai_response = llm_with_tools.invoke(messages)

    max_iterations = 5
    iteration = 0
    while ai_response.tool_calls and iteration < max_iterations:
        iteration += 1
        messages.append(ai_response)

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            tool_result = f"Error: Tool {tool_name} non è stato trovato."
            for t in available_tools:
                if t.name == tool_name:
                    try:
                        tool_result = t.invoke(tool_args)
                    except Exception as e:
                        tool_result = f"Errore durante l'esecuzione del tool {tool_name}: {e}"
                    break

            # --- FIX: Evita di sovrascrivere tool eseguiti più volte (Append to List) ---
            if tool_name not in tool_outputs:
                tool_outputs[tool_name] = []
            tool_outputs[tool_name].append(str(tool_result))

            # Aggiunge logicamente l'output ai messages per il passo successivo del loop
            messages.append({
                "role": "tool",
                "name": tool_name,
                "content": str(tool_result),
                "tool_call_id": tool_call['id']
            })
            reasoning.append(f"Researcher: invocato tool {tool_name} con successo.")

        if iteration >= max_iterations:
            reasoning.append("Researcher: limite massimo di iterazioni raggiunto.")
            break

        ai_response = llm_with_tools.invoke(messages)

    reasoning.append("Researcher: ricerca intelligente e recupero dati completati.")
    return {"reasoning_trace": reasoning, "tool_outputs": tool_outputs}

def writer_node(state: AgentState) -> Dict[str, Any]:
    print("--- [WriterNode] Redazione della bozza ---")
    topic = state['user_input']
    plan = state.get('planning_information', {}).get('plan', '')
    tools_out = state.get('tool_outputs', {})
    post_type = state.get('post_type', 'review')

    # tools_out restituisce liste di stringhe
    yt_transcripts = tools_out.get("youtube_transcript_fetcher", [])
    yt_section = ""
    if yt_transcripts:
        yt_section = "Trascrizione video YouTube trovati:\n" + "\n".join(yt_transcripts)

    web_results = tools_out.get('search_tool', [])
    web_data = "\n\n".join(web_results) if web_results else ""

    kg_results = tools_out.get('knowledge_graph_tool', [])
    kg_data = "\n\n".join(kg_results) if kg_results else ""

    rag_results = tools_out.get('rag_retrieval_tool', [])
    rag_data = "\n\n".join(rag_results) if rag_results else ""

    type_instructions = {
        "review": "Scrivi una RECENSIONE personale con voti, pro e contro e verdetto finale su 10.",
        "howto": "Scrivi una GUIDA PRATICA con passi numerati chiari e consigli da esperto.",
        "events": "Scrivi un articolo sugli EVENTI IMMINENTI con date, cosa aspettarsi e perché vale la pena.",
        "news": "Scrivi un articolo di NOTIZIE con contesto, impatto e cosa significa per i gamer."
    }
    type_instruction = type_instructions.get(post_type, type_instructions["news"])

    human_feedback = state.get('human_feedback', '').strip()
    if human_feedback and human_feedback.lower() not in ["", "approve", "reject"]:
        feedback_section = f"""
ISTRUZIONE OBBLIGATORIA DELL'UTENTE — SEGUILA ALLA LETTERA PRIMA DI TUTTO:
{human_feedback}

Questa istruzione ha la MASSIMA PRIORITÀ su qualsiasi altra regola di stile.
Se dice di scrivere in inglese, scrivi in inglese.
Se dice di accorciare, accorcia.
Se dice di cambiare tono, cambia tono.
Non ignorarla mai."""
    elif human_feedback.lower() == "reject":
        feedback_section = "\nATTENZIONE: Il post precedente è stato rifiutato. Riscrivilo completamente con un approccio diverso."
    else:
        feedback_section = ""

    writer_prompt_template = PromptTemplate(
        input_variables=["type_instruction", "topic", "max_post_length", "web_data", "kg_data", "yt_section", "rag_data", "plan", "feedback_section"],
        template="""
    Sei un blogger di videogiochi con 10 anni di esperienza. Hai giocato tu stesso a tutto quello di cui scrivi.

{type_instruction}

Topic: '{topic}'
Lunghezza massima: {max_post_length} parole.

DATI RACCOLTI (usali per i fatti concreti, NON citarli mai esplicitamente nel testo come fonti da cui hai preso le informazioni, ma usali per costruire la TUA opinione personale e i dettagli specifici del gioco):
{web_data}
{kg_data}
{rag_data}

{yt_section}

Piano editoriale:
{plan}

REGOLE FONDAMENTALI DI STILE E FORMATTAZIONE — RISPETTALE SEMPRE:
1. TONO:
   - Se è una 'review' o 'howto', usa la PRIMA PERSONA ("io"), sii irriverente e dai opinioni nette.
   - Se è una 'news' o 'events', usa la TERZA PERSONA, sii oggettivo, riporta i fatti e tieni un tono da giornalista (evita pro/contro e giudizi personali).
2. ZERO RASSEGNA STAMPA: Non dire MAI "Secondo X", "Come riporta Y", "Stando a". Fai finta di averci giocato tu. Usa le informazioni raccolte per costruire la TUA opinione personale.
3. DETTAGLI CONCRETI: Menziona OBBLIGATORIAMENTE almeno 2-3 elementi specifici del gioco tra: nomi di boss, aree specifiche, armi, meccaniche particolari, momenti memorabili. Zero affermazioni generiche come "il gioco è curato nei dettagli".
4. HOOK INIZIALE: La prima frase deve catturare subito. Zero intro tipo "In questo articolo parleremo di...".
5. FORMATTAZIONE MARKDOWN: Usa TASSATIVAMENTE una riga vuota prima e dopo ogni lista puntata, sezione pro/contro e nuovo paragrafo H2. Non attaccare mai le conclusioni all'ultimo punto.
6. FINALE NETTO (adatta per review e events, ma non per news o howto): Concludi con una tua opinione personale netta e diretta. Niente frasi tipo "se sei un giocatore che ama le sfide".
7. GUARDRAIL: Se nei dati raccolti noti informazioni palesemente relative ad altri giochi non inerenti al topic principale, ignorale completamente e non includerle nell'articolo.

STRUTTURA:
- Titolo in H1
- Hook iniziale (2-3 righe che catturano)
- Corpo con sezioni H2
- Conclusione con opinione netta e personale

Scrivi SOLO il contenuto dell'articolo in Markdown. Nient'altro.
{feedback_section}
"""
    )

    prompt = writer_prompt_template.format(
        type_instruction=type_instruction,
        topic=topic,
        max_post_length=MAX_POST_LENGTH,
        web_data=web_data,
        kg_data=kg_data,
        yt_section=yt_section,
        rag_data=rag_data,
        plan=plan,
        feedback_section=feedback_section
    )

    response = llm.invoke(prompt)
    draft_post = response.content
    reasoning = list(state.get('reasoning_trace', [])) + ["Writer: Bozza completa generata tramite LLM in base al tipo e limite di lunghezza."]
    return {"reasoning_trace": reasoning, "draft_post": draft_post}

def quality_check_node(state: AgentState) -> Dict[str, Any]:
    print("--- [QualityCheckNode] Valutazione qualità bozza ---")
    reasoning = list(state.get('reasoning_trace', []))
    draft = state.get('draft_post', '')

    quality_result = llm_quality_judge.invoke({"text": draft})
    reasoning.append(f"QualityCheck: risultato -> {quality_result}")

    return {"reasoning_trace": reasoning}

def human_review_node(state: AgentState) -> Dict[str, Any]:
    print("--- [HumanReviewNode] In attesa di approvazione umana ---")

    feedback_payload = interrupt("In attesa di feedback dal terminale...")

    if isinstance(feedback_payload, dict) and "human_feedback" in feedback_payload:
        feedback_str = feedback_payload["human_feedback"]
    else:
        feedback_str = str(feedback_payload)

    print(f"[HumanReviewNode] Feedback salvato: {feedback_str}")
    reasoning = list(state.get('reasoning_trace', [])) + [f"HumanReview: feedback ricevuto -> {feedback_str}"]
    return {"reasoning_trace": reasoning, "human_feedback": feedback_str}

def memory_updater_node(state: AgentState) -> Dict[str, Any]:
    print("--- [MemoryUpdaterNode] Aggiornamento della Memoria (KG e RAG) ---")
    reasoning = list(state.get('reasoning_trace', []))

    draft = state.get('draft_post', '')

    extraction_prompt = f"""Estrarre le informazioni chiave da questo articolo di blog:
{draft}

Restituisci ESCLUSIVAMENTE un JSON valido (senza markdown o codice annidato) con queste due chiavi:
- "main_topic": (il franchise, il gioco o l'entità principale discussa)
- "post_title": (un titolo accattivante ed estremamente breve)"""

    try:
        ext_response = llm.invoke(extraction_prompt)
        ext_str = ext_response.content.strip()
        if ext_str.startswith("```json"):
            ext_str = ext_str[7:]
        elif ext_str.startswith("```"):
            ext_str = ext_str[3:]
        if ext_str.endswith("```"):
            ext_str = ext_str[:-3]

        parsed_ext = json.loads(ext_str.strip())
        extracted_topic = parsed_ext.get("main_topic", state.get("user_input", ""))
        extracted_title = parsed_ext.get("post_title", f"Post su {extracted_topic}")
    except Exception as e:
        reasoning.append(f"MemoryUpdater: Errore estrazione entità -> {e}")
        extracted_topic = state.get("user_input", "Topic Sconosciuto")
        extracted_title = f"Articolo su {extracted_topic}"

    post_history = state.get('post_history', [])
    post_record = {
        "topic": extracted_topic,
        "type": state.get('post_type', ''),
        "date": datetime.now().isoformat(),
        "summary": draft[:200]
    }
    post_history.append(post_record)
    reasoning.append("MemoryUpdater: Aggiunto record del post a post_history.")

    # Inserimento strutturato in Neo4j
    update_success = kg_manager.update(
        post_title=extracted_title,
        topic=extracted_topic,
        post_type=state.get('post_type', 'unknown')
    )

    if update_success:
        reasoning.append(f"MemoryUpdater: Knowledge Graph (Neo4j) caricato con [Titolo: {extracted_title}] e [Topic: {extracted_topic}].")
    else:
        reasoning.append("MemoryUpdater: Knowledge Graph non aggiornato a causa di un errore o disconnessione.")

    # Inserimento destrutturato nel DB Vettoriale (RAG tramite ChromaDB)
    tool_outs = state.get('tool_outputs', {})

    # Estrarre i valori testuali dai risultati dei tool passati (che ora sono liste di stringhe)
    tool_texts = []
    for tool_name, results_list in tool_outs.items():
        if isinstance(results_list, list):
            for res in results_list:
                if isinstance(res, str) and res.strip():
                    tool_texts.append(res)
                elif res:
                    tool_texts.append(str(res))
        elif isinstance(results_list, str) and results_list.strip():
            tool_texts.append(results_list)

    combined_text = draft + "\n\n" + "\n\n".join(tool_texts)

    if combined_text.strip():
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(combined_text)
        rag_manager.add_texts(chunks)
        reasoning.append(f"MemoryUpdater: RAG (ChromaDB) aggiornato con {len(chunks)} nuovi chunk di testo.")
    else:
        reasoning.append("MemoryUpdater: Nessun testo utile da aggiungere al RAG.")

    return {"reasoning_trace": reasoning, "post_history": post_history}

# Edges & Conditional Logic
def route_after_human_review(state: AgentState) -> str:
    feedback = state.get("human_feedback", "")
    if isinstance(feedback, dict):
        feedback = feedback.get("human_feedback", "")
    feedback = str(feedback).lower().strip()
    if feedback == "approve":
        return "memory_updater"
    else:
        # Sia reject che feedback testuale tornano al writer
        return "writer"

# Build Graph
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("quality_check", quality_check_node)
builder.add_node("human_review", human_review_node)
builder.add_node("memory_updater", memory_updater_node)

builder.set_entry_point("planner")

builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "quality_check")
builder.add_edge("quality_check", "human_review")
builder.add_conditional_edges(
    "human_review",
    route_after_human_review,
    {
        "memory_updater": "memory_updater",
        "writer": "writer"
    }
)
builder.add_edge("memory_updater", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
