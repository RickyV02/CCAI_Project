import os
import re
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from tools import search_tool, rag_retrieval_tool, knowledge_graph_tool, videogame_api_fetcher, llm_quality_judge, youtube_transcript_fetcher, article_summarizer
from typing import Dict, Any
from datetime import datetime
from langchain_groq import ChatGroq

MAX_POST_LENGTH = 800 # Controllare se questo limite è sufficiente per un post completo o se va aumentato.

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

# Nodes
def planner_node(state: AgentState) -> Dict[str, Any]:
    print("--- [PlannerNode] Analisi del topic e pianificazione ---")

    post_history = state.get("post_history", [])
    type_counts = {"review": 0, "howto": 0, "events": 0, "news": 0}
    for p in post_history:
        t = p.get("type", "")
        if t in type_counts:
            type_counts[t] += 1
    suggested_type = min(type_counts, key=type_counts.get) # Questa logica verrà sostituita quando useremo l'LLM per suggerire il tipo di post (in particolare quando verrà implementato il rag_retrieval_tool per recuperare i post passati più rilevanti al topic attuale).

    prompt = f"Sei un Content Strategist per un blog di videogiochi. Basandoti su questo topic: '{state['user_input']}', pianifica una sequenza dettagliata di 3 post gaming correlati. Tieni conto che il tipo suggerito meno utilizzato di recente è '{suggested_type}'. I tipi disponibili sono: review, howto, events, news. Restituisci solo l'outline del piano."
    response = llm.invoke(prompt)
    reasoning = list(state.get('reasoning_trace', [])) + ["Planner: Piano generato tramite LLM.", f"Planner: Post type assegnato -> {suggested_type}"]
    return {"reasoning_trace": reasoning, "planning_information": {"plan": response.content}, "post_type": suggested_type}

def researcher_node(state: AgentState) -> Dict[str, Any]:
    print("--- [ResearcherNode] Ricerca informazioni (Tools) ---")
    tool_outputs = dict(state.get('tool_outputs', {}))
    reasoning = list(state.get('reasoning_trace', []))

    # Ricerca web standard
    query_prompt = f"""Sei un assistente per la ricerca web.
    Topic dell'utente: '{state['user_input']}'

    Genera UNA SOLA query di ricerca in INGLESE per trovare informazioni accurate.

    REGOLE TASSATIVE:
    1. Massimo 4-5 parole totali.
    2. Includi il nome esatto e specifico del gioco (es. "Shadow of the Erdtree", non solo "DLC Elden Ring").
    3. NON fare elenchi di parole chiave.
    4. La query deve essere in INGLESE.
    5. Restituisci SOLO la stringa di ricerca. Niente virgolette, niente spiegazioni."""

    try:
        # Prima genera la query
        search_query = llm.invoke(query_prompt).content.strip()
        reasoning.append(f"Researcher: Query generata -> {search_query}")

        # Poi esegue la ricerca con Tavily usando la query appena creata
        web_res = search_tool.invoke({"query": search_query})
        reasoning.append("Researcher: ricerca web completata.")
    except Exception as e:
        web_res = []
        reasoning.append(f"Researcher: errore ricerca web -> {e}")

    # Tavily ritorna un dict con 'results' dentro, oppure direttamente una lista
    articles = web_res.get('results', []) if isinstance(web_res, dict) else (web_res if isinstance(web_res, list) else [])

    summaries = []
    for article in articles:
        if isinstance(article, dict):
            raw = article.get('raw_content', '') or article.get('content', '')
            url = article.get('url', '')
            if raw:
                try:
                    summary = article_summarizer.invoke({
                        "raw_text": raw,
                        "topic": state['user_input']
                    })
                    summaries.append(f"Fonte: {url}\n{summary}")
                    reasoning.append(f"Researcher: articolo riassunto da {url}")
                except Exception as e:
                    summaries.append(f"Fonte: {url}\nErrore: {e}")

    if summaries:
        llm_fast = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0) #Anche per questo, potremmo fare un tool
        meta_prompt = f"""Sei un editor gaming esperto. Hai questi riassunti di articoli su '{state['user_input']}':

{chr(10).join(summaries)}

Crea UN SOLO documento consolidato con:
1. FATTI PRINCIPALI: punti chiave dell'esperienza
2. DETTAGLI CONCRETI: nomi di boss, aree, armi, meccaniche, voti
3. PRO E CONTRO: quelli più citati

Elimina i duplicati. Massimo 400 parole."""
        tool_outputs['web_search'] = llm_fast.invoke(meta_prompt).content
        reasoning.append(f"Researcher: meta-summary di {len(summaries)} articoli generato.")
    elif articles:
        contents = []
        for article in articles:
            if isinstance(article, dict):
                content = article.get('content', '')
                url = article.get('url', '')
                if content:
                    contents.append(f"Fonte: {url}\n{content}")
        tool_outputs['web_search'] = "\n\n---\n\n".join(contents) if contents else str(web_res)
    else:
        tool_outputs['web_search'] = str(web_res)

    # Knowledge Graph mock
    kg_res = knowledge_graph_tool.invoke({"entity": state['user_input']})
    tool_outputs['kg_search'] = kg_res

    # Cerca URL YouTube nei risultati con regex invece di usare l'LLM
    yt_url = None
    web_res_str = str(articles) if articles else ""
    yt_matches = re.findall(r'https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)[\w-]+', web_res_str)
    if yt_matches:
        yt_url = yt_matches[0]

    tool_outputs['youtube_transcript'] = ""
    if yt_url:
        try:
            transcript = youtube_transcript_fetcher.invoke({"video_url": yt_url})
            tool_outputs['youtube_transcript'] = transcript
            reasoning.append(f"Researcher: trovato e trascritto video YouTube -> {yt_url}")
        except Exception as e:
            tool_outputs['youtube_transcript'] = f"Errore trascrizione: {e}"
            reasoning.append(f"Researcher: errore trascrizione YouTube -> {e}")
    else:
        reasoning.append("Researcher: nessun URL YouTube trovato nei risultati web.")

    return {"reasoning_trace": reasoning, "tool_outputs": tool_outputs}

def writer_node(state: AgentState) -> Dict[str, Any]:
    print("--- [WriterNode] Redazione della bozza ---")
    topic = state['user_input']
    plan = state.get('planning_information', {}).get('plan', '')
    tools_out = state.get('tool_outputs', {})
    post_type = state.get('post_type', 'review')
    yt_transcript = tools_out.get("youtube_transcript", "")
    yt_section = f"Trascrizione video YouTube trovato:\n{yt_transcript}" if yt_transcript else ""

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

    prompt = f"""
    Sei un blogger di videogiochi con 10 anni di esperienza. Hai giocato tu stesso a tutto quello di cui scrivi.

{type_instruction}

Topic: '{topic}'
Lunghezza massima: {MAX_POST_LENGTH} parole.

DATI RACCOLTI (usali per i fatti concreti, NON citarli mai esplicitamente nel testo):
{tools_out.get('web_search', '')}

{yt_section}

Piano editoriale:
{plan}

REGOLE FONDAMENTALI DI STILE E FORMATTAZIONE — RISPETTALE SEMPRE:
1. TONO: Scrivi in PRIMA PERSONA ("io"). Sii appassionato, irriverente e usa gergo da vero gamer (es. "tryhardare", "run punitive", "drop rate", "hitbox", "frame perfect"). Non essere formale.
2. ZERO RASSEGNA STAMPA: Non dire MAI "Secondo X", "Come riporta Y", "Stando a". Fai finta di averci giocato tu. Usa le informazioni raccolte per costruire la TUA opinione personale.
3. DETTAGLI CONCRETI: Menziona OBBLIGATORIAMENTE almeno 2-3 elementi specifici del gioco tra: nomi di boss, aree specifiche, armi, meccaniche particolari, momenti memorabili. Zero affermazioni generiche come "il gioco è curato nei dettagli".
4. HOOK INIZIALE: La prima frase deve catturare subito. Zero intro tipo "In questo articolo parleremo di...".
5. FORMATTAZIONE MARKDOWN: Usa TASSATIVAMENTE una riga vuota prima e dopo ogni lista puntata, sezione pro/contro e nuovo paragrafo H2. Non attaccare mai le conclusioni all'ultimo punto.
6. FINALE NETTO: Concludi con una tua opinione personale netta e diretta. Niente frasi tipo "se sei un giocatore che ama le sfide".
7. GUARDRAIL: Se nei dati raccolti noti informazioni palesemente relative ad altri giochi non inerenti al topic principale, ignorale completamente e non includerle nell'articolo.

STRUTTURA:
- Titolo in H1
- Hook iniziale (2-3 righe che catturano)
- Corpo con sezioni H2
- Conclusione con opinione netta e personale

Scrivi SOLO il contenuto dell'articolo in Markdown. Nient'altro.
{feedback_section}
"""

    response = llm.invoke(prompt)
    draft_post = response.content
    reasoning = list(state.get('reasoning_trace', [])) + ["Writer: Bozza completa generata tramite LLM in base al tipo e limite di lunghezza."]
    return {"reasoning_trace": reasoning, "draft_post": draft_post}

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

def kg_updater_node(state: AgentState) -> Dict[str, Any]:
    print("--- [KGUpdaterNode] Aggiornamento del Knowledge Graph ---")
    reasoning = list(state.get('reasoning_trace', [])) + ["KGUpdater: Knowledge Graph aggiornato."]

    post_history = state.get('post_history', [])
    post_record = {
        "topic": state.get('user_input', ''),
        "type": state.get('post_type', ''),
        "date": datetime.now().isoformat(),
        "summary": state.get('draft_post', '')[:200]
    }
    post_history.append(post_record)
    reasoning.append("KGUpdater: Aggiunto record del post a post_history.")

    return {"reasoning_trace": reasoning, "post_history": post_history}

# Edges & Conditional Logic
def route_after_human_review(state: AgentState) -> str:
    feedback = state.get("human_feedback", "")
    if isinstance(feedback, dict):
        feedback = feedback.get("human_feedback", "")
    feedback = str(feedback).lower().strip()
    if feedback == "approve":
        return "kg_updater"
    else:
        # Sia reject che feedback testuale tornano al writer
        return "writer"

# Build Graph
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("human_review", human_review_node)
builder.add_node("kg_updater", kg_updater_node)

builder.set_entry_point("planner")

builder.add_edge("planner", "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "human_review")
builder.add_conditional_edges(
    "human_review",
    route_after_human_review,
    {
        "kg_updater": "kg_updater",
        "writer": "writer"
    }
)
builder.add_edge("kg_updater", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
