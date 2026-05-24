from langchain.tools import tool
from langchain_tavily import TavilySearch
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as _ChatGroq
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_manager import RAGManager
from kg_manager import KGManager

load_dotenv()

# Instanziazione del RAGManager (database vettoriale locale Chroma)
rag_instance = RAGManager()

# Instanziazione del KGManager (database a grafo remoto/locale Neo4j)
kg_instance = KGManager()

@tool
def search_tool(query: str) -> str:
    """Ricerca sul web informazioni aggiornate. Scarica il contenuto completo dei siti trovati e li salva nel database vettoriale locale per permettere ricerche di dettaglio."""
    try:
        tavily = TavilySearch(max_results=3, include_raw_content=True)
        results = tavily.invoke({"query": query})

        combined_text = ""
        for res in results:
            if isinstance(res, dict):
                content = res.get("raw_content") or res.get("content") or ""
            else:
                content = getattr(res, 'page_content', str(res))

            if content:
                combined_text += content + "\n\n"

        if not combined_text.strip():
            return "Nessun risultato utile trovato sul web."

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(combined_text)

        rag_instance.add_texts(chunks)

        return "Ricerca web completata. I contenuti completi dei siti sono stati scaricati e salvati nel database vettoriale locale (ChromaDB). Usa IMMEDIATAMENTE il tool 'rag_retrieval_tool' facendo query specifiche per estrarre i dettagli che ti servono."
    except Exception as e:
        return f"Errore durante la ricerca web: {e}"

@tool
def rag_retrieval_tool(query: str) -> str:
    """Search the local knowledge base for game mechanics, lore, internal blog guidelines, and past reviews. Use this tool BEFORE searching the web if you need to know deep factual mechanics of a game we already documented."""
    return rag_instance.retrieve(query)

@tool
def knowledge_graph_tool(entity: str) -> str:
    """Search the internal Knowledge Graph for structured relationships and entities (e.g. characters, developers, past covered topics). Use this to avoid factual errors regarding game relationships."""
    return kg_instance.query(entity)

@tool
def videogame_api_fetcher(game_name: str) -> str:
    """Mock di una chiamata API (es. IGDB) per recuperare dati fattuali su un videogioco. Per adesso è un placeholder"""
    return f"Dati API: {game_name} ha venduto 1M di copie, voto Metacritic 90/100."

@tool
def llm_quality_judge(text: str) -> str:
    """
    Questo tool simula un LLM fine-tunato che fa da giudice di qualità.
    Per ora restituisce casualmente 'APPROVED' o 'NEEDS REVISION'.
    In futuro useremo un modello open-source fine-tunato su esempi di articoli gaming di alta qualità per valutare se la bozza è pronta o necessita di ulteriori revisioni.
    """
    if "Elden Ring" in text:
        return "APPROVED"
    return "NEEDS REVISION"

@tool
def youtube_transcript_fetcher(video_url: str) -> str:
    """Recupera la trascrizione completa di un video YouTube dato il suo URL e la salva nel database vettoriale locale."""
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return "Errore: URL YouTube non riconosciuto."

        ytt = YouTubeTranscriptApi()
        transcript_data = ytt.fetch(video_id)
        raw_text = " ".join([t.text for t in transcript_data])

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(raw_text)

        rag_instance.add_texts(chunks)

        word_count = len(raw_text.split())
        return f"Trascrizione completa ({word_count} parole) scaricata e salvata con successo nel database vettoriale locale (ChromaDB). Per leggere le informazioni, utilizza IMMEDIATAMENTE il tool 'rag_retrieval_tool' facendo query specifiche su ciò che stai cercando."

    except Exception as e:
        return f"Impossibile estrarre la trascrizione: {e}"
