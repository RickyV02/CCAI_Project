from langchain.tools import tool
from langchain_tavily import TavilySearch
from youtube_transcript_api import YouTubeTranscriptApi
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq as _ChatGroq
from pydantic import BaseModel, Field

class ArticleSummary(BaseModel):
    """Schema per il riassunto strutturato di un articolo di gaming."""
    summary: str = Field(description="Riassunto dei fatti principali: trama, meccaniche, impressioni generali.")
    key_facts: str = Field(description="Fatti concreti e specifici: nomi esatti di boss, aree, armi, meccaniche, voti numerici, pro e contro.")

load_dotenv()

# Nelle prossime fasi di sviluppo, questi tool saranno implementati in maniera procedurale con logiche reali.

search_tool = TavilySearch(max_results=5, include_raw_content=False)

@tool
def rag_retrieval_tool(query: str) -> str:
    """Mock di ricerca nei documenti locali (RAG)."""
    return f"Estratto dal database dei documenti per: {query}"

@tool
def knowledge_graph_tool(entity: str) -> str:
    """Mock per la ricerca di entità nel Knowledge Graph."""
    return f"Informazioni dal KG per l'entità: {entity}"

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
    """Recupera e riassume la trascrizione completa di un video YouTube dato il suo URL."""
    try:
        if "v=" in video_url:
            video_id = video_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
        else:
            return "Errore: URL YouTube non riconosciuto."

        ytt = YouTubeTranscriptApi()
        transcript_data = ytt.fetch(video_id)
        testo_intero = " ".join([t.text for t in transcript_data])

        # Se il testo è breve, lo restituiamo direttamente
        if len(testo_intero) <= 8000:
            return f"<youtube_transcript>\n{testo_intero}\n</youtube_transcript>"

        # Altrimenti usiamo Map-Reduce per ridurne la dimensione
        llm_fast = _ChatGroq(model="llama-3.1-8b-instant", temperature=0.0) # Potremmo metterlo come tool (sistema multi-agent)
        chunk_size = 8000
        chunks = [testo_intero[i:i+chunk_size] for i in range(0, len(testo_intero), chunk_size)]

        chunk_summaries = []
        for chunk in chunks:
            prompt = f"""Estrai i fatti concreti gaming da questa trascrizione video:
            nomi di boss, aree, armi, meccaniche, opinioni del creator, voti.
            Testo: {chunk}
            Massimo 150 parole."""
            summary = llm_fast.invoke(prompt).content
            chunk_summaries.append(summary)

        if len(chunk_summaries) > 1:
            final_prompt = f"""Unisci questi riassunti in un unico testo coerente.
            Elimina i duplicati. Massimo 300 parole.
            {chr(10).join(chunk_summaries)}"""
            return f"<youtube_transcript>\n{llm_fast.invoke(final_prompt).content}\n</youtube_transcript>"

        return f"<youtube_transcript>\n{chunk_summaries[0]}\n</youtube_transcript>"

    except Exception as e:
        return f"Impossibile estrarre la trascrizione: {e}"

@tool
def article_summarizer(raw_text: str, topic: str) -> str:
    """Riassume un articolo lungo estraendo fatti concreti gaming: boss, aree, armi, meccaniche, voti, pro e contro. Usalo per processare raw content lungo prima di passarlo al writer."""
    try:
        llm_fast = _ChatGroq(model="llama-3.1-8b-instant", temperature=0.0) # Come per i video di yt, questo potremmo metterlo come tool (quindi creare un tool che fa riassunti di testo, rimuovendo la stessa logica da entrambi i tool e centralizzandola in un unico tool di summarization multi-purpose).
        #In questo modo potremmo usarlo anche per riassumere i contenuti lunghi recuperati dal rag_retrieval_tool o dal knowledge_graph_tool, se necessario.

        structured_summarizer = llm_fast.with_structured_output(ArticleSummary)

        chunk_size = 8000
        chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]

        all_summaries = []
        all_facts = []

        for chunk in chunks:
            try:
                result = structured_summarizer.invoke(
                    f"Analizza questo testo gaming relativo a '{topic}' ed estrai un riassunto e i fatti concreti specifici: nomi esatti di boss, aree, armi, meccaniche, voti numerici, pro e contro.\n\nTesto:\n{chunk}"
                )
                all_summaries.append(result.summary)
                all_facts.append(result.key_facts)
            except Exception:
                all_summaries.append("")
                all_facts.append("")

        combined_summary = " ".join([s for s in all_summaries if s])
        combined_facts = "\n".join([f for f in all_facts if f])

        return f"<summary>\n{combined_summary}\n</summary>\n\n<key_facts>\n{combined_facts}\n</key_facts>"

    except Exception as e:
        return f"Errore summarizer: {e}"
