from langchain.tools import tool
from langchain_tavily import TavilySearch
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag_manager import RAGManager
from kg_manager import KGManager
from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP
import re
from bs4 import BeautifulSoup
import trafilatura

rag_manager = RAGManager()
kg_manager = KGManager()


@tool
def search_tool(query: str) -> str:
    """
    Esegue una ricerca web usando Tavily. Usa questo tool per trovare su internet news, recensioni o guide recenti.

    Args:
        query (str): La stringa esatta di ricerca da inviare al motore. Inserisci SOLO la query.
    """
    try:
        tavily = TavilySearch(max_results=3, include_raw_content=True)
        search_query = f"{query}"
        results = tavily.invoke({"query": search_query})

        all_documents = []
        sources_found = []

        results_list = results.get("results", []) if isinstance(results, dict) else results

        for res in results_list:
            if isinstance(res, dict):
                raw = res.get("raw_content", "")
                if raw:
                    extracted_text = trafilatura.extract(
                        raw,
                        include_comments=False,
                        include_tables=False,
                        favor_precision=True
                    )
                    if extracted_text:
                        content = extracted_text
                        print(f"   [Scraper] 🟢 Trafilatura usato con successo per: {res.get('url', '')}")
                    else:
                        soup = BeautifulSoup(raw, "html.parser")
                        content = soup.get_text(separator=' ', strip=True)
                        print(f"   [Scraper] 🟡 Fallback su BS4 per: {res.get('url', '')}")
                else:
                    content = res.get("content", "")

                url = res.get("url", "")
                title = res.get("title", "Fonte web")
            else:
                content = getattr(res, 'page_content', str(res))
                url = ""
                title = "Fonte web"

            content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
            content = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', content)
            content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)

            if not content.strip():
                continue

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP
            )
            chunks = splitter.split_text(content)

            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source_url": url,
                        "source_name": title,
                        "source_type": "web",
                        "topic": query,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                    }
                )
                all_documents.append(doc)

            if url:
                sources_found.append(f"{title} ({url})")

        added = rag_manager.add_documents(all_documents)
        sources_str = ", ".join(sources_found) if sources_found else "nessuna fonte trovata"
        return f"Ricerca web completata. {added} chunk salvati da {len(sources_found)} fonti: {sources_str}. Usa 'rag_retrieval_tool' per leggere i dettagli."

    except Exception as e:
        return f"Errore durante la ricerca web: {e}"


@tool
def rag_retrieval_tool(query: str) -> str:
    """
    Esegue una ricerca semantica nei documenti testuali e articoli scaricati in locale (ChromaDB).

    Args:
        query (str): La frase o domanda da cercare nei documenti locali (es. "Come funziona il combat system?").
    """
    return rag_manager.retrieve(query)


@tool
def knowledge_graph_tool(entity: str) -> str:
    """
    Interroga il database interno Knowledge Graph (Neo4j) per vedere se un argomento è già noto al blog.

    Args:
        entity_name (str): Il nome specifico e preciso del soggetto da cercare (es. "Malenia", "Parry", "FromSoftware"). Inserisci SOLO il nome, senza preposizioni o frasi.
    """
    return kg_manager.query(entity)


@tool
def deep_read_article(source_url: str, offset: int | str = 0, limit: int | str = 3) -> str:
    """
    Legge il testo completo di un articolo da un URL trovato tramite la ricerca web.
    IMPORTANTE: Passa SOLO l'URL esatto, non testo generico.

    Args:
        source_url (str): L'URL della pagina web da leggere (es. "https://www.ign.com/articolo").
        offset (int): Da quale paragrafo iniziare a leggere (default: 0).
        limit (int): Quanti paragrafi leggere alla volta (default: 3).
    """
    try:
        offset = int(offset)
        limit = int(limit)
        docs = rag_manager.get_article_chunks(source_url, offset=offset, limit=limit)

        if not docs:
            return f"Nessun chunk trovato per {source_url} (offset={offset}). L'articolo potrebbe essere terminato."

        content_parts = []
        for doc in docs:
            idx = doc.metadata.get('chunk_index', '?')
            total = doc.metadata.get('total_chunks', '?')
            content_parts.append(f"[Chunk {idx}/{total}]\n{doc.page_content}")

        next_offset = offset + limit
        content = "\n\n".join(content_parts)
        return f"Contenuto articolo ({source_url}):\n{content}\n\n[Per leggere oltre, usa offset={next_offset}]"

    except Exception as e:
        return f"Errore lettura articolo: {e}"


@tool
def youtube_transcript_fetcher(video_url: str) -> str:
    """Scarica la trascrizione completa di un video YouTube e la salva nel database vettoriale locale con metadata.

    Args:
        video_url (str): L'URL del video YouTube da cui estrarre la trascrizione (es. "https://www.youtube.com/watch?v=abc123").
    """
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

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(raw_text)

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source_url": video_url,
                    "source_name": f"YouTube Video ({video_id})",
                    "source_type": "youtube",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        added = rag_manager.add_documents(documents)
        word_count = len(raw_text.split())
        return f"Trascrizione YouTube ({word_count} parole) salvata: {added} chunk. Usa 'rag_retrieval_tool' per leggere i dettagli."

    except Exception as e:
        return f"Impossibile estrarre la trascrizione: {e}"
