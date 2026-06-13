from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import RAG_PERSIST_DIRECTORY, RAG_EMBEDDING_MODEL, RAG_RETRIEVE_K

class RAGManager:
    """Gestore del database vettoriale ChromaDB."""

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or RAG_PERSIST_DIRECTORY
        self.embeddings = HuggingFaceEmbeddings(model_name=RAG_EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="gaming_blog"
        )

    def add_documents(self, documents: list[Document]) -> int:
        """Aggiunge documenti deduplicando per source_url."""
        if not documents:
            return 0

        new_urls = {d.metadata.get("source_url") for d in documents if d.metadata.get("source_url")}

        urls_to_skip = set()
        for url in new_urls:
            try:
                existing = self.vectorstore._collection.get(where={"source_url": url}, limit=1)
                if existing and existing["ids"]:
                    urls_to_skip.add(url)
            except Exception:
                pass

        filtered_docs = [d for d in documents if d.metadata.get("source_url") not in urls_to_skip]

        if filtered_docs:
            self.vectorstore.add_documents(filtered_docs)
        return len(filtered_docs)

    def retrieve(self, query: str, k: int = None) -> str:
        """Cerca nel database locale e restituisce risultati formattati con metadata."""
        k = k or RAG_RETRIEVE_K
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            if not docs:
                return "Nessuna informazione rilevante trovata nel database locale."

            formatted = []
            for i, doc in enumerate(docs):
                source_url = doc.metadata.get("source_url", "N/A")
                source_name = doc.metadata.get("source_name", "Fonte sconosciuta")
                formatted.append(
                    f"--- Risultato {i+1} [Fonte: {source_name} | URL: {source_url}] ---\n{doc.page_content}"
                )
            return "\n\n".join(formatted)

        except Exception as e:
            return f"Errore durante la ricerca nel RAG locale: {e}"

    def retrieve_with_sources(self, query: str, k: int = None) -> list[dict]:
        """Restituisce risultati con metadata separati."""
        k = k or RAG_RETRIEVE_K
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [
                {
                    "content": doc.page_content,
                    "source_url": doc.metadata.get("source_url", "N/A"),
                    "source_name": doc.metadata.get("source_name", "N/A"),
                    "chunk_index": doc.metadata.get("chunk_index", -1),
                    "total_chunks": doc.metadata.get("total_chunks", -1),
                }
                for doc in docs
            ]
        except Exception as e:
            print(f"Errore retrieve_with_sources: {e}")
            return []

    def get_article_chunks(self, source_url: str, offset: int = 0, limit: int = 3) -> list[Document]:
        """Recupera chunk specifici di un articolo per URL con offset progressivo."""
        try:
            results = self.vectorstore._collection.get(
                where={
                    "$and": [
                        {"source_url": source_url},
                        {"chunk_index": {"$gte": offset}},
                        {"chunk_index": {"$lt": offset + limit}}
                    ]
                },
                limit=limit
            )
            if results and results["documents"]:
                docs = []
                for i, doc_text in enumerate(results["documents"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    docs.append(Document(page_content=doc_text, metadata=metadata))
                return docs
        except Exception as e:
            print(f"Errore get_article_chunks: {e}")
        return []
