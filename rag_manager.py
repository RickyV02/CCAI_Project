from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import numpy as np

from config import RAG_PERSIST_DIRECTORY, RAG_EMBEDDING_MODEL, RAG_RETRIEVE_K

class RAGManager:
    """Gestore del database vettoriale ChromaDB (Hybrid Search + Reranking)."""

    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory or RAG_PERSIST_DIRECTORY
        self.embeddings = HuggingFaceEmbeddings(model_name=RAG_EMBEDDING_MODEL)
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="gaming_blog"
        )

        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

        self.all_texts = []
        self.all_metadatas = []
        self.bm25 = None
        self._init_bm25()

    def _init_bm25(self):
        """Costruisce l'indice BM25 leggendo i dati da ChromaDB."""
        all_db_data = self.vectorstore._collection.get()
        self.all_texts = all_db_data.get("documents", [])
        self.all_metadatas = all_db_data.get("metadatas", [])

        if self.all_texts:
            tokenized_corpus = [doc.lower().split() for doc in self.all_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

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
            self._init_bm25()

        return len(filtered_docs)

    def retrieve(self, query: str, k: int = None) -> str:
        """Ricerca Ibrida (Dense + Keyword) seguita da Reranking con Cross-Encoder."""
        k = k or RAG_RETRIEVE_K
        num_candidates = k * 3  # Recuperiamo più candidati per darli in pasto al Reranker

        try:
            # =========================================================
            # STAGE 1: HYBRID RETRIEVAL (Dense + Lexical)
            # =========================================================

            # 1A: Dense Retrieval (tramite ChromaDB)
            dense_docs = self.vectorstore.similarity_search(query, k=num_candidates)

            # Se il db è vuoto, stoppa qui
            if not dense_docs:
                 return "Nessuna informazione rilevante trovata nel database locale."

            # 1B: Lexical Retrieval (BM25)
            bm25_docs = []
            if self.bm25 and self.all_texts:
                tokenized_query = query.lower().split()
                bm25_scores = self.bm25.get_scores(tokenized_query)

                top_n_idx = np.argsort(bm25_scores)[-num_candidates:][::-1]
                for idx in top_n_idx:
                    if bm25_scores[idx] > 0:
                        doc = Document(page_content=self.all_texts[idx], metadata=self.all_metadatas[idx])
                        bm25_docs.append(doc)

            # Uniamo i risultati di Dense e BM25 rimuovendo i duplicati (usando il testo come chiave)
            combined_docs = {}
            for doc in dense_docs + bm25_docs:
                combined_docs[doc.page_content] = doc

            candidate_list = list(combined_docs.values())

            # =========================================================
            # STAGE 2: RERANKING (Cross-Encoder)
            # =========================================================
            if len(candidate_list) <= 1:
                final_docs = candidate_list
            else:
                pairs = [[query, doc.page_content] for doc in candidate_list]
                scores = self.reranker.predict(pairs)

                scored_docs = list(zip(scores, candidate_list))
                scored_docs.sort(key=lambda x: x[0], reverse=True)

                final_docs = [doc for score, doc in scored_docs[:k]]

            formatted = []
            for i, doc in enumerate(final_docs):
                source_url = doc.metadata.get("source_url", "N/A")
                source_name = doc.metadata.get("source_name", "Fonte sconosciuta")
                formatted.append(
                    f"--- Risultato {i+1} [Fonte: {source_name} | URL: {source_url}] ---\n{doc.page_content}"
                )
            return "\n\n".join(formatted)

        except Exception as e:
            return f"Errore durante la ricerca nel RAG locale: {e}"

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
