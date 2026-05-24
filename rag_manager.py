import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RAGManager:
    """
    Gestore del sistema Retrieval-Augmented Generation (RAG).
    Utilizza ChromaDB come vector store persistente locale e
    HuggingFaceEmbeddings per generare i vettori.
    """
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Inizializza il modello di embedding
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = persist_directory

        # Inizializza il vector store persistente
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_texts(self, texts: list[str], metadatas: list[dict] = None):
        try:
            if metadatas:
                self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            else:
                self.vectorstore.add_texts(texts=texts)
        except Exception as e:
            print(f"Errore durante l'aggiunta di testi al RAG: {e}")
        except Exception as e:
            print(f"Errore durante l'aggiunta di testi al RAG: {e}")

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Cerca informazioni nel database locale per query semantica.
        Ritorna una stringa formattata unendo il contenuto dei top k risultati.
        """
        try:
            # Ottiene i documenti più pertinenti alla query
            docs = self.vectorstore.similarity_search(query, k=k)

            if not docs:
                return "Nessuna informazione rilevante trovata nel database locale."

            # Unisce i risultati
            formatted_docs = []
            for i, doc in enumerate(docs):
                formatted_docs.append(f"--- Risultato Locale {i+1} ---\n{doc.page_content}")

            return "\n\n".join(formatted_docs)

        except Exception as e:
            return f"Errore durante la ricerca nel RAG locale: {e}"
