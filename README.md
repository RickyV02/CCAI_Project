# 🎮 Agentic AI - Blogger Videoludico Autonomo

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge_Graph-blue)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

Progetto per il corso di **Cognitive Computing and Artificial Intelligence (A.A. 2025/2026)**.  
**Autore:** Riccardo Maria Villaggio

Questo repository implementa un **sistema multi-agente** basato su **LangGraph** per generare, rifinire e salvare recensioni videoludiche in modo semi-autonomo. L'agente usa un LLM su **Groq**, un **Knowledge Graph Neo4j** esposto via **MCP**, e un **Vector Store ChromaDB** per recuperare informazioni, verificare fonti, produrre bozze e aggiornare la memoria del progetto dopo l'approvazione umana.

## 🚀 Cosa Fa

Il flusso reale del progetto è questo:

1. `main.py` avvia una CLI testuale, controlla le variabili d'ambiente e legge il piano corrente dal Knowledge Graph.
2. `agent_graph.py` orchestra il grafo LangGraph con i nodi di planner, researcher, summarizer, writer, quality check, human review e memory updater.
3. `tools.py` fornisce i tool di ricerca web, retrieval RAG, interrogazione del KG, deep read degli articoli e trascrizione YouTube.
4. L'utente può approvare la bozza, chiedere una riscrittura, richiedere più ricerca o cambiare gioco.
5. Dopo l'approvazione, la review viene salvata in **Neo4j** e **ChromaDB**, così il sistema accumula memoria e contesto per le esecuzioni successive.

## 🧠 Funzionalità Principali

- **LangGraph agentico:** il comportamento è diviso in nodi specializzati invece di un singolo prompt monolitico.
- **K-RAG:** il researcher usa il Knowledge Graph per espandere le query e interrogare il RAG con contesto più mirato.
- **Ricerca web + deep reading:** `search_tool` salva chunk web in locale e `deep_read_article` consente di leggere gli articoli a blocchi.
- **Trascrizioni YouTube:** `youtube_transcript_fetcher` estrae sottotitoli e li salva nel vettore locale per il retrieval.
- **Filtro qualità ML locale:** `ml_manager.py` carica un classificatore `bert-base-multilingual-cased` con adapter **LoRA/PEFT** per scartare contenuti non informativi, spam o e-commerce.
- **Human-in-the-loop:** l'esecuzione si interrompe prima del salvataggio finale per raccogliere feedback esplicito dell'utente.
- **Memoria persistente:** le informazioni approvate vengono aggiornate nel KG e nel database vettoriale, così il progetto non riparte da zero a ogni run.

## 🏗️ Architettura e Componenti

La struttura è pensata come una piccola redazione automatica:

1. **Planner**: valuta il topic, usa il KG per evitare duplicati e prepara l'angolo editoriale.
2. **Researcher**: combina query web, retrieval locale, KG, e trascrizioni video per raccogliere materiale.
3. **Summarizer**: pulisce il rumore delle fonti e compone una base utile per la stesura.
4. **Writer**: produce la bozza finale in markdown con tono da blog videoludico.
5. **Quality Check**: verifica coerenza, completezza e rispetto delle regole di output.
6. **Human Review**: gestisce approvazione, rewrite, approfondimento o cambio topic.
7. **Memory Updater**: estrae entità e metadati dalla review finale e aggiorna KG e RAG.

## ⚙️ Installazione e Setup

### Prerequisiti

- Python 3.10+
- Accesso a **Groq** per il modello LLM
- Accesso a **Tavily** per la ricerca web
- Un'istanza **Neo4j AuraDB** o un'istanza Neo4j raggiungibile via URI
- Facoltativo: **LangSmith** per il tracing

### Installazione dipendenze

```bash
git clone https://github.com/RickyV02/CCAI_Project.git
cd CCAI_Project
pip install -r requirements.txt
```

### Configurazione ambiente

Copia `.env.example` in `.env` e compila almeno queste variabili:

```env
GROQ_API_KEY=la_tua_api_key
TAVILY_API_KEY=la_tua_api_key

NEO4J_URI=neo4j+s://tuo_indirizzo.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=la_tua_password
NEO4J_DATABASE=neo4j

LANGSMITH_API_KEY=la_tua_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=blog-copilot
```

Se non usi Neo4j AuraDB, `config.py` ha dei default locali, ma il Knowledge Graph funziona davvero solo se l'istanza è raggiungibile.

### Inizializzare il Knowledge Graph

Prima del primo avvio esegui il seed base del grafo con il file `init_kg.cypher`, così il planner trova già un catalogo iniziale da consultare.

## ▶️ Esecuzione

Avvia l'interfaccia CLI con:

```bash
python main.py
```

All'avvio il programma mostra se esiste un piano attivo nel Knowledge Graph e ti guida nella scelta del gioco o del prossimo articolo. Durante l'esecuzione puoi approvare la bozza, richiedere modifiche, chiedere ulteriori ricerche o cambiare tema.

## 📂 Struttura del Progetto

- `main.py`: entry point della CLI e gestione dell'interazione con l'utente.
- `agent_graph.py`: definizione del grafo LangGraph, dei nodi e del routing.
- `tools.py`: tool di ricerca, retrieval, KG, deep read e YouTube transcript.
- `kg_manager.py`: interfaccia al Knowledge Graph Neo4j via MCP.
- `rag_manager.py`: gestione del retrieval su ChromaDB.
- `ml_manager.py`: caricamento del classificatore LoRA usato per filtrare le fonti.
- `config.py`: configurazione centralizzata delle variabili d'ambiente e dei parametri di RAG.
- `helpers.py`: funzioni di supporto per parsing, formattazione e utility testuali.
- `schemas.py`: modelli Pydantic per output strutturati.
- `state.py`: stato condiviso del grafo
- `init_kg.cypher`: script per creare l'ontologia e i vincoli iniziali del KG.
- `finetuning/`: dataset, notebook, script e pesi del filtro ML.
- `scripts/`: utility di supporto e script di test.
- `chroma_db/`: persistenza locale del database vettoriale.

## 📄 Note Tecniche

- Il progetto usa `langgraph` con checkpoint in memoria per mantenere lo stato della conversazione.
- Il KG viene accesso tramite `neo4j-mcp-server` e adattatori MCP, non con il driver Neo4j diretto.
- Il filtro ML cade in fallback permissivo se il modello non riesce a caricarsi, così l'app rimane eseguibile anche in ambienti incompleti.
- Le dipendenze principali sono elencate in `requirements.txt`.

## 📄 Licenza & Autore

Sviluppato da **Riccardo Maria Villaggio** per l'esame di Cognitive Computing and Artificial Intelligence (Università degli Studi di Catania).
