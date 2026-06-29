import os
import csv
import time
import logging
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

class LabelerOutput(BaseModel):
    label: int = Field(
        ...,
        description="1 se articolo/recensione utile, 0 se e-commerce/junk/forum/login."
    )

def build_dataset() -> None:
    logging.info("Avvio procedura di costruzione automatica del Dataset.")

    output_file = "finetuning/source_quality_dataset.csv"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["testo", "label"])
        writer.writeheader()

    try:
        tavily = TavilySearch(max_results=3, include_raw_content=False)
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0) # Il modello potrebbe venire dismesso da Groq, quindi è consigliabile sostituirlo con un modello LLM più recente se necessario.
        labeler_llm = llm.with_structured_output(LabelerOutput)
    except Exception as e:
        logging.error(f"Errore nell'inizializzazione dei client: {e}")
        return

    queries_informative = [
        "Elden Ring recensione completa", "Silent Hill 2 Remake analisi trama",
        "Hades 2 gameplay meccaniche spiegazione", "Cyberpunk 2077 lore del mondo",
        "The Last of Us recensione opinioni", "Bloodborne spiegazione finale",
        "Final Fantasy 15 recensione IGN", "Sekiro combat system come funziona",
        "Zelda Tears of the Kingdom recensione", "Persona 5 Royal analisi personaggi",
        "Hollow Knight esplorazione mappa", "God of War Ragnarok sviluppo personaggi",
        "Red Dead Redemption 2 analisi narrativa", "Dark Souls 3 guida ai boss",
        "Baldur's Gate 3 recensione", "Alan Wake 2 analisi storia",
        "The Witcher 3 spiegazione finali", "Mass Effect 2 recensione trama",
        "Ghost of Tsushima combat system", "Celeste recensione platforming",
        "Resident Evil 4 Remake differenze originale", "Horizon Zero Dawn lore mondo",
        "Outer Wilds recensione meccaniche", "Stardew Valley guida iniziale",
        "Disco Elysium spiegazione finale", "Death Stranding recensione completa",
        "Nier Automata analisi filosofica", "Control recensione narrativa",
        "Returnal spiegazione lore", "Bioshock analisi personaggi",
        "Demon's Souls recensione level design", "Armored Core 6 combat system",
        "Lies of P recensione soulslike", "Starfield spiegazione mondo aperto",
        "Super Mario Odyssey level design", "Metroid Dread recensione esplorazione",
        "Doom Eternal gameplay loop", "Half-Life 2 recensione retrospettiva",
        "Fallout New Vegas analisi GDR", "Skyrim lore draghi spiegazione",
        "GTA 5 recensione satira sociale", "Yakuza 0 recensione narrativa",
        "Metal Gear Solid 3 analisi trama", "Undertale recensione pacifist",
        "Hades recensione roguelike", "Cuphead analisi finale",
        "Spiderman 2 PS5 recensione", "Final Fantasy 7 Rebirth analisi",
        "Dragon's Dogma 2 recensione", "Helldivers 2 gameplay meccaniche"
    ]

    queries_junk = [
        "Acquista Elden Ring PS5 sconti", "Final Fantasy 15 recupero password forum",
        "Cyberpunk 2077 aggiungi al carrello spedizione gratuita", "The Last of Us usato ebay",
        "Bloodborne login iscriviti alla newsletter", "Sekiro accetta cookie policy privacy",
        "Zelda Tears of the Kingdom prezzo gamestop", "Persona 5 Royal facebook accedi per vedere",
        "Hades 2 key steam comprare", "Silent Hill 2 Remake preordine sconti",
        "Hollow Knight acquista nintendo eshop", "God of War Ragnarok bundle ps5 prezzo",
        "Red Dead Redemption 2 shark card", "Dark Souls 3 key pc global",
        "Baldur's Gate 3 instant gaming sconto", "Alan Wake 2 epic games store prezzo",
        "The Witcher 3 goty sconti ps store", "Mass Effect 2 origin login",
        "Ghost of Tsushima director's cut usato", "Celeste nintendo switch sconti",
        "Resident Evil 4 Remake deluxe edition prezzo", "Horizon Zero Dawn pc steam key",
        "Outer Wilds cdkeys sconto", "Stardew Valley mobile prezzo store",
        "Disco Elysium gog login password", "Death Stranding director's cut upgrade prezzo",
        "Nier Automata game of the yorha edition ebay", "Control ultimate edition sconti",
        "Returnal usato gamestop", "Bioshock collection switch prezzo",
        "Demon's Souls ps5 acquista ora", "Armored Core 6 collector's edition preordine",
        "Lies of P game pass xbox login", "Starfield premium edition upgrade",
        "Super Mario Odyssey usato subito.it", "Metroid Dread sconti black friday",
        "Doom Eternal deluxe edition carrello", "Half-Life 2 orange box key",
        "Fallout New Vegas ultimate edition steam prezzo", "Skyrim anniversary edition sconti",
        "GTA 5 shark cards acquisto", "Yakuza 0 key steam global",
        "Metal Gear Solid master collection prezzo", "Undertale steam login",
        "Hades sconti eshop nintendo", "Cuphead fisico amazon",
        "Spiderman 2 PS5 console bundle prezzo", "Final Fantasy 7 Rebirth preordine amazon",
        "Dragon's Dogma 2 deluxe edition sconti", "Helldivers 2 super credits acquista"
    ]

    all_queries_with_targets = [{"query": q, "expected": 1} for q in queries_informative] + \
                               [{"query": q, "expected": 0} for q in queries_junk]

    righe_salvate = 0
    scartati = 0

    prompt_template = """Sei un Data Annotator esperto. Il tuo compito è classificare frammenti di testo estratti dal web per addestrare un filtro di un sistema RAG videoludico.
Devi rispondere SOLO con 1 o 0.

REGOLA PER CLASSE 1 (FONTE UTILE):
Assegna 1 SOLO SE il testo contiene informazioni valide per scrivere una RECENSIONE di un videogioco (es. analisi della trama, recensioni di testate giornalistiche, spiegazione del gameplay, guide, pareri approfonditi).

REGOLA PER CLASSE 0 (SPAZZATURA / JUNK):
Assegna 0 a TUTTO il resto. In particolare, assegna 0 se il testo riguarda:
- E-commerce, acquisti, prezzi, sconti (Amazon, eBay, Steam, Gamestop).
- Pagine di Login, Policy sui Cookie, Errori 404, Newsletter.
- Notizie su aggiornamenti, patch, o upgrade gratuiti.
- Polemiche sui pre-ordini o commenti non informativi.
- Pagine social vuote che richiedono autenticazione.

Testo da classificare:
{test}
"""

    for item in all_queries_with_targets:
        query_text = item["query"]
        expected_label = item["expected"]

        logging.info(f"Esecuzione query: '{query_text}' (Atteso: {expected_label})")
        try:
            results = tavily.invoke({"query": query_text})
            results_list = results.get("results", []) if isinstance(results, dict) else results

            for res in results_list:
                if not isinstance(res, dict):
                    continue

                title = res.get('title', '')
                snippet = res.get('content', '')
                dirty_text = f"{title} | {snippet}".replace('\n', ' ').replace('"', "'").strip()

                if len(dirty_text) < 30:
                    continue

                prompt = prompt_template.format(test=dirty_text)
                response = labeler_llm.invoke(prompt)

                if response.label == expected_label:
                    with open(output_file, "a", encoding="utf-8", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["testo", "label"])
                        writer.writerow({"testo": dirty_text, "label": response.label})

                    righe_salvate += 1
                    logging.info(f"   [ACCETTATO E SALVATO] Label: {response.label}")
                else:
                    scartati += 1
                    logging.warning(f"   [SCARTATO] Mismatch (Atteso {expected_label}, Predetto {response.label})")

            time.sleep(2)

        except Exception as e:
            logging.error(f"Errore durante l'elaborazione della query '{query_text}': {e}")

    logging.info(f"Dataset completato! Righe salvate: {righe_salvate} | Righe scartate: {scartati}")

if __name__ == "__main__":
    build_dataset()
