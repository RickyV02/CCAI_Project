from pydantic import BaseModel, Field


class PlannerIntent(BaseModel):
    """Capisce se l'utente vuole un gioco specifico o un suggerimento."""
    mode: str = Field(..., description="'specific' se l'utente ha indicato un gioco preciso, 'suggest' se vuole un suggerimento")
    game_name: str = Field(default="", description="Nome del gioco se mode='specific', vuoto altrimenti")


class PlannerOutput(BaseModel):
    """Output strutturato del planner con generazione di un calendario editoriale."""
    reasoning_process: str = Field(
        ...,
        description="Pensa ad alta voce: analizza i giochi e le recensioni nel Knowledge Graph, individua i filoni mancanti e ragiona su quale dovrebbe essere la prossima mossa editoriale."
    )
    sequence_of_posts: list[str] = Field(
    default=[],
    description="Lista di 3 prossimi argomenti... (se richiesto)."
    )
    justification: str = Field(
        ...,
        description="Giustificazione strategica dell'ordine della sequenza basata sulle lacune o sui collegamenti del Knowledge Graph."
    )
    suggested_game: str = Field(
        ...,
        description="Il PRIMO gioco della sequence_of_posts. Questo è il gioco che verrà recensito OGGI in questa singola esecuzione."
    )
    review_angle: str = Field(
        ...,
        description="Angolo scelto per la review del suggested_game (es. 'Recensione Completa e Generale', 'Combat system')."
    )
    plan: str = Field(
        ...,
        description="Piano editoriale dettagliato esclusivamente per la stesura dell'articolo di oggi sul suggested_game."
    )


class SourceEvaluation(BaseModel):
    """Valutazione qualità di una singola fonte."""
    url: str = Field(default="", description="URL della fonte")
    name: str = Field(..., description="Nome della testata o fonte (es. 'IGN Italia', 'Everyeye')")
    credibility: str = Field(..., description="'alta' = testata nota gaming, 'media' = blog specializzato, 'bassa' = fonte non riconosciuta")
    key_info: str = Field(..., description="Informazione principale estratta da questa fonte")
    is_relevant: bool = Field(default=True, description="True se le info sono pertinenti al gioco in esame")


class GameResearchExtraction(BaseModel):
    """Estrazione strutturata di TUTTE le informazioni dal materiale di ricerca."""
    lore_and_story_details: str = Field(default="", description="Paragrafo ESTESAMENTE DETTAGLIATO sulla trama, l'ambientazione e la lore trovata nelle fonti (personaggi, setting, ecc...).")
    gameplay_and_mechanics_deep_dive: str = Field(default="", description="Analisi PROFONDA di gameplay, combat system, armi e meccaniche specifiche. Non fare liste, scrivi un testo discorsivo che racchiuda più informazioni possibili sul gioco.")
    graphics_audio_notes: str = Field(default="", description="Paragrafo dettagliato su grafica, art direction, audio e performance tecniche (se presenti o intuibili dalle fonti utilizzate).")
    bosses_mentioned: list[str] = Field(default=[], description="Nomi dei boss citati (se presenti o intuibili dalle fonti utilizzate)")
    difficulty_notes: str = Field(default="", description="Commenti sulla difficoltà generale (se presenti o intuibili dalle fonti utilizzate)")
    release_info: str = Field(default="", description="Data uscita, piattaforme, sviluppatore (se presenti o intuibili dalle fonti utilizzate)")
    scores_ratings: list[str] = Field(default=[], description="Voti/punteggi citati (es. 'IGN 9/10'), se presenti o intuibili dalle fonti utilizzate")
    sources: list[SourceEvaluation] = Field(default=[], description="Valutazione di OGNI fonte usata")
    fact_check_notes: str = Field(default="", description="Segnala eventuali contraddizioni trovate: sia quelle tra i testi web e il KG, SIA quelle interne tra le varie fonti web stesse (es. se una fonte dice una cosa e un'altra ne dice un'altra). Usa questo campo SOLO per segnalare contraddizioni, se tutti concordano e non ci sono contraddizioni, DEVI lasciare questo campo vuoto ('')")


class QualityVerdict(BaseModel):
    """Verdetto del quality check."""
    reasoning_process: str = Field(
        ...,
        description="Pensa ad alta voce: analizza la bozza rispetto ai requisiti prima di emettere il verdetto."
    )
    passed: bool = Field(..., description="True se il post supera il controllo qualità")
    reason: str = Field(..., description="Motivazione del verdetto")
    missing_elements: list[str] = Field(default=[], description="Elementi mancanti se non passa")


class PostEntities(BaseModel):
    """Entità estratte dalla review approvata per aggiornare il KG."""
    main_topic: str = Field(..., description="Nome esatto del gioco principale (es. 'Elden Ring')")
    post_title: str = Field(..., description="Titolo dell'articolo")
    review_angle: str = Field(default="", description="Angolo della review (es. 'combat system', 'narrativa')")
    bosses: list[str] = Field(default=[], description="Nomi dei boss menzionati")
    mechanics: list[str] = Field(default=[], description="Meccaniche di gioco menzionate")
    characters: list[str] = Field(default=[], description="Personaggi DEL VIDEOGIOCO. NON inserire MAI sviluppatori, direttori o persone reali, a meno che non siano personaggi del videogioco. Se una persona reale è menzionata come parte del gioco, inseriscila SOLO se è chiaramente identificata come personaggio del gioco. Altrimenti, non va inserita.")
    claims: list[str] = Field(default=[], description="1-3 affermazioni chiave o opinioni forti")
    sources: list[str] = Field(default=[], description="URL o nomi delle fonti citate")
    similar_games: list[str] = Field(
        default=[],
        description="Lista di ALTRI VIDEOGIOCHI specifici citati come paragoni. 🚨 ATTENZIONE: Inserisci SOLO titoli esatti (es. 'Dark Souls', 'Silent Hill'). NON inserire MAI generi o categorie videoludiche (es. 'Soulslike', 'GDR', 'RPG', 'Action')."
    )
    genres: list[str] = Field(
        default=[], 
        description="Generi videoludici a cui appartiene il gioco. ATTENZIONE: Deduci il genere dagli appunti di ricerca se non è esplicito. (es. 'Horror', 'Metroidvania', 'Action RPG')."
    )
    studios: list[str] = Field(default=[], description="Studi di sviluppo o publisher menzionati (es. 'Capcom', 'FromSoftware').")
    platforms: list[str] = Field(default=[], description="Piattaforme su cui è disponibile il gioco (es. 'PC', 'PlayStation 5').")
    release_year: int | None = Field(default=None, description="Anno di uscita del gioco (es. 2022). Lascia vuoto se non menzionato.")


class FeedbackRouting(BaseModel):
    """Classificazione del feedback umano per decidere il prossimo nodo."""
    decision: str = Field(..., description="'need_research' se mancano info, 'rewrite' se va solo riscritto, 'change_topic' per cambiare gioco, 'approve' se va bene")
    reasoning: str = Field(..., description="Motivazione della scelta")

class PlanApprovalRouting(BaseModel):
    """Classificazione del feedback utente durante la fase di pianificazione."""
    decision: str = Field(..., description="'approve' se l'utente accetta di procedere o conferma la proposta, 'modify' se chiede modifiche al piano o vuole cambiare gioco.")
    new_game: str = Field(default="", description="Se l'utente nel feedback chiede di recensire un gioco DIVERSO (es. 'allora facciamo Zelda'), scrivi qui il nome esatto del nuovo gioco. Altrimenti lascia vuoto.")
    reasoning: str = Field(..., description="Motivazione della classificazione.")
