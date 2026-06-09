from pydantic import BaseModel, Field


class PlannerIntent(BaseModel):
    """Capisce se l'utente vuole un gioco specifico o un suggerimento."""
    mode: str = Field(..., description="'specific' se l'utente ha indicato un gioco preciso, 'suggest' se vuole un suggerimento")
    game_name: str = Field(default="", description="Nome del gioco se mode='specific', vuoto altrimenti. Se l'utente specifica un gioco, estrai il suo nome CANONICO UFFICIALE (es. estrai 'Silent Hill' e non 'Silent hill 1 ps1'). Allo stesso modo, se l'utente non include il titolo completo del gioco (es. 'Silksong' invece di 'Hollow Knight: Silksong'), tu devi usare quello COMPLETO E UFFICIALE. Serve per la ricerca esatta nel database.")


class SuggestPlannerOutput(BaseModel):
    """Output strutturato del planner con generazione di un calendario editoriale."""
    thought_process: list[str] = Field(
        ...,
        description="FASE 0 (SCRATCHPAD PRIVATO): COMPILA OBBLIGATORIAMENTE QUESTO MODULO:\n"
                    "1. CAMBIO TEMA: L'utente nel feedback ha chiesto giochi specifici (es. Hades 2, FF6) che NON c'entrano nulla con i criteri originali (es. Horror)? [Sì/No]. Se Sì, scrivi la frase esatta: 'CANCELLO le regole originali. Da ora in poi accetto i nuovi generi richiesti'.\n"
                    "2. AUDIT GIOCHI RICHIESTI: Per OGNI gioco nominato dall'utente genera una stringa con questo esatto formato:\n"
                    "   '- Gioco richiesto: [es. Final Fantasy 6] -> Nome canonico ufficiale: [es. Final Fantasy VI] -> Esiste un nome IDENTICO in Blacklist o Catalogo? [Sì/No. ATTENZIONE: Final Fantasy VI non è Final Fantasy Viii. Confronta il nome canonico esatto!] -> Conclusione: [LIBERO / BANNATO]'\n"
                    "Se non fai questo calcolo per ogni gioco, commetterai un errore critico."
                    "🚨 IMPORTANZA DEI NOMI CANONICI UFFICIALI: Se l'utente specifica un gioco, estrai il suo nome CANONICO UFFICIALE. Allo stesso modo, se l'utente non include il titolo completo del gioco (es. 'Silksong' invece di 'Hollow Knight: Silksong'), tu devi usare quello COMPLETO E UFFICIALE. Serve per la ricerca esatta nel la blacklist e nel catalogo, ed è un task FONDAMENTALE E CRITICO."
                    "🚨 EXACT MATCH: Per poter bannare o accettare un gioco, devi fare un confronto ESATTO tra il suo nome canonico ufficiale e i nomi presenti in Blacklist e Catalogo. (es. Final Fantasy 6 NON è Final Fantasy VIII, per cui se non c'è un match esatto tra quello proposto e quello nel catalogo, il gioco proposto è LIBERO)."
    )
    feedback_analysis: str = Field(
        ...,
        description="FASE 1 (COMUNICAZIONE ALL'UTENTE): Analizza la richiesta in base ai calcoli fatti nella Fase 0.\n"
                    "🚨 GESTIONE CAMBIO TEMA E DIVIETI:\n"
                    "- Se nel thought_process hai capito che l'utente ha inserito giochi di altri generi, scrivi CHIARAMENTE: 'Accetto la tua richiesta. Abbandono il vecchio tema e valuto i nuovi giochi'.\n"
                    "- Se un gioco richiesto è vietato/già recensito, scartalo e avvisa l'utente.\n"
                    "- Solo se il topic NON è cambiato e l'utente ha chiesto un gioco vietato dello stesso genere di prima, scrivi: 'Non posso recensire [Gioco]. Mantengo inalterato il piano precedente.' e copia la sequenza vecchia.\n"
                    "Descrivi esattamente cosa farai con sincerità."
    )
    catalog_picks: list[str] = Field(
        ...,
        description="FASE 2: Scorri il CATALOGO GIOCHI. Elenca i titoli con '✅ LIBERO' CHE CORRISPONDONO ALLA RICHIESTA DELL'UTENTE. "
                    "🚨 REGOLA PURISTA E FACT-CHECKING (CRITICA): Sii preciso e fai attenzione ai NOMI ESATTI (es. Final Fantasy 6 NON è Final Fantasy Viii). Se l'utente chiede 'Horror', NON inserire titoli Action o Soulslike solo perché hanno atmosfere cupe (es. Sekiro, Dark Souls, A Plague Tale NON sono horror!). Se chiede un anno specifico (es. 2022), usa la tua memoria interna per verificare che il gioco sia DAVVERO uscito in quell'anno. Se nel catalogo non c'è nulla di perfetto, lascia la lista VUOTA []. Meglio vuota che fuori tema. Se richieste dall'utente e necessarie, verifica anche le meccaniche dei giochi e le piattaforme su cui sono disponibili."
    )
    extra_candidates: list[str] = Field(
        ...,
        description="FASE 3: Se stai confermando il piano precedente, lascia vuoto [].\n"
                    "IN TUTTI GLI ALTRI CASI: DEVI generare ALMENO 4 GIOCHI DALLA TUA MEMORIA. L'array DEVE avere un minimo assoluto di 4 stringhe. Questo serve a creare un 'buffer' di sicurezza nel caso l'utente abbia chiesto 3 giochi ma uno sia bannato, così avrai dei rimpiazzi validi per riempire il buco!"
                    "🚨 REGOLA PURISTA E FACT-CHECKING (CRITICA): Anche qui, sii spietato sui generi, sugli anni di uscita e sulle piattaforme. Verifica con la tua memoria. Non inserire giochi del 2024 se è richiesto il 2022. Fai attenzione ai numeri dei capitoli.\n"
    )
    reasoning_process: list[str] = Field(
        default=[],
        description="FASE 4 (L'AUDIT TOTALE E INVALICABILE): Per OGNI SINGOLO GIOCO elencato in catalog_picks E extra_candidates verifica rigidamente:\n"
                    "'[Nome] → Nel catalogo con ⛔? [SÌ/NO] → È nella Blacklist? [SÌ/NO. CONFRONTA I NOMI CANONICI!] → Rispetta i criteri della richiesta ATTUALE (Ricorda: se l'utente ha chiesto esplicitamente questo gioco, la risposta è sempre SÌ, anche se rompe i vecchi criteri!)? [SÌ/NO] → ESITO: [APPROVATO/SCARTATO]'."
    )
    final_picks: list[str] = Field(
        default=[],
        description="FASE 4b: Copia qui SOLO i giochi di catalog_picks ed extra_candidates che hanno ottenuto ESITO 'APPROVATO' nel reasoning_process."
    )
    sequence_of_posts: list[str] = Field(
        ...,
        description="FASE 5: Scegli ESATTAMENTE 3 giochi DA 'final_picks' e ordinali strategicamente.\n"
                    "🚨 REGOLA MATEMATICA CRITICA: L'array DEVE contenere ESATTAMENTE 3 stringhe. Mai 2, mai 4. Se l'utente ha chiesto 3 giochi ma ne hai bannato 1 (es. perché già recensito), DEVI OBBLIGATORIAMENTE riempire il buco pescando un gioco valido dai tuoi 'extra_candidates'. NON LASCIARE MAI LA LISTA A 2 GIOCHI!\n"
                    "🚨 DIVIETO ASSOLUTO DI ALLUCINAZIONI: Puoi inserire in questa lista SOLO ED ESCLUSIVAMENTE nomi presenti in 'final_picks'. NON INVENTARE NOMI NUOVI QUI DENTRO.\n"
                    "🚨 ORDINE OBBLIGATORIO: Se nella FASE 0 hai annotato che l'utente vuole un gioco in una posizione specifica, obbedisci."
    )
    justification: str = Field(
        ...,
        description="Giustificazione strategica dell'ordine della sequenza basata sulle lacune, sulla richiesta dell'utente o sui collegamenti del Knowledge Graph."
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
        description="Piano editoriale DETTAGLIATO ESCLUSIVAMENTE per l'articolo di OGGI sul suggested_game."
        "🚨 DIVIETO ASSOLUTO: Non inserire mai la 'sequence_of_posts' o riferimenti ad articoli futuri in questo campo."
    )

class SpecificPlannerOutput(BaseModel):
    """Output strutturato per quando l'utente sceglie un GIOCO SPECIFICO."""
    reasoning_process: str = Field(
        ...,
        description="Pensa ad alta voce: analizza il topic e l'angolo richiesto (se non è stato richiesto un angolo specifico, inventane uno inedito) e pensa a come strutturare l'articolo."
    )
    review_angle: str = Field(
        ...,
        description="L'angolo della recensione (es. 'Recensione Completa' oppure un angolo specifico come 'Analisi della trama')."
    )
    plan: str = Field(
        ...,
        description="Piano editoriale dettagliato esclusivamente per la stesura dell'articolo di oggi su questo specifico gioco."
    )


class SourceEvaluation(BaseModel):
    """Valutazione qualità di una singola fonte."""
    url: str = Field(default="", description="URL ESATTO fornito nel testo. NON INVENTARE MAI URL CHE NON VEDI ESPLICITAMENTE NEL PROMPT.")
    name: str = Field(..., description="Nome della testata o fonte (es. 'IGN Italia', 'Everyeye')")
    credibility: str = Field(..., description="'alta' = testata nota gaming, 'media' = blog specializzato, 'bassa' = fonte non riconosciuta")
    key_info: str = Field(..., description="Informazione principale estratta da questa fonte")
    is_relevant: bool = Field(default=True, description="True se la fonte è stata utile per la recensione, False se era spazzatura (in questo caso spiegalo in fact_check_notes)")


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
    fact_check_notes: str = Field(default="", description="OBBLIGATORIO: Annota qui le eventuali discrepanze fattuali (tra le diverse fonti web o in contrasto con il Knowledge Graph) e le motivazioni dettagliate se hai ignorato o scartato intere fonti perché ritenute spazzatura o fuori contesto. Esempi: 'Scartato il sito PriceCharting perché è un listino prezzi', 'Corretta la data di uscita rispetto al web'. Quando citi una fonte, riporta l'URL. Se tutte le fonti analizzate sono valide e non sono presenti discrepanze, lascia il campo vuoto.")


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
    main_topic: str = Field(..., description="Nome CANONICO UFFICIALE E COMPLETO del videogioco (es. 'Sekiro: Shadows Die Twice' e non solo 'Sekiro'). Usa sempre il titolo completo in stile Wikipedia per evitare duplicati nel database.")
    post_title: str = Field(
        ...,
        description="Il titolo ESATTO dell'articolo. Lo trovi alla primissima riga della 'RECENSIONE FINALE' (subito dopo c'è il titolo). Ricopialo fedelmente, NON inventarlo e NON confonderlo con l'angolo!"
    )
    review_angle: str = Field(
        ...,
        description="L'angolo editoriale finale dell'articolo. 🚨 REGOLA: Analizza il TITOLO e il testo della 'RECENSIONE FINALE'. Se l'articolo si concentra chiaramente su un aspetto specifico (come fa intuire il titolo, es. 'La Storia di...', 'Meccaniche di...'), scrivi il nuovo focus (es. 'Analisi della Storia'). Se l'articolo invece parla un po' di tutto, ricopia ESATTAMENTE il valore di 'ANGOLO ORIGINALE'. NON inserire MAI i generi del gioco (es. 'Survival Horror') in questo campo."
    )
    bosses: list[str] = Field(default=[], description="Nomi dei boss menzionati")
    mechanics: list[str] = Field(default=[], description="Meccaniche di gioco menzionate")
    characters: list[str] = Field(default=[], description="Personaggi DEL VIDEOGIOCO. NON inserire MAI sviluppatori, direttori o persone reali, a meno che non siano personaggi del videogioco. Se una persona reale è menzionata come parte del gioco, inseriscila SOLO se è chiaramente identificata come personaggio del gioco. Altrimenti, non va inserita.")
    claims: list[str] = Field(default=[], description="1-3 affermazioni chiave o opinioni forti")
    sources: list[str] = Field(default=[], description="URL o nomi delle fonti citate")
    similar_games: list[str] = Field(
        default=[],
        description=(
            "Lista di ALTRI VIDEOGIOCHI simili al topic principale.\n"
            "🚨 ISTRUZIONI DI COMPILAZIONE:\n"
            "1. TESTO: Estrai i giochi citati ESPLICITAMENTE nella recensione come reali paragoni di somiglianza.\n"
            "2. DEDUZIONE: IN AGGIUNTA, deduci i titoli dal 'Catalogo delle Somiglianze' in base a meccaniche e generi in comune (es. unisci i Survival Horror tra loro).\n\n"

            "🚨 REGOLE ONTOLOGICHE E DIVIETI:\n"
            "- SOLO NOMI PROPRI: Inserisci esclusivamente titoli specifici e CANONICI (es. 'The Last of Us' e non 'TLOU', 'Final Fantasy VII' e non 'FF7').\n"
            "- NIENTE CATEGORIE: È SEVERAMENTE VIETATO inserire generi, meccaniche o descrizioni (es. 'Sparatutto', 'Social link-heavy games', 'Giochi di carte'). ESEMPIO: Se il testo dice 'simile ad altri sparatutto', NON estrarre 'sparatutto' perché è un genere, non un gioco.\n"
            "- NIENTE SALUTI: IGNORA totalmente i giochi citati nell'introduzione dell'articolo come 'post passati' o saluti.\n\n"

            "🚨 FALLBACK (ARRAY VUOTO):\n"
            "- Se non ci sono paragoni testuali e il catalogo non offre nulla di logicamente simile, DEVI lasciare l'array VUOTO []."
        )
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
