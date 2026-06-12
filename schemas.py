from pydantic import BaseModel, Field


class PlannerIntent(BaseModel):
    """Capisce se l'utente vuole un gioco specifico o un suggerimento."""
    mode: str = Field(..., description="'specific' se l'utente ha indicato un gioco preciso o ha confermato il prossimo gioco del piano attivo, 'suggest' se vuole un suggerimento/un nuovo piano editoriale.")
    game_name: str = Field(default="", description="Nome del gioco se mode='specific', vuoto altrimenti. Se l'utente specifica un gioco, estrai il suo nome CANONICO UFFICIALE (es. estrai 'Silent Hill' e non 'Silent hill 1 ps1'). Allo stesso modo, se l'utente non include il titolo completo del gioco (es. 'Silksong' invece di 'Hollow Knight: Silksong'), tu devi usare quello COMPLETO E UFFICIALE. Serve per la ricerca esatta nel database.")


class SuggestPlannerOutput(BaseModel):
    """Output strutturato del planner con generazione di un calendario editoriale."""
    thought_process: list[str] = Field(
        ...,
        description="FASE 0 (SCRATCHPAD PRIVATO): COMPILA OBBLIGATORIAMENTE QUESTO ELENCO PUNTATO SECONDO LE REGOLE SOTTO:\n"
                    "1. CHECK TITOLI SPECIFICI: Rispondi esplicitamente: 'L'utente ha chiesto un gioco specifico? [Sì/No]'. (ATTENZIONE: 'giochi horror del 2022' o 'sparatutto' sono categorie generiche di richieste, quindi la risposta è No).\n"
                    "2. AUDIT GIOCHI RICHIESTI: Se hai risposto 'No' al punto 1, scrivi SOLO 'Nessun gioco specifico nominato' e NON fare nessun audit."
                    " SE E SOLO SE hai risposto 'Sì' al punto 1, valuta OGNI gioco richiesto dall'utente con questo esatto formato:\n"
                    "   '- Gioco richiesto: [es. Final Fantasy 6] -> Nome canonico ufficiale: [es. Final Fantasy VI] -> ESISTE UN NOME IDENTICO in Blacklist? [Sì/No. ATTENZIONE: Confronta il nome canonico esatto! Ad esempio, Final Fantasy VI non è Final Fantasy Viii.] -> Presenza nel Catalogo? [No / Sì con ✅ LIBERO / Sì con ⛔ GIÀ RECENSITO. ATTENZIONE: Confronta il nome canonico esatto! Ad esempio, Final Fantasy VI non è Final Fantasy Viii.] -> Conclusione: [BANNATO se è in Blacklist o se è nel Catalogo con ⛔ GIÀ RECENSITO/ LIBERO (dal Catalogo) se ha ✅ LIBERO nel Catalogo / LIBERO (da Memoria) se è del tutto assente sia dalla Blacklist che dal Catalogo]'\n"
                    "Se non fai questo calcolo per ogni gioco, commetterai un errore critico."
                    "🚨 Fai l'audit SOLO se l'utente ha nominato TITOLI PRECISI (es. 'Madison', 'Hades 2', 'Silent Hill'). Se ha chiesto una categoria generica (es. 'giochi horror del 2022'), scrivi SOLO la frase: 'Nessun gioco specifico nominato'."
                    "🚨 IMPORTANZA DEI NOMI CANONICI UFFICIALI: Se l'utente specifica un gioco, estrai il suo nome CANONICO UFFICIALE. Allo stesso modo, se l'utente non include il titolo completo del gioco (es. 'Silksong' invece di 'Hollow Knight: Silksong'), tu devi usare quello COMPLETO E UFFICIALE. Serve per la ricerca esatta nel la blacklist e nel catalogo, ed è un task FONDAMENTALE E CRITICO."
                    "🚨 EXACT MATCH: Per poter bannare o accettare un gioco, devi fare un confronto ESATTO tra il suo nome canonico ufficiale e i nomi presenti in Blacklist e Catalogo. (es. Final Fantasy 6 NON è Final Fantasy VIII, per cui se non c'è un match esatto tra quello proposto e quello nel catalogo, il gioco proposto è LIBERO)."
    )
    feedback_analysis: str = Field(
        ...,
        description="FASE 1 (COMUNICAZIONE ALL'UTENTE): Analizza la richiesta basandoti ESCLUSIVAMENTE sui calcoli fatti nella Fase 0.\n"
                    "🚨 GESTIONE CAMBIO TEMA:\n"
                    "- Se nel thought_process hai rilevato un cambio tema TRAMITE FEEDBACK, scrivi: 'Accetto la tua richiesta. Abbandono il vecchio tema e valuto i nuovi giochi'.\n"
                    "- Viceversa, se è la prima richiesta o il tema non cambia, scrivi: 'Accetto la richiesta e creo il piano basato su [tuoi criteri]'.\n"
                    "- Solo se il topic NON è cambiato e l'utente ha chiesto un gioco vietato dello stesso genere di prima, scrivi: 'Non posso recensire [Gioco]. Mantengo inalterato il piano precedente.' e copia la sequenza vecchia.\n"
                    "🚨 REGOLA ANTI-ALLUCINAZIONE SUI DIVIETI: Avvisa l'utente che un gioco è stato scartato SOLO E UNICAMENTE se nella Fase 0 hai esplicitamente scritto 'Conclusione: BANNATO' per quel titolo specifico. È SEVERAMENTE VIETATO menzionare, inventare o scartare giochi che l'utente non ha mai richiesto, o giochi usati solo come esempi nelle istruzioni. Parla solo ed esclusivamente dei giochi valutati nella Fase 0!"
    )
    catalog_picks: list[str] = Field(
        ...,
        description="FASE 2 (COPIA-INCOLLA DAL CATALOGO): Scorri il '🎮 CATALOGO GIOCHI' fornito nel prompt. Estrai i titoli SOLO se rispettano TUTTE queste condizioni:\n"
                    "1. Sono FISICAMENTE SCRITTI nel '🎮 CATALOGO GIOCHI' del prompt\n"
                    "2. Hanno l'etichetta '✅ LIBERO'.\n"
                    "3. Corrispondono alla richiesta dell'utente (l'utente può chiedere giochi sia in termini generali, es. 'horror 2022', sia specifici es. 'Hades').\n Inserisci qui i giochi richiesti dall'utente nella fase di feedback SOLO SE sono fisicamente presenti nell'elenco del catalogo, e soprattutto se hanno '✅ LIBERO'\n."
                    "🚨 DIVIETO ASSOLUTO DI ALLUCINAZIONE: Se l'utente ti ha chiesto di inserire un gioco specifico (es. 'Madison', 'Hades 2') ma tu NON lo vedi scritto letteralmente nell'elenco del Catalogo con '✅ LIBERO', È SEVERAMENTE VIETATO inserirlo qui. Inserisci i titoli in questo array SOLO ED ESCLUSIVAMENTE se li vedi SCRITTI TESTUALMENTE in quell'elenco con il flag '✅ LIBERO' e se sono pertinenti. I giochi richiesti ma assenti dal catalogo vanno inseriti SOLO negli 'extra_candidates'."
                    "🚨 REGOLA INFLESSIBILE: Inserisci qui i giochi richiesti dall'utente nella fase di feedback, presenti nel campo 'thought_process' SOLO SE hai risposto alla domanda 'Presenza nel Catalogo?' con 'Sì con ✅ LIBERO'."
                    "🚨 REGOLA INFLESSIBILE SUI GIOCHI SPECIFICI: Se l'utente ti ha chiesto di inserire un gioco specifico (es. 'Madison'), puoi inserirlo qui SOLO E UNICAMENTE se nel 'thought_process' ha ottenuto la dicitura 'LIBERO (dal Catalogo)'. Se ha ottenuto 'LIBERO (da Memoria)', È SEVERAMENTE VIETATO inserirlo qui e andrà negli extra_candidates.\n"
                    "🚨 REGOLA PURISTA E FACT-CHECKING (CRITICA): Sii preciso e fai attenzione ai NOMI ESATTI (es. Final Fantasy 6 NON è Final Fantasy Viii). Se l'utente chiede 'Horror', NON inserire titoli Action o Soulslike solo perché hanno atmosfere cupe (es. Sekiro, Dark Souls, A Plague Tale NON sono horror!). Se chiede un anno specifico (es. 2022), usa la tua memoria interna per verificare che il gioco sia DAVVERO uscito in quell'anno. Se nel catalogo non c'è nulla di perfetto, lascia la lista VUOTA []. Meglio vuota che fuori tema. Se richieste dall'utente e necessarie, verifica anche le meccaniche dei giochi e le piattaforme su cui sono disponibili."
    )
    extra_candidates: list[str] = Field(
        ...,
        description="FASE 3 (IL MAGAZZINO VINCOLATO - OBBLIGO DI 5 GIOCHI): Se stai confermando il piano precedente, lascia vuoto [].\n"
                    "IN TUTTI GLI ALTRI CASI: Questo array DEVE contenere SEMPRE ESATTAMENTE 5 GIOCHI DIVERSI Nessuna eccezione e nessuna scusa. Componi l'array seguendo questo rigoroso ordine:\n"
                    "1. 🚨 VINCOLO SUPREMO SULLE RICHIESTE UTENTE: Inserisci obbligatoriamente TUTTI che nel 'thought_process' siano risultati 'LIBERO (da Memoria)' e che tu non li abbia già messi in 'catalog_picks'. È vietato omettere un gioco libero richiesto!\n"
                    "2. 🚨 RIEMPIMENTO BUFFER: Dopo aver inserito i giochi richiesti, AGGIUNGI altri titoli pertinenti al nuovo tema pescati dalla tua memoria interna.\n"
                    "2.5. 🚨 CASO CATEGORIA GENERICA: Se l'utente non ha chiesto titoli specifici ma solo un genere/anno (es. 'horror 2022'), DEVI GENERARE 5 GIOCHI pertinenti attingendo alla tua memoria interna.\n"
                    "3. 🚨 REGOLA MATEMATICA INVALICABILE: Anche se hai già trovato 3 o più giochi validi nel 'catalog_picks', SEI OBBLIGATO a generare ESATTAMENTE 5 stringhe in questo array. Conta fino a 5 prima di chiudere l'array! Devi raggiungere TASSATIVAMENTE un totale di esattamente 5 elementi in questo campo !!!\n"
                    "🚨 DIVIETO DI DOPPIONI: È severamente vietato inserire qui dentro i giochi che hai già inserito in 'catalog_picks'!\n"
                    "🚨 ATTENZIONE ALLA MATEMATICA, REGOLA INFLESSIBILE: Questo array DEVE contenere SEMPRE ESATTAMENTE 5 GIOCHI. Mai 3, mai 4. È il tuo magazzino di riserva.\n"
                    "🚨 ERRORE CRITICO: Anche se ti servono solo 3 giochi per il piano finale, il magazzino DEVE fornirne 5."
    )
    reasoning_process: list[str] = Field(
        ...,
        description="FASE 4 (L'AUDIT SEQUENZIALE TOTALE): 🚨 ORDINE E QUANTITÀ OBBLIGATORI: DEVI valutare TUTTI i giochi presenti in 'catalog_picks' e in 'extra_candidates' ESATTAMENTE NELLO STESSO ORDINE E NELLA LORO TOTALITÀ in cui li hai scritti nei rispettivi array. Non saltare nessun gioco e non sceglierli a caso!\n"
                    "🚨 REGOLA MATEMATICA (ESEMPIO): Se la somma dei giochi in catalog_picks + extra_candidates è 5, DEVI generare TASSATIVAMENTE 5 stringhe in questo array. Se la loro somma è diversa da 5, ad esempio 7, ovviamente vale la stessa regola, devi valutarli TUTTI e 7 (e cosi via in generale per qualsiasi altro numero totale di elementi). Fermarsi prima di averli controllati e valutati tutti singolarmente è un errore critico.\n"
                    "🚨 LOGICA DI APPROVAZIONE (CRITICA): L'esito finale NON è a tua discrezione, ma segue un'equazione logica rigida. Se per un gioco la risposta è 'SÌ' alla presenza nel catalogo con ⛔, OPPURE è 'SÌ' alla presenza in Blacklist, OPPURE è 'NO' al rispetto dei criteri, L'ESITO DEVE ESSERE TASSATIVAMENTE 'SCARTATO'. L'esito è 'APPROVATO' SE E SOLO SE il gioco ottiene contemporaneamente questi tre valori esatti: (⛔? NO), (Blacklist? NO), (Criteri? SÌ).\n"
                    "Formato obbligatorio per ogni riga:\n"
                    "'[Nome esatto] → Nel catalogo con ⛔? [SÌ/NO] → È nella Blacklist? [SÌ/NO. CONFRONTA I NOMI CANONICI!] → Rispetta i criteri della richiesta ATTUALE (Ricorda: se l'utente ha chiesto esplicitamente questo gioco, la risposta è sempre SÌ, anche se rompe i vecchi criteri!)? [SÌ/NO] → ESITO: [APPROVATO/SCARTATO]'."
    )
    final_picks: list[str] = Field(
        default=[],
        description="FASE 4b (LA FILTRAZIONE): Copia qui SOLO i giochi che hanno ottenuto ESITO 'APPROVATO' nel reasoning_process, MANTENENDO RIGOROSAMENTE L'ORDINE ORIGINALE. I giochi che l'utente ha esplicitamente richiesto devono assolutamente rimanere nelle primissime posizioni di questa lista!"
                    "🚨 ATTENZIONE, DEVI RICOPIARE TUTTI I GIOCHI CHE HANNO OTTENUTO ESITO 'APPROVATO' nel reasoning_process, NON UN LORO SOTTOINSIEME !!!"
    )
    sequence_of_posts: list[str] = Field(
        ...,
        description="FASE 5 (LA SEQUENZA): Estrai ESATTAMENTE i PRIMI 3 giochi presenti in 'final_picks'.\n"
                    "🚨 REGOLA DELLA PRIORITÀ: Se l'utente ha chiesto giochi specifici (es. Madison, Hades 2, FF6) e questi sono stati approvati, DEVONO far parte dei 3 giochi scelti qui dentro. Non sostituirli con altri giochi (es. Bloodborne) se i giochi richiesti sono validi e liberi!\n"
                    "🚨 REGOLA MATEMATICA CRITICA: L'array DEVE contenere ESATTAMENTE 3 stringhe. Mai 2, mai 4."
                    "🚨 DIVIETO ASSOLUTO DI ALLUCINAZIONI: Puoi inserire in questa lista SOLO ED ESCLUSIVAMENTE nomi presenti in 'final_picks'. NON INVENTARE NOMI NUOVI QUI DENTRO.\n"
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
        description="Piano editoriale per l'articolo di OGGI.\n"
                    "🚨 REGOLE DI STESURA (UNA SOLA FRASE): Usa questo esatto template e poi fermati immediatamente.\n"
                    "Template obbligatorio: 'Oggi recensiremo [Nome del suggested_game], concentrandoci su [review_angle].'\n"
                    "NON AGGIUNGERE NESSUN'ALTRA PAROLA. Qualsiasi riferimento a giochi successivi provocherà un errore di sistema."
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
        description="Piano editoriale dettagliato esclusivamente per la stesura dell'articolo di oggi su questo specifico gioco (focus sul [review_angle])."
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
