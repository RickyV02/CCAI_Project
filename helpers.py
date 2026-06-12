import json
import re
import ast

def create_react_entry(node: str, thought: str, action: str, observation: str) -> dict:
    """Crea un entry strutturata per la reasoning trace ReAct."""
    return {
        "node": node,
        "thought": thought,
        "action": action,
        "observation": observation
    }


def parse_llm_json(text: str) -> dict:
    """Parsing robusto di JSON da output LLM con pulizia markdown."""
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def format_extraction_for_writer(extraction) -> str:
    """Converte un GameResearchExtraction in testo formattato per il writer."""
    sections = []

    if extraction.lore_and_story_details:
        sections.append(f"## LORE E STORIA\n{extraction.lore_and_story_details}")

    if extraction.gameplay_and_mechanics_deep_dive:
        sections.append(f"\n## GAMEPLAY E MECCANICHE\n{extraction.gameplay_and_mechanics_deep_dive}")

    if extraction.bosses_mentioned:
        sections.append(f"\n## BOSS MENZIONATI\n{', '.join(extraction.bosses_mentioned)}")

    if extraction.difficulty_notes:
        sections.append(f"Difficoltà: {extraction.difficulty_notes}")

    if extraction.graphics_audio_notes:
        sections.append(f"\n## ASPETTI TECNICI E ARTISTICI\n{extraction.graphics_audio_notes}")

    if extraction.release_info:
        sections.append(f"\n## INFO RILASCIO E SVILUPPO\n{extraction.release_info}")

    if extraction.scores_ratings:
        sections.append(f"\n## VOTI CRITICI\n{', '.join(extraction.scores_ratings)}")

    if extraction.fact_check_notes:
        sections.append(f"\n## ⚠️ NOTE FACT-CHECK\n{extraction.fact_check_notes}")

    if extraction.sources:
        sections.append(f"\n## FONTI")
        for s in extraction.sources:
            if s.is_relevant:
                sections.append(f"- [{s.name}]({s.url}) [Credibilità: {s.credibility}] — {s.key_info}")

    return "\n".join(sections)


def truncate_text(text: str, max_chars: int = 500) -> str:
    """Tronca il testo a max_chars caratteri mantenendo parole intere."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(' ', 1)[0]
    return truncated + "..."

def format_blacklist_for_llm(raw) -> str:
    data = json.loads(ast.literal_eval(str(raw))[0]['text'])
    games = {entry.get('Gioco', '').strip() for entry in data if entry.get('Gioco')}
    return "\n".join(f"- {game}" for game in sorted(games))

def format_catalog_for_llm(raw) -> str:
    data = json.loads(ast.literal_eval(str(raw))[0]['text'])
    return "\n".join(
        f"- {e.get('Gioco', '?')} | Anno: {e.get('Anno') or 'N/D'} | "
        f"Generi: {', '.join(e.get('Generi', [])) or 'N/D'} | "
        f"Meccaniche: {', '.join(e.get('Meccaniche', [])) or 'N/D'} | "
        f"Piattaforme: {', '.join(e.get('Piattaforme', [])) or 'N/D'} | "
        f"Simili: {', '.join(e.get('Giochi_Simili', [])) or 'N/D'} | "
        f"Sviluppatore: {', '.join(e.get('Sviluppatore', [])) or 'N/D'} | "
        f"{'✅ LIBERO' if e.get('Numero_Review', 0) == 0 else '⛔ GIÀ RECENSITO'}"
        for e in data
    )

def format_kg_context(raw) -> str:
    """Formatta i risultati della query generica sul gioco per il context dell'agente."""
    try:
        data = json.loads(ast.literal_eval(str(raw))[0]['text'])
        if not data: return "Nessuna informazione nel Knowledge Graph."

        lines = []
        for e in data:
            lines.append(f"🎮 GIOCO: {e.get('Gioco', 'N/D')} (Anno: {e.get('Anno', 'N/D')})")
            lines.append(f"   - Sviluppatore: {', '.join(e.get('Sviluppatore', []))}")
            lines.append(f"   - Generi: {', '.join(e.get('Generi', []))}")
            lines.append(f"   - Piattaforme: {', '.join(e.get('Piattaforme', []))}")
            lines.append(f"   - Boss Noti: {', '.join(e.get('Boss_Noti', []))}")
            lines.append(f"   - Meccaniche: {', '.join(e.get('Meccaniche_Note', []))}")
            lines.append(f"   - Personaggi: {', '.join(e.get('Personaggi', []))}")
            lines.append(f"   - Giochi Simili: {', '.join(e.get('Giochi_Simili', []))}")

            reviews = e.get('Review_Scritte', [])
            rev_str = ", ".join([f"'{r['titolo']}' (Focus: {r['angolo']})" for r in reviews if r.get('titolo')])
            lines.append(f"   - Recensioni nel Blog: {rev_str if rev_str else 'Nessuna'}")
        return "\n".join(lines)
    except Exception:
        return str(raw) # Fallback al testo grezzo in caso di errore

def format_existing_reviews(raw) -> str:
    """Formatta la lista delle recensioni già esistenti per il Planner."""
    try:
        data = json.loads(ast.literal_eval(str(raw))[0]['text'])
        if not data: return "Nessuna recensione esistente per questo titolo."

        lines = []
        for e in data:
            count = e.get('Numero_Review', 0)
            reviews = e.get('Review_Esistenti', [])
            rev_list = "\n".join([f"      * '{r['titolo']}' (Focus: {r['angolo']})" for r in reviews])
            lines.append(f"- GIOCO: {e.get('Gioco', 'N/D')} | Totale articoli scritti: {count}\n   Dettaglio articoli:\n{rev_list}")
        return "\n".join(lines)
    except Exception:
        return str(raw)

def format_similarity_catalog(raw) -> str:
    """Formatta il mega-catalogo delle somiglianze in modo super compatto per risparmiare token."""
    try:
        data = json.loads(ast.literal_eval(str(raw))[0]['text'])
        lines = []
        for e in data:
            gioco = e.get('Gioco', 'N/D')
            generi = ', '.join(e.get('Generi', [])) or 'N/D'
            meccaniche = ', '.join(e.get('Meccaniche', [])) or 'N/D'
            studi = ', '.join(e.get('Sviluppatori', [])) or 'N/D'
            # Tutto su una riga per compattezza massima
            lines.append(f"- {gioco} | Studi: {studi} | Generi: {generi} | Meccaniche: {meccaniche}")
        return "\n".join(lines)
    except Exception:
        return str(raw)

def format_krag_entities(raw) -> str:
    """Formatta le entità estratte per il K-RAG in modo leggibile."""
    try:
        data = json.loads(ast.literal_eval(str(raw))[0]['text'])
        if not data: return "Nessuna entità trovata."

        lines = []
        for e in data:
            gioco = e.get('Gioco', 'N/D')
            boss = ', '.join(e.get('Boss', []))
            meccaniche = ', '.join(e.get('Meccaniche', []))
            simili = ', '.join(e.get('Giochi_Simili', []))
            generi = ', '.join(e.get('Generi', []))
            personaggi = ', '.join(e.get('Personaggi', []))

            lines.append(f"Gioco: {gioco}")
            if boss: lines.append(f"- Boss: {boss}")
            if meccaniche: lines.append(f"- Meccaniche: {meccaniche}")
            if simili: lines.append(f"- Giochi Simili: {simili}")
            if generi: lines.append(f"- Generi: {generi}")
            if personaggi: lines.append(f"- Personaggi: {personaggi}")

        return "\n".join(lines)
    except Exception:
        return str(raw)

def format_recent_posts_for_writer(raw) -> str:
    """Formatta gli ultimi post per darli in pasto al Writer in modo leggibile."""
    try:
        data = json.loads(ast.literal_eval(str(raw))[0]['text'])
        if not data: return "Nessun post recente."

        lines = []
        for e in data:
            titolo = e.get('Titolo', 'N/D')
            gioco = e.get('Gioco', 'N/D')
            angolo = e.get('Angolo_Trattato', 'N/D')
            lines.append(f"- '{titolo}' (Tratta di: {gioco} | Focus: {angolo})")
        return "\n".join(lines)
    except Exception:
        return str(raw)
