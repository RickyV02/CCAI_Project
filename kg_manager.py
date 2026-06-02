import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import concurrent.futures
import traceback

class KGManager:
    """Gestore del Knowledge Graph Neo4j via MCP"""

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "neo4j")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.server_params = StdioServerParameters(
            command="python",
            args=["-m", "neo4j_mcp_server"],
            env={
                "NEO4J_URI": self.uri,
                "NEO4J_USERNAME": self.username,
                "NEO4J_PASSWORD": self.password,
                "NEO4J_DATABASE": self.database,
                "PATH": os.environ.get("PATH", "")
            }
        )

    def _run_async(self, coro):
        """Esegue una coroutine in modo sicuro anche se un event loop è già attivo."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)
        except Exception as e:
            return f"Errore async: {e}"

    async def _execute_cypher_async(self, cypher: str, tool_name: str = "read-cypher") -> str:
        """Metodo interno generico per eseguire query Cypher via MCP."""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    cypher_tool = next((t for t in tools if t.name == tool_name), None)
                    if not cypher_tool:
                        return f"Tool '{tool_name}' non trovato nel server MCP."
                    result = await cypher_tool.ainvoke({"query": cypher})
                    return str(result) if result else "Nessun risultato."
        except Exception as e:
            print(f"\n[DEBUG MCP SERVER CRASH] Tipo errore: {type(e).__name__}")
            print(f"[DEBUG MCP DETTAGLIO]: {str(e)}")
            traceback.print_exc()
            return f"KG non disponibile: {e}"

    def query(self, entity: str) -> str:
        """Interroga il KG per un'entità specifica."""
        safe_entity = entity.replace("'", "\\'")
        cypher = f"""
        MATCH (g:Game)
        WHERE toLower(g.name) CONTAINS toLower('{safe_entity}')
        OPTIONAL MATCH (g)-[:COVERED_IN]->(b:BlogPost)
        OPTIONAL MATCH (g)-[:HAS_BOSS]->(boss:Boss)
        OPTIONAL MATCH (g)-[:USES_MECHANIC]->(mech:Mechanic)
        OPTIONAL MATCH (g)-[:DEVELOPED_BY]->(studio:Studio)
        OPTIONAL MATCH (g)-[:PART_OF_GENRE]->(genre:Genre)
        OPTIONAL MATCH (g)-[:AVAILABLE_ON]->(plat:Platform)
        OPTIONAL MATCH (g)-[:HAS_CHARACTER]->(char:Character)
        OPTIONAL MATCH (g)-[:SIMILAR_TO]->(sim:Game)
        RETURN g.name AS Gioco,
               g.release_year AS Anno,
               collect(DISTINCT studio.name) AS Sviluppatore,
               collect(DISTINCT genre.name) AS Generi,
               collect(DISTINCT plat.name) AS Piattaforme,
               collect(DISTINCT {{titolo: b.title, angolo: coalesce(b.angle, 'Generico')}}) AS Review_Scritte,
               collect(DISTINCT boss.name) AS Boss_Noti,
               collect(DISTINCT mech.name) AS Meccaniche_Note,
               collect(DISTINCT char.name) AS Personaggi,
               collect(DISTINCT sim.name) AS Giochi_Simili
        LIMIT 5
        """
        result = self._run_async(self._execute_cypher_async(cypher, "read-cypher"))
        if not result or "Nessun risultato" in str(result):
            return f"Nessuna informazione nel KG per: '{entity}'"
        return f"Risultati KG per '{entity}':\n{result}"

    def check_existing_reviews(self, topic: str) -> str:
        """Controlla se esistono già review per un gioco specifico."""
        safe_topic = topic.replace("'", "\\'")
        cypher = f"""
        MATCH (g:Game)-[:COVERED_IN]->(b:BlogPost)
        WHERE toLower(g.name) CONTAINS toLower('{safe_topic}')
        RETURN g.name AS Gioco,
               collect(DISTINCT {{titolo: b.title, angolo: coalesce(b.angle, 'Nessun angolo specificato')}}) AS Review_Esistenti,
               count(b) AS Numero_Review
        """
        return self._run_async(self._execute_cypher_async(cypher, "read-cypher"))

    def query_all_games(self) -> str:
        """Restituisce tutti i giochi nel KG con il numero di review per ciascuno."""
        cypher = """
        MATCH (g:Game)
        OPTIONAL MATCH (g)-[:COVERED_IN]->(b:BlogPost)
        OPTIONAL MATCH (g)-[:PART_OF_GENRE]->(genre:Genre)
        OPTIONAL MATCH (g)-[:DEVELOPED_BY]->(studio:Studio)
        RETURN g.name AS Gioco,
               g.release_year AS Anno,
               collect(DISTINCT genre.name) AS Generi,
               collect(DISTINCT studio.name) AS Sviluppatore,
               count(DISTINCT b) AS Numero_Review,
               collect(DISTINCT {titolo: b.title, angolo: coalesce(b.angle, 'Generico')}) AS Review_Scritte
        ORDER BY Numero_Review ASC
        """
        return self._run_async(self._execute_cypher_async(cypher, "read-cypher"))

    def get_entities_for_krag(self, topic: str) -> str:
        """Estrae entità strutturate dal KG per espandere le query RAG."""
        safe_topic = topic.replace("'", "\\'")
        cypher = f"""
        MATCH (g:Game)
        WHERE toLower(g.name) CONTAINS toLower('{safe_topic}')
        OPTIONAL MATCH (g)-[:HAS_BOSS]->(boss:Boss)
        OPTIONAL MATCH (g)-[:USES_MECHANIC]->(mech:Mechanic)
        OPTIONAL MATCH (g)-[:SIMILAR_TO]->(sim:Game)
        OPTIONAL MATCH (g)-[:PART_OF_GENRE]->(genre:Genre)
        OPTIONAL MATCH (g)-[:HAS_CHARACTER]->(char:Character)
        RETURN g.name AS Gioco,
               collect(DISTINCT boss.name) AS Boss,
               collect(DISTINCT mech.name) AS Meccaniche,
               collect(DISTINCT sim.name) AS Giochi_Simili,
               collect(DISTINCT genre.name) AS Generi,
               collect(DISTINCT char.name) AS Personaggi
        """
        return self._run_async(self._execute_cypher_async(cypher, "read-cypher"))
    
    def get_recent_posts(self, limit: int = 3) -> str:
        """Restituisce gli ultimi N articoli pubblicati sul blog."""
        cypher = f"""
        MATCH (b:BlogPost)<-[:COVERED_IN]-(g:Game)
        WHERE b.created_at IS NOT NULL
        RETURN b.title AS Titolo, g.name AS Gioco, b.angle AS Angolo_Trattato
        ORDER BY b.created_at DESC
        LIMIT {limit}
        """
        return self._run_async(self._execute_cypher_async(cypher, "read-cypher"))

    def update(self, post_title: str, topic: str, review_angle: str = "",
               bosses: list = None, mechanics: list = None,
               characters: list = None, claims: list = None,
               sources: list = None, similar_games: list = None,
               genres: list = None,
               studios: list = None,
               platforms: list = None,
               release_year: int | None = None) -> bool:
        """Aggiorna il KG con le entità estratte dalla review approvata."""
        
        topic = topic.title()
        similar_games = [g.title() for g in (similar_games or [])]
        genres = [g.title() for g in (genres or [])]
        studios = [s.title() for s in (studios or [])]
        
        safe_topic = topic.replace("'", "\\'")
        safe_title = post_title.replace("'", "\\'")
        safe_angle = review_angle.replace("'", "\\'")

        cypher_lines = [
            f"MERGE (g:Game {{name: '{safe_topic}'}})",
            f"MERGE (b:BlogPost {{title: '{safe_title}'}})",
            f"ON CREATE SET b.type = 'review', b.angle = '{safe_angle}', b.created_at = datetime()",
            f"MERGE (g)-[:COVERED_IN]->(b)"
        ]

        if release_year:
            cypher_lines.append(f"SET g.release_year = {release_year}")

        if bosses:
            b_list = "[" + ", ".join([f"'{b.replace(chr(39), chr(92)+chr(39))}'" for b in bosses]) + "]"
            cypher_lines.append(f"FOREACH (x IN {b_list} | MERGE (boss:Boss {{name: x}}) MERGE (g)-[:HAS_BOSS]->(boss))")

        if mechanics:
            m_list = "[" + ", ".join([f"'{m.replace(chr(39), chr(92)+chr(39))}'" for m in mechanics]) + "]"
            cypher_lines.append(f"FOREACH (x IN {m_list} | MERGE (mech:Mechanic {{name: x}}) MERGE (g)-[:USES_MECHANIC]->(mech))")

        if claims:
            c_list = "[" + ", ".join([f"'{c.replace(chr(39), chr(92)+chr(39))}'" for c in claims]) + "]"
            cypher_lines.append(f"FOREACH (x IN {c_list} | MERGE (claim:Claim {{text: x}}) MERGE (b)-[:CLAIMS]->(claim))")

        if sources:
            s_list = "[" + ", ".join([f"'{s.replace(chr(39), chr(92)+chr(39))}'" for s in sources]) + "]"
            cypher_lines.append(f"FOREACH (x IN {s_list} | MERGE (src:Source {{url: x}}) MERGE (b)-[:USED_SOURCE]->(src))")

        if similar_games:
            sim_list = "[" + ", ".join([f"'{sg.replace(chr(39), chr(92)+chr(39))}'" for sg in similar_games]) + "]"
            cypher_lines.append(f"FOREACH (x IN {sim_list} | MERGE (sim:Game {{name: x}}) MERGE (g)-[:SIMILAR_TO]->(sim))")

        if genres:
            g_list = "[" + ", ".join([f"'{g.replace(chr(39), chr(92)+chr(39))}'" for g in genres]) + "]"
            cypher_lines.append(f"FOREACH (x IN {g_list} | MERGE (gen:Genre {{name: x}}) MERGE (g)-[:PART_OF_GENRE]->(gen))")

        if studios:
            st_list = "[" + ", ".join([f"'{s.replace(chr(39), chr(92)+chr(39))}'" for s in studios]) + "]"
            cypher_lines.append(f"FOREACH (x IN {st_list} | MERGE (studio:Studio {{name: x}}) MERGE (g)-[:DEVELOPED_BY]->(studio))")

        if platforms:
            p_list = "[" + ", ".join([f"'{p.replace(chr(39), chr(92)+chr(39))}'" for p in platforms]) + "]"
            cypher_lines.append(f"FOREACH (x IN {p_list} | MERGE (plat:Platform {{name: x}}) MERGE (g)-[:AVAILABLE_ON]->(plat))")

        if characters:
            ch_list = "[" + ", ".join([f"'{c.replace(chr(39), chr(92)+chr(39))}'" for c in characters]) + "]"
            cypher_lines.append(f"FOREACH (x IN {ch_list} | MERGE (char:Character {{name: x}}) MERGE (g)-[:HAS_CHARACTER]->(char))")

        final_cypher = "\n".join(cypher_lines)
        result = self._run_async(self._execute_cypher_async(final_cypher, "write-cypher"))
        return "Errore" not in str(result) and "non disponibile" not in str(result)
