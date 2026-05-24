import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import concurrent.futures

class KGManager:
    """
    Gestore del Knowledge Graph tramite protocollo MCP.
    Utilizza direttamente mcp.client per evitare problemi di TaskGroup.
    """
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
        """Helper per eseguire coroutine async in contesto sincrono."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except Exception as e:
            return f"Errore async: {e}"

    async def _query_async(self, entity: str) -> str:
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = await load_mcp_tools(session)
                    cypher_tool = next((t for t in tools if t.name == "read-cypher"), None)

                    if not cypher_tool:
                        return "Tool 'read-cypher' non trovato nel server MCP Neo4j."

                    safe_entity = entity.replace("'", "\\'")

                    cypher = f"""
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS toLower('{safe_entity}')
                    OR toLower(n.title) CONTAINS toLower('{safe_entity}')
                    RETURN n.name AS Name, n.title AS Title, n.type AS Formato
                    LIMIT 10
                    """
                    result = await cypher_tool.ainvoke({"query": cypher})
                    if not result:
                        return f"Nessun risultato nel KG per: '{entity}'"
                    return f"Risultati KG per '{entity}':\n{result}"
        except Exception as e:
            return f"KG non disponibile: {e}"

    async def _update_async(self, post_title: str, topic: str, post_type: str) -> bool:
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    tools = await load_mcp_tools(session)
                    cypher_tool = next((t for t in tools if t.name == "write-cypher"), None)

                    if not cypher_tool:
                        print("Tool 'write-cypher' non trovato. Assicurati che NEO4J_READ_ONLY non sia true.")
                        return False

                    safe_topic = topic.replace("'", "\\'")
                    safe_title = post_title.replace("'", "\\'")

                    cypher = f"""
                    MERGE (t:Topic {{name: '{safe_topic}'}})
                    MERGE (b:BlogPost {{title: '{safe_title}'}})
                    ON CREATE SET b.type = '{post_type}', b.created_at = datetime()
                    MERGE (t)-[:COVERED_IN]->(b)
                    """
                    await cypher_tool.ainvoke({"query": cypher})
                    return True
        except Exception as e:
            print(f"Errore KG update: {e}")
            return False

    def query(self, entity: str) -> str:
        return self._run_async(self._query_async(entity))

    def update(self, post_title: str, topic: str, post_type: str) -> bool:
        return self._run_async(self._update_async(post_title, topic, post_type))
