import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from shared.prompts import ABSTENTION_ANSWER
from shared.settings import settings

MCP_SERVER = "scribe-lab"
MCP_TOOLS = ("buscar_documentos",)
REPO_ROOT = Path(__file__).resolve().parents[1]


@tool(
    description=(
        "Declara que los documentos recuperados no contienen la respuesta. Úsala en lugar de "
        "responder cuando la búsqueda no encuentre la información."
    )
)
def no_hay_respuesta(motivo: str) -> str:

    return ABSTENTION_ANSWER


@asynccontextmanager
async def agent_tools() -> AsyncIterator[list[BaseTool]]:

    client = MultiServerMCPClient(
        {
            MCP_SERVER: {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "cwd": str(REPO_ROOT),
                "env": {"SCRIBE_API_URL": settings.scribe_api_url},
            }
        }
    )

    async with client.session(MCP_SERVER) as session:
        tools = await load_mcp_tools(session)

        yield [t for t in tools if t.name in MCP_TOOLS] + [no_hay_respuesta]
