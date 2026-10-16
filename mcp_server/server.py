from mcp.server.fastmcp import FastMCP

from mcp_server import tools

server = FastMCP("scribe-lab")

server.tool(
    description=(
        "Busca en la base documental los fragmentos más parecidos a la consulta y los devuelve "
        "con su documento, posición y score de similitud, sin generar una respuesta."
    )
)(tools.buscar_documentos)

server.tool(
    description=(
        "Responde una pregunta con el pipeline RAG completo: recupera fragmentos, genera la "
        "respuesta con el LLM y devuelve las fuentes usadas."
    )
)(tools.preguntar)

if __name__ == "__main__":
    server.run()
