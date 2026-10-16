import anyio
import httpx
import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session

from mcp_server import tools
from mcp_server.server import server

CHUNK = {
    "chunk_id": "c1",
    "doc_id": "d.md",
    "chunk_index": 3,
    "score": 0.81,
    "text": "texto",
    "tables": ["| a | b |"]
}

QUERY_BODY = {
    "answer": "respuesta",
    "abstained": False,
    "citations": [CHUNK],
    "grounding": None,
    "latency_ms": 120.5,
    "latencies_ms": {"total": 120.5},
    "usage": {"tokens_in": 10, "tokens_out": 5},
    "provider": "ollama",
    "model": "llama3.2:3b",
    "pipeline_version": "test",
    "prompt_version": "v1",
}


@pytest.fixture
def api(monkeypatch):

    calls = []
    replies = []

    def post(url, json, timeout):
        calls.append((url, json, timeout))
        reply = replies.pop(0)

        if isinstance(reply, Exception):
            raise reply

        return reply

    monkeypatch.setattr(tools.settings, "scribe_api_url", "http://api")
    monkeypatch.setattr(tools.httpx, "post", post)

    return calls, replies


def ok(body):

    return httpx.Response(200, json=body)


def test_buscar_documentos_posts_to_retrieve(api):

    calls, replies = api
    replies.append(
        ok(
            {
                "chunks": [CHUNK],
                "latency_ms": 5.0,
                "latencies_ms": {},
                "pipeline_version": "t"
            }
        )
    )

    result = tools.buscar_documentos("¿qué modelo?", top_k=3)

    assert calls == [
        (
            "http://api/v1/retrieve",
            {"query": "¿qué modelo?", "top_k": 3},
            tools.RETRIEVE_TIMEOUT
        )
    ]

    assert result == [
        {
            "doc_id": "d.md",
            "chunk_index": 3,
            "score": 0.81,
            "texto": "texto\n\n| a | b |"
        }
    ]


def test_preguntar_posts_to_query(api):

    calls, replies = api
    replies.append(ok(QUERY_BODY))

    result = tools.preguntar("¿qué modelo?")

    assert calls == [
        (
            "http://api/v1/query",
            {"question": "¿qué modelo?"},
            tools.QUERY_TIMEOUT
        )
    ]

    assert result == {
        "respuesta": "respuesta",
        "abstencion": False,
        "fuentes": [
            {
                "doc_id": "d.md",
                "chunk_index": 3,
                "score": 0.81
            }
        ],
        "modelo": "llama3.2:3b",
        "latencia_ms": 120.5,
    }


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (httpx.Response(503, json={"detail": "Un servicio interno no está disponible."}), "respondió 503"),
        (httpx.ReadTimeout("t"), "no respondió en 30 s"),
        (httpx.ConnectError("c"), "No se pudo conectar"),
    ],
)
def test_api_failures_become_readable_errors(api, reply, message):

    _, replies = api
    replies.append(reply)

    with pytest.raises(RuntimeError, match=message):
        tools.buscar_documentos("q")


def test_server_lists_both_tools_with_their_parameters():

    listed = {t.name: t for t in anyio.run(server.list_tools)}

    assert set(listed) == {"buscar_documentos", "preguntar"}
    assert set(listed["buscar_documentos"].inputSchema["properties"]) == {"consulta", "top_k"}
    assert listed["buscar_documentos"].inputSchema["required"] == ["consulta"]
    assert listed["preguntar"].inputSchema["required"] == ["pregunta"]


def test_api_failure_reaches_the_client_as_a_tool_error(api):

    _, replies = api
    replies.append(httpx.Response(503, json={"detail": "Un servicio interno no está disponible."}))

    async def call():
        async with create_connected_server_and_client_session(server) as client:
            return await client.call_tool("buscar_documentos", {"consulta": "q"})

    result = anyio.run(call)

    assert result.isError is True
    assert "respondió 503" in result.content[0].text
