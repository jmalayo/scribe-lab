import anyio
import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_ollama")
pytest.importorskip("langchain_mcp_adapters")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import ToolException, tool
from pydantic import Field

from agent.graph import build_graph, run_agent
from agent.tools import agent_tools, no_hay_respuesta
from shared.prompts import ABSTENTION_ANSWER

USAGE = {
    "input_tokens": 90,
    "output_tokens": 10,
    "total_tokens": 100
}


class ScriptedModel(BaseChatModel):
    replies: list[AIMessage]
    seen: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:

        return "scripted"

    def bind_tools(self, tools, **kwargs):

        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:

        self.seen.append(messages)

        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


@tool(description="Busca fragmentos.")
def buscar_documentos(consulta: str, top_k: int = 5) -> list[dict]:

    return [
        {
            "doc_id": "d.md",
            "chunk_index": 3,
            "score": 0.81,
            "texto": "El agente usa qwen3:8b."
        }
    ]


@tool(description="Busca fragmentos.", name_or_callable="buscar_documentos")
def buscar_documentos_caido(consulta: str) -> list[dict]:

    raise ToolException("La API respondió 503 en /v1/retrieve")


def call(name: str, args: dict, i: int = 0) -> AIMessage:

    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": f"call_{i}"
            }
        ],
        usage_metadata=USAGE,
    )


def answer(text: str) -> AIMessage:

    return AIMessage(content=text, usage_metadata=USAGE)


def run(model: ScriptedModel, tools: list | None = None, **limits):

    graph = build_graph(model, tools or [buscar_documentos, no_hay_respuesta], **limits)

    return anyio.run(run_agent, graph, "¿Qué modelo usa el agente?")


def test_agent_searches_then_answers():

    model = ScriptedModel(
        replies=[
            call("buscar_documentos", {"consulta": "modelo del agente"}),
            answer("Usa qwen3:8b [d.md]."),
        ]
    )

    result = run(model)

    assert result.outcome == "respuesta"
    assert result.answer == "Usa qwen3:8b [d.md]."
    assert result.steps == 2
    assert result.tokens == 200
    assert [(r["tool"], r["args"], r["error"]) for r in result.tool_calls] == [
        (
            "buscar_documentos",
            {"consulta": "modelo del agente"},
            False
        )
    ]
    assert result.tool_calls[0]["latency_ms"] >= 0
    assert isinstance(model.seen[0][0], SystemMessage)
    assert "qwen3:8b" in model.seen[1][-1].content


def test_no_hay_respuesta_ends_with_abstention():

    model = ScriptedModel(
        replies=[
            call("buscar_documentos", {"consulta": "precio"}),
            call("no_hay_respuesta", {"motivo": "los fragmentos no hablan de precios"}, i=1),
        ]
    )

    result = run(model)

    assert result.outcome == "abstencion"
    assert result.answer == ABSTENTION_ANSWER
    assert result.reason == "los fragmentos no hablan de precios"
    assert result.steps == 2
    assert model.replies == []


def test_step_limit_stops_the_loop():

    model = ScriptedModel(replies=[call("buscar_documentos", {"consulta": "q"}, i) for i in range(10)])

    result = run(model, max_steps=3)

    assert result.outcome == "limite"
    assert result.answer is None
    assert result.steps == 3
    assert len(result.tool_calls) == 3


def test_token_limit_stops_the_loop():

    model = ScriptedModel(replies=[call("buscar_documentos", {"consulta": "q"}, i) for i in range(10)])

    result = run(model, max_tokens=150)

    assert result.outcome == "limite"
    assert result.steps == 2
    assert result.tokens == 200


def test_tool_error_goes_back_to_the_model():

    model = ScriptedModel(
        replies=[
            call("buscar_documentos", {"consulta": "q"}),
            answer("No pude consultar los documentos."),
        ]
    )

    result = run(model, tools=[buscar_documentos_caido, no_hay_respuesta])

    assert result.outcome == "respuesta"
    assert result.tool_calls[0]["error"] is True
    assert "respondió 503" in model.seen[1][-1].content


def test_agent_tools_load_search_from_the_mcp_server_plus_local_abstention():

    async def names():
        async with agent_tools() as tools:
            return [t.name for t in tools]

    assert anyio.run(names) == ["buscar_documentos", "no_hay_respuesta"]
