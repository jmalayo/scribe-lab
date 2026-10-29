import time
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState, ToolCallRecord
from agent.tools import no_hay_respuesta
from shared.prompts import ABSTENTION_ANSWER
from shared.settings import settings

MAX_STEPS = 6
MAX_TOKENS = 24000
NUM_CTX = 8192


@dataclass
class AgentResult:
    answer: str | None
    outcome: Literal["respuesta", "abstencion", "limite"]
    reason: str | None
    steps: int
    tokens: int
    tool_calls: list[ToolCallRecord]
    latency_ms: float


def build_llm(model: str = settings.llm_model_agent) -> ChatOllama:

    return ChatOllama(
        model=model,
        base_url=settings.llm_host,
        temperature=0,
        num_ctx=NUM_CTX,
        reasoning=False,
    )


async def _timed(request, execute):

    t0 = time.perf_counter()
    message = await execute(request)
    message.response_metadata["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    return message


def build_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
) -> CompiledStateGraph:

    model = llm.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True, awrap_tool_call=_timed)

    async def call_model(state: AgentState) -> dict:

        response = await model.ainvoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        usage = response.usage_metadata or {}

        return {
            "messages": [response],
            "steps": state["steps"] + 1,
            "tokens": state["tokens"] + usage.get("total_tokens", 0),
        }

    async def call_tools(state: AgentState) -> dict:

        args = {c["id"]: c["args"] for c in state["messages"][-1].tool_calls}
        output = await tool_node.ainvoke(state)

        records = [
            ToolCallRecord(
                tool=m.name,
                args=args[m.tool_call_id],
                latency_ms=m.response_metadata.get("latency_ms", 0.0),
                error=m.status == "error",
            )
            for m in output["messages"]
        ]

        return {"messages": output["messages"], "tool_calls": records}

    def after_model(state: AgentState) -> str:

        return "tools" if state["messages"][-1].tool_calls else END

    def after_tools(state: AgentState) -> str:

        abstained = any(r["tool"] == no_hay_respuesta.name for r in state["tool_calls"])
        exhausted = state["steps"] >= max_steps or state["tokens"] >= max_tokens

        return END if abstained or exhausted else "model"

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", after_model, ["tools", END])
    graph.add_conditional_edges("tools", after_tools, ["model", END])

    return graph.compile()


async def run_agent(graph: CompiledStateGraph, question: str) -> AgentResult:

    t0 = time.perf_counter()
    state = await graph.ainvoke(
        {
            "messages": [HumanMessage(question)],
            "steps": 0,
            "tokens": 0,
            "tool_calls": [],
        }
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    last = state["messages"][-1]
    abstention = next(
        (r for r in state["tool_calls"] if r["tool"] == no_hay_respuesta.name),
        None,
    )

    if abstention:
        outcome, answer, reason = "abstencion", ABSTENTION_ANSWER, abstention["args"].get("motivo")
    elif isinstance(last, AIMessage) and not last.tool_calls:
        outcome, answer, reason = "respuesta", last.content, None
    else:
        outcome, answer, reason = "limite", None, None

    return AgentResult(
        answer=answer,
        outcome=outcome,
        reason=reason,
        steps=state["steps"],
        tokens=state["tokens"],
        tool_calls=state["tool_calls"],
        latency_ms=latency_ms,
    )
