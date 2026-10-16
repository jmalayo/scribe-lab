import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ToolCallRecord(TypedDict):
    tool: str
    args: dict
    latency_ms: float
    error: bool


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    tokens: int
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
