from pydantic import BaseModel, Field

from shared.eval.grounding import GroundingResult


class HistoryTurn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(max_length=8000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int | None = None
    score: float | None = None
    text: str
    tables: list[str] = []


class Usage(BaseModel):
    tokens_in: int | None = None
    tokens_out: int | None = None


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    citations: list[Citation]
    grounding: GroundingResult | None
    latency_ms: float
    latencies_ms: dict[str, float]
    usage: Usage
    provider: str | None
    model: str | None
    pipeline_version: str
    prompt_version: str
    rewritten_question: str | None = None


class RetrieveResponse(BaseModel):
    chunks: list[Citation]
    latency_ms: float
    latencies_ms: dict[str, float]
    pipeline_version: str

