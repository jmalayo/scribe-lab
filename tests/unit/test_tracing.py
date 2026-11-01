import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.telemetry import configure_tracing, flush
from shared.retrieval import BM25Index
from tests.unit.test_memory import HISTORY, memory_pipeline
from tests.unit.test_pipeline import make_chunk, make_config, make_pipeline

EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def provider():

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(tracer_provider)

    yield tracer_provider


@pytest.fixture
def spans():

    EXPORTER.clear()

    def finished():
        return {s.name: s for s in EXPORTER.get_finished_spans()}

    return finished


def test_query_is_one_tree_of_stage_spans(spans):

    memory_pipeline().run("¿Y en la otra?", history=HISTORY)
    by_name = spans()
    root = by_name["query"]

    assert set(by_name) == {"query", "rewrite", "retrieve", "embedding", "dense_search", "llm"}
    assert root.parent is None
    assert {by_name[n].parent.span_id for n in ("rewrite", "retrieve", "llm")} == {root.context.span_id}
    assert {by_name[n].parent.span_id for n in ("embedding", "dense_search")} == {by_name["retrieve"].context.span_id}
    assert len({s.context.trace_id for s in by_name.values()}) == 1


def test_spans_carry_openinference_attributes(spans):

    memory_pipeline().run("¿Y en la otra?", history=HISTORY)
    by_name = spans()
    llm = by_name["llm"].attributes
    retrieve = by_name["retrieve"].attributes

    assert by_name["query"].attributes["openinference.span.kind"] == "CHAIN"
    assert by_name["query"].attributes["output.value"] == "respuesta"
    assert (llm["openinference.span.kind"], llm["llm.model_name"]) == ("LLM", "fake-model")
    assert (llm["llm.token_count.prompt"], llm["llm.token_count.completion"]) == (11, 3)
    assert "Pregunta:" in llm["input.value"]
    assert by_name["rewrite"].attributes["output.value"] == "¿Por qué varía tanto la balanced accuracy?"
    assert retrieve["openinference.span.kind"] == "RETRIEVER"
    assert retrieve["retrieval.documents.0.document.id"] == "c0"
    assert retrieve["retrieval.documents.0.document.score"] == pytest.approx(0.9)


def test_hybrid_search_has_bm25_and_fusion_spans(spans):

    pipeline = make_pipeline(config=make_config(method="hybrid_rrf"))
    pipeline._bm25 = BM25Index([make_chunk(i, 0.0) for i in range(5)])
    pipeline.run("texto 1")
    by_name = spans()

    assert {"bm25", "fusion"} <= set(by_name)
    assert by_name["fusion"].attributes["retrieval.documents.0.document.id"]


def test_without_endpoint_tracing_is_not_configured(monkeypatch):

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert configure_tracing() is None
    assert flush() is True
