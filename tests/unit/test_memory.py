import json

import pytest
import requests
from fastapi.testclient import TestClient

from app.deps import get_pipeline
from app.main import app
from shared.llm import LLMResult
from shared.memory import MAX_TURNS, build_messages, parse_rewrite
from shared.prompts import REWRITE_PROMPTS
from tests.unit.test_pipeline import FakeLLM, make_config, make_pipeline

HISTORY = [("¿Qué métrica usa el benchmark?", "Usa balanced accuracy, con perc_ratio eliminado.")]


class FakeRewriter(FakeLLM):

    def __init__(self, question="¿Por qué varía tanto la balanced accuracy?", raw=None, exc=None):
        super().__init__()
        self.raw = raw if raw is not None else json.dumps(
            {"necesita_historial": True, "pregunta": question, "terminos_clave": ["balanced accuracy"]}
        )
        self.exc = exc
        self.messages = []

    def chat(self, messages, json_format=False):
        self.messages.append((messages, json_format))
        if self.exc:
            raise self.exc
        return LLMResult(text=self.raw, tokens_in=1200, tokens_out=60)


def memory_pipeline(rewriter=None, llm=None):

    config = make_config(memory={"enabled": True})

    return make_pipeline(config=config, llm=llm, rewriter=rewriter or FakeRewriter())


def test_parse_keeps_anchored_terms_and_discards_invented_ones():

    out = parse_rewrite(
        '{"necesita_historial": true, "pregunta": "¿Fórmula de balanced accuracy?", '
        '"terminos_clave": ["balanced accuracy", "inventado_xyz"]}',
        HISTORY,
        "¿Y su fórmula?",
    )

    assert out["pregunta"] == "¿Fórmula de balanced accuracy?"
    assert out["terminos_clave"] == ["balanced accuracy"]
    assert out["terminos_descartados"] == ["inventado_xyz"]
    assert out["fallback"] is None


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("no es json", "json_invalido"),
        ("[1, 2]", "json_invalido"),
        ('{"pregunta": "  "}', "vacia"),
        (json.dumps({"pregunta": "palabra " * 61}), "demasiado_larga"),
    ],
)
def test_parse_falls_back_to_the_literal_question(raw, reason):

    out = parse_rewrite(raw, HISTORY, "¿Y su fórmula?")

    assert out["fallback"] == reason
    assert out["pregunta"] == "¿Y su fórmula?"


def test_messages_hold_system_few_shot_and_only_the_last_turns():

    history = [(f"pregunta {i}", f"respuesta {i}") for i in range(4)]
    messages = build_messages(history, "¿Y eso?")
    _, few_shot = REWRITE_PROMPTS["v2"]

    assert messages[0]["role"] == "system"
    assert len(messages) == 1 + 2 * len(few_shot) + 1
    assert "pregunta 0" not in messages[-1]["content"]
    assert f"pregunta {4 - MAX_TURNS}" in messages[-1]["content"]
    assert messages[-1]["content"].endswith("Última pregunta: ¿Y eso?")


def test_long_answers_are_trimmed_in_the_history():

    messages = build_messages([("q", "x" * 1000)], "¿Y eso?")

    assert "x" * 401 not in messages[-1]["content"]
    assert "x" * 400 + "…" in messages[-1]["content"]


def test_without_history_the_rewriter_is_not_called():

    rewriter = FakeRewriter()
    pipeline = memory_pipeline(rewriter=rewriter)

    result = pipeline.run("¿pregunta?")

    assert rewriter.messages == []
    assert result.rewritten_question is None
    assert len(pipeline.client.calls) == 1
    assert "rewrite" not in result.latencies_ms


def test_with_memory_disabled_history_is_ignored():

    rewriter = FakeRewriter()
    pipeline = make_pipeline(rewriter=rewriter)

    result = pipeline.run("¿Y por qué varía tanto?", history=HISTORY)

    assert rewriter.messages == []
    assert result.rewritten_question is None


def test_with_history_both_questions_are_searched_and_the_rewrite_is_answered():

    llm = FakeLLM()
    rewriter = FakeRewriter()
    pipeline = memory_pipeline(rewriter=rewriter, llm=llm)

    result = pipeline.run("¿Y por qué varía tanto?", history=HISTORY)

    assert rewriter.messages[0][1] is True
    assert result.rewritten_question == "¿Por qué varía tanto la balanced accuracy?"
    assert len(pipeline.client.calls) == 2
    assert [c["chunk_id"] for c in result.chunks] == ["c0", "c1", "c2"]
    assert "Pregunta: ¿Por qué varía tanto la balanced accuracy?" in llm.prompts[0]
    assert {"rewrite", "embedding", "retrieval", "llm", "total"} <= set(result.latencies_ms)


def test_an_unchanged_rewrite_searches_once():

    pipeline = memory_pipeline(rewriter=FakeRewriter(question="¿Cuántas canciones tiene el dataset?"))

    result = pipeline.run("¿Cuántas canciones tiene el dataset?", history=HISTORY)

    assert len(pipeline.client.calls) == 1
    assert result.rewritten_question == "¿Cuántas canciones tiene el dataset?"


def test_retrieve_fused_searches_both_questions_without_rewriting():

    rewriter = FakeRewriter()
    pipeline = memory_pipeline(rewriter=rewriter)

    result = pipeline.retrieve_fused("¿Y por qué varía tanto?", "¿Por qué varía tanto la balanced accuracy?")

    assert rewriter.messages == []
    assert len(pipeline.client.calls) == 2
    assert [c["chunk_id"] for c in result.chunks] == ["c0", "c1", "c2"]


@pytest.mark.parametrize(
    "rewriter",
    [
        FakeRewriter(raw="no es json"),
        FakeRewriter(exc=requests.ConnectionError("ollama caído")),
    ],
)
def test_a_failed_rewrite_uses_the_literal_question(rewriter):

    llm = FakeLLM()
    pipeline = memory_pipeline(rewriter=rewriter, llm=llm)

    result = pipeline.run("¿Y por qué varía tanto?", history=HISTORY)

    assert result.rewritten_question == "¿Y por qué varía tanto?"
    assert len(pipeline.client.calls) == 1
    assert "Pregunta: ¿Y por qué varía tanto?" in llm.prompts[0]


def test_memory_enabled_without_rewriter_is_rejected():

    with pytest.raises(ValueError, match="rewriter"):
        make_pipeline(config=make_config(memory={"enabled": True}))


def test_unknown_memory_prompt_is_rejected():

    with pytest.raises(ValueError, match="memory.prompt_version"):
        make_pipeline(config=make_config(memory={"prompt_version": "v9"}))


@pytest.fixture
def api_with_memory():

    pipeline = memory_pipeline()
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    yield TestClient(app, raise_server_exceptions=False)

    app.dependency_overrides.clear()


def test_query_accepts_history_and_returns_the_rewrite(api_with_memory):

    resp = api_with_memory.post(
        "/v1/query",
        json={
            "question": "¿Y por qué varía tanto?",
            "history": [{"question": HISTORY[0][0], "answer": HISTORY[0][1]}],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["rewritten_question"] == "¿Por qué varía tanto la balanced accuracy?"


def test_query_without_history_keeps_the_previous_contract(api_with_memory):

    resp = api_with_memory.post("/v1/query", json={"question": "¿pregunta?"})

    assert resp.status_code == 200
    assert resp.json()["rewritten_question"] is None


def test_history_turns_are_validated(api_with_memory):

    resp = api_with_memory.post(
        "/v1/query",
        json={"question": "¿Y eso?", "history": [{"question": "", "answer": "x"}]},
    )

    assert resp.status_code == 422
