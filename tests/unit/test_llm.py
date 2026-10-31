from types import SimpleNamespace

import pytest
import requests

from shared import llm


class FakeResponse:

    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=SimpleNamespace(status_code=self.status_code))


@pytest.fixture
def posts(monkeypatch):

    calls = []
    queue = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm.requests, "post", fake_post)

    return calls, queue


def test_ollama_parses_text_and_token_counts(posts):

    calls, queue = posts
    queue.append(FakeResponse(body={"response": "  hola  ", "prompt_eval_count": 7, "eval_count": 2}))

    result = llm.OllamaClient("http://ollama:1014/", "llama3.2:3b").complete("p")

    assert (result.text, result.tokens_in, result.tokens_out) == ("hola", 7, 2)
    assert calls[0]["url"] == "http://ollama:1014/api/generate"
    assert calls[0]["json"]["options"]["temperature"] == 0.0


@pytest.mark.parametrize("failure", [requests.ConnectionError(), requests.Timeout(), FakeResponse(429), FakeResponse(503)])
def test_ollama_fails_on_first_error(posts, failure):

    calls, queue = posts
    queue.append(failure)

    with pytest.raises((requests.ConnectionError, requests.Timeout, requests.HTTPError)):
        llm.OllamaClient("http://o", "m").complete("p")

    assert len(calls) == 1
    assert calls[0]["timeout"] == 120


def test_build_llm_selects_provider():

    assert isinstance(llm.build_llm("ollama", "m"), llm.OllamaClient)

    for provider in ("groq", "openai", "bedrock"):
        with pytest.raises(ValueError, match="no soportado"):
            llm.build_llm(provider, "m")
