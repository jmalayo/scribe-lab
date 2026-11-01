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

    assert isinstance(llm.build_llm("bedrock", "amazon.nova-micro-v1:0"), llm.BedrockClient)
    assert isinstance(llm.build_llm("none", "m"), llm.NoLLM)

    for provider in ("groq", "openai"):
        with pytest.raises(ValueError, match="no soportado"):
            llm.build_llm(provider, "m")


@pytest.fixture
def bedrock():

    import boto3
    from botocore.stub import Stubber

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-2",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    with Stubber(client) as stubber:
        yield llm.BedrockClient("amazon.nova-micro-v1:0", max_tokens=200, client=client), stubber


def converse_reply(text, tokens_in, tokens_out):

    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": tokens_in,
            "outputTokens": tokens_out,
            "totalTokens": tokens_in + tokens_out,
        },
        "metrics": {"latencyMs": 120},
    }


def test_bedrock_complete_sends_one_user_message(bedrock):

    client, stubber = bedrock
    stubber.add_response(
        "converse",
        converse_reply(" respuesta ", 900, 12),
        {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [{"role": "user", "content": [{"text": "prompt"}]}],
            "inferenceConfig": {"maxTokens": 200, "temperature": 0.0},
        },
    )

    result = client.complete("prompt")

    assert (result.text, result.tokens_in, result.tokens_out) == ("respuesta", 900, 12)
    stubber.assert_no_pending_responses()


def test_bedrock_chat_moves_system_messages_out_of_the_turns(bedrock):

    client, stubber = bedrock
    stubber.add_response(
        "converse",
        converse_reply('{"pregunta": "q"}', 300, 8),
        {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [
                {"role": "user", "content": [{"text": "antes"}]},
                {"role": "assistant", "content": [{"text": "algo"}]},
                {"role": "user", "content": [{"text": "¿y eso?"}]},
            ],
            "system": [{"text": "reescribe"}],
            "inferenceConfig": {"maxTokens": 200, "temperature": 0.0},
        },
    )

    result = client.chat(
        [
            {"role": "system", "content": "reescribe"},
            {"role": "user", "content": "antes"},
            {"role": "assistant", "content": "algo"},
            {"role": "user", "content": "¿y eso?"},
        ],
        json_format=True,
    )

    assert result.text == '{"pregunta": "q"}'


def test_bedrock_errors_reach_the_caller(bedrock):

    from botocore.exceptions import ClientError

    client, stubber = bedrock
    stubber.add_client_error("converse", service_error_code="ThrottlingException", http_status_code=429)

    with pytest.raises(ClientError, match="ThrottlingException"):
        client.complete("prompt")


def test_no_llm_refuses_to_generate_but_stays_ready():

    client = llm.NoLLM()

    assert client.ping() is True

    with pytest.raises(llm.LLMUnavailableError):
        client.complete("prompt")
