from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

from shared.settings import settings


@dataclass
class LLMResult:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None

class LLMClient(Protocol):

    provider: str
    model: str

    def complete(self, prompt: str) -> LLMResult: ...

    def chat(self, messages: list[dict], json_format: bool | dict = False) -> LLMResult: ...

    def ping(self) -> bool: ...

class OllamaClient:

    provider = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        num_predict: int = 500,
        num_ctx: int = 4096,
        repeat_penalty: float | None = 1.0,
        stop: list[str] | None = None,
        timeout: float = 120,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.options = {
            "temperature": 0.0,
            "num_ctx": num_ctx,
            "num_thread": 8,
            "num_predict": num_predict,
        }
        if stop:
            self.options["stop"] = stop
        if repeat_penalty:
            self.options["repeat_penalty"] = repeat_penalty
        self.timeout = timeout

    def complete(self, prompt: str) -> LLMResult:

        resp = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model, 
                "prompt": prompt, 
                "stream": False, 
                "options": self.options
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()

        body = resp.json()

        return LLMResult(
            text=body["response"].strip(),
            tokens_in=body.get("prompt_eval_count"),
            tokens_out=body.get("eval_count"),
        )

    def chat(self, messages: list[dict], json_format: bool | dict = False) -> LLMResult:

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": self.options,
        }

        if json_format:
            payload["format"] = json_format if isinstance(json_format, dict) else "json"

        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()

        body = resp.json()

        return LLMResult(
            text=body.get("message", {}).get("content", "").strip(),
            tokens_in=body.get("prompt_eval_count"),
            tokens_out=body.get("eval_count"),
        )

    def ping(self) -> bool:

        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=5).json()

        except (requests.RequestException, ValueError):
            return False

        return any(
            m.get("name") == self.model 
                for m in tags.get("models", [])
        )

class LLMUnavailableError(RuntimeError):
    pass

class NoLLM:

    provider = "none"
    model = "none"

    def complete(self, prompt: str) -> LLMResult:

        raise LLMUnavailableError("Este despliegue no tiene modelo de lenguaje configurado.")

    def chat(self, messages: list[dict], json_format: bool | dict = False) -> LLMResult:

        raise LLMUnavailableError("Este despliegue no tiene modelo de lenguaje configurado.")

    def ping(self) -> bool:

        return True

class BedrockClient:

    provider = "bedrock"

    def __init__(
        self,
        model: str,
        max_tokens: int = 500,
        region: str | None = None,
        client=None,
        control_client=None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.region = region or settings.aws_region
        self._client = client
        self._control_client = control_client

    @property
    def client(self):

        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)

        return self._client

    def _converse(self, messages: list[dict], system: list[dict] | None = None) -> LLMResult:

        request = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": 0.0,
            },
        }

        if system:
            request["system"] = system

        body = self.client.converse(**request)
        content = body["output"]["message"]["content"]
        usage = body.get("usage", {})

        return LLMResult(
            text="".join(part.get("text", "") for part in content).strip(),
            tokens_in=usage.get("inputTokens"),
            tokens_out=usage.get("outputTokens"),
        )

    def complete(self, prompt: str) -> LLMResult:

        return self._converse([{"role": "user", "content": [{"text": prompt}]}])

    def chat(self, messages: list[dict], json_format: bool | dict = False) -> LLMResult:

        system = [
            {"text": m["content"]}
                for m in messages if m["role"] == "system"
        ]
        turns = [
            {"role": m["role"], "content": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"
        ]

        return self._converse(turns, system)

    def ping(self) -> bool:

        try:
            if self._control_client is None:
                import boto3

                self._control_client = boto3.client("bedrock", region_name=self.region)

            self._control_client.get_foundation_model(modelIdentifier=self.model.removeprefix("us."))

        except Exception:
            return False

        return True

def build_llm(
    provider: str,
    model: str,
    max_tokens: int = 500,
    base_url: str | None = None,
    repeat_penalty: float | None = 1.0,
) -> LLMClient:

    if provider == "ollama":
        return OllamaClient(
            host=base_url or settings.llm_host,
            model=model,
            num_predict=max_tokens,
            repeat_penalty=repeat_penalty,
        )

    if provider == "bedrock":
        return BedrockClient(model=model, max_tokens=max_tokens)

    if provider == "none":
        return NoLLM()

    raise ValueError(f"LLM provider no soportado: {provider}")
