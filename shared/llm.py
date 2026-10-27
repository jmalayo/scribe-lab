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

    def ping(self) -> bool:

        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=5).json()

        except (requests.RequestException, ValueError):
            return False

        return any(
            m.get("name") == self.model 
                for m in tags.get("models", [])
        )

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

    raise ValueError(f"LLM provider no soportado: {provider}")
