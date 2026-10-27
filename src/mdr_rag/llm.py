"""Thin client for an OpenAI-compatible endpoint (Ollama, vLLM, ...)."""

from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel

from mdr_rag.config import Settings


class Llm(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def chat(self, messages: list[dict], schema: type[BaseModel] | None = None) -> str: ...


class OpenAiLlm:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=600, max_retries=0
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for i in range(0, len(texts), 16):
            response = self.client.embeddings.create(
                model=self.settings.embed_model, input=texts[i : i + 16]
            )
            vectors.extend(d.embedding for d in response.data)
        return vectors

    def chat(self, messages: list[dict], schema: type[BaseModel] | None = None) -> str:
        extra = {}
        if schema:
            # Constrained decoding: Ollama and vLLM both honour a JSON schema here.
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        response = self.client.chat.completions.create(
            model=self.settings.chat_model, messages=messages, temperature=0, **extra
        )
        return response.choices[0].message.content or ""
