"""Reduce prompt size before sending requests to an OpenAI-compatible API."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import tiktoken
from openai import OpenAI


_PROTECTED_BLOCK = re.compile(
    r"(?P<protected>"
    r"(?:^|\n)(?:CRITICAL|REQUIREMENTS?|FILES?|ERRORS?|INSTRUCTIONS?):[^\n]*"
    r"(?:\n(?![A-Z][A-Z _-]{2,}:).*)*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TokenStats:
    before: int
    after: int


class TokenOptimizer:
    """Compress non-critical prompt content and optionally use a GPTCache server.

    Set ``GPTCACHE_URI`` to a running GPTCache server (for example,
    ``http://localhost:8000``) to cache repeated requests.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        compression_rate: float = 0.5,
        cache_uri: str | None = None,
    ) -> None:
        if not 0 < compression_rate <= 1:
            raise ValueError("compression_rate must be greater than 0 and at most 1")
        self.model = model
        self.compression_rate = compression_rate
        self.cache_uri = cache_uri or os.getenv("GPTCACHE_URI")
        self.client = OpenAI()
        self._encoding = tiktoken.encoding_for_model(model)
        self.last_stats: TokenStats | None = None

    def _count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def _compress(self, text: str) -> str:
        from llmlingua import PromptCompressor

        compressor = PromptCompressor(use_llmlingua2=True)
        result = compressor.compress_prompt(text, rate=self.compression_rate)
        return result["compressed_prompt"]

    def compress(self, prompt: str) -> str:
        """Compress only non-critical sections while preserving protected text."""
        protected: list[str] = []

        def replace(match: re.Match[str]) -> str:
            protected.append(match.group("protected"))
            return f"\n__PROTECTED_{len(protected) - 1}__\n"

        template = _PROTECTED_BLOCK.sub(replace, prompt)
        compressed = self._compress(template) if self._count(template) > 100 else template
        for index, value in enumerate(protected):
            compressed = compressed.replace(f"__PROTECTED_{index}__", value.strip())
        self.last_stats = TokenStats(self._count(prompt), self._count(compressed))
        return compressed

    def _cache(self):
        if not self.cache_uri:
            return None
        from gptcache.client import Client

        return Client(uri=self.cache_uri)

    @staticmethod
    def _cache_key(messages: Iterable[dict[str, Any]], model: str) -> str:
        payload = json.dumps(
            {"model": model, "messages": list(messages)},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Compress user messages, then return a cached or fresh model response."""
        optimized: list[dict[str, Any]] = []
        for message in messages:
            item = dict(message)
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                item["content"] = self.compress(item["content"])
            optimized.append(item)

        cache = self._cache()
        key = self._cache_key(optimized, self.model)
        if cache:
            try:
                cached = cache.get(key)
            except Exception:
                cached = None
            if cached:
                return cached

        response = self.client.chat.completions.create(
            model=self.model,
            messages=optimized,
            **kwargs,
        )
        answer = response.choices[0].message.content or ""
        if cache:
            try:
                cache.put(key, answer)
            except Exception:
                pass
        return answer


if __name__ == "__main__":
    optimizer = TokenOptimizer()
    prompt = input("Consulta: ")
    print(optimizer.chat([{"role": "user", "content": prompt}]))
    if optimizer.last_stats:
        stats = optimizer.last_stats
        print(f"Tokens: {stats.before} -> {stats.after}")
