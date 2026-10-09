"""One client for every model call in the demos.

Every call goes to OpenRouter's OpenAI-compatible API, and every call, failed
or not, is appended to a JSONL ledger with its model, tokens, cost (as
OpenRouter reports it), latency and a tag, so each concept's spend is exact.

The key is read from the environment (OPENROUTER_API_KEY) and never printed or
written anywhere. On the laptop, run through `bin/with-openrouter-key`; on the
cloud machine, `set -a; . ~/.experiments.env; set +a` first.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

API = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class LLMError(RuntimeError):
    pass


@dataclass
class Reply:
    """What a call returns: the message, the parsed usage and the timings."""

    message: dict[str, Any]
    model: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    cost: float
    latency_s: float
    first_token_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def text(self) -> str:
        content = self.message.get("content")
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
        return content or ""

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return self.message.get("tool_calls") or []


def _key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise LLMError("OPENROUTER_API_KEY is not set (see bin/with-openrouter-key)")
    return key


class Ledger:
    """Append-only JSONL record of every call. One line per call."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, separators=(",", ":")) + "\n"
        # A single write of one short line in append mode is atomic enough for
        # several processes appending to the same file.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)


class LLM:
    """Async client. Share one per process; it pools connections.

    llm = LLM(ledger="runs/ledger.jsonl", tag="my-concept")
    reply = await llm.chat("anthropic/claude-haiku-5.5", [{"role": "user", "content": "hi"}])
    """

    def __init__(
        self,
        ledger: str | os.PathLike[str],
        tag: str,
        *,
        max_concurrency: int = 16,
        timeout_s: float = 600.0,
        max_retries: int = 6,
    ):
        self.ledger = Ledger(ledger)
        self.tag = tag
        self.max_retries = max_retries
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            base_url=API,
            timeout=httpx.Timeout(timeout_s, connect=30.0),
            limits=httpx.Limits(max_connections=max_concurrency * 2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "LLM":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
            "X-Title": "unify demos",
        }

    def _body(self, model: str, messages: list[dict[str, Any]], kw: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages, "usage": {"include": True}}
        for k in ("tools", "tool_choice", "temperature", "max_tokens", "response_format", "seed", "top_p", "stop"):
            if kw.get(k) is not None:
                body[k] = kw[k]
        if kw.get("reasoning") is not None:
            # OpenRouter's unified reasoning control, e.g. {"effort": "low"} or {"max_tokens": 0}
            body["reasoning"] = kw["reasoning"]
        if kw.get("provider") is not None:
            body["provider"] = kw["provider"]
        return body

    def _record(self, model: str, t0: float, tag: str | None, meta: dict[str, Any] | None, **fields: Any) -> None:
        row = {"ts": round(t0, 3), "tag": tag or self.tag, "model": model, **fields}
        if meta:
            row["meta"] = meta
        self.ledger.write(row)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tag: str | None = None,
        meta: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Reply:
        """One chat completion. kw: tools, tool_choice, temperature, max_tokens,
        response_format, seed, top_p, stop, reasoning, provider."""
        body = self._body(model, messages, kw)
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.time()
                try:
                    r = await self._client.post("/chat/completions", headers=self._headers(), json=body)
                    latency = time.time() - t0
                    if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                        self._record(model, t0, tag, meta, error=f"HTTP {r.status_code}", latency_s=round(latency, 3), attempt=attempt)
                        await asyncio.sleep(min(60, 2**attempt + random.random()))
                        continue
                    data = r.json()
                    if r.status_code != 200 or "choices" not in data:
                        err = data.get("error") if isinstance(data, dict) else None
                        self._record(model, t0, tag, meta, error=f"HTTP {r.status_code}: {json.dumps(err)[:300]}", latency_s=round(latency, 3), attempt=attempt)
                        if attempt < self.max_retries and r.status_code >= 500:
                            await asyncio.sleep(min(60, 2**attempt + random.random()))
                            continue
                        raise LLMError(f"{model}: HTTP {r.status_code}: {json.dumps(err)[:300]}")
                    return self._finish(model, data, t0, latency, None, tag, meta)
                except (httpx.TransportError, json.JSONDecodeError) as e:
                    latency = time.time() - t0
                    self._record(model, t0, tag, meta, error=f"{type(e).__name__}: {e}"[:300], latency_s=round(latency, 3), attempt=attempt)
                    if attempt >= self.max_retries:
                        raise LLMError(f"{model}: {type(e).__name__}: {e}") from e
                    await asyncio.sleep(min(60, 2**attempt + random.random()))
        raise LLMError(f"{model}: retries exhausted")

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tag: str | None = None,
        meta: dict[str, Any] | None = None,
        **kw: Any,
    ) -> Reply:
        """A streamed completion, for measuring time to the first token of content.
        Returns the assembled message (text only; no tool calls) with first_token_s set."""
        body = self._body(model, messages, kw)
        body["stream"] = True
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.time()
                first: float | None = None
                parts: list[str] = []
                usage: dict[str, Any] = {}
                served = model
                try:
                    async with self._client.stream("POST", "/chat/completions", headers=self._headers(), json=body) as r:
                        if r.status_code != 200:
                            text = (await r.aread()).decode(errors="replace")[:300]
                            self._record(model, t0, tag, meta, error=f"HTTP {r.status_code}: {text}", latency_s=round(time.time() - t0, 3), attempt=attempt)
                            if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                                await asyncio.sleep(min(60, 2**attempt + random.random()))
                                continue
                            raise LLMError(f"{model}: HTTP {r.status_code}: {text}")
                        async for line in r.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                d = json.loads(chunk)
                            except json.JSONDecodeError:
                                continue
                            served = d.get("model", served)
                            if d.get("usage"):
                                usage = d["usage"]
                            for ch in d.get("choices", []):
                                delta = (ch.get("delta") or {}).get("content")
                                if delta:
                                    if first is None:
                                        first = time.time() - t0
                                    parts.append(delta)
                    latency = time.time() - t0
                    data = {"model": served, "usage": usage, "choices": [{"message": {"role": "assistant", "content": "".join(parts)}}]}
                    return self._finish(model, data, t0, latency, first, tag, meta)
                except httpx.TransportError as e:
                    self._record(model, t0, tag, meta, error=f"{type(e).__name__}: {e}"[:300], latency_s=round(time.time() - t0, 3), attempt=attempt)
                    if attempt >= self.max_retries:
                        raise LLMError(f"{model}: {type(e).__name__}: {e}") from e
                    await asyncio.sleep(min(60, 2**attempt + random.random()))
        raise LLMError(f"{model}: retries exhausted")

    def _finish(self, model: str, data: dict[str, Any], t0: float, latency: float, first: float | None, tag: str | None, meta: dict[str, Any] | None) -> Reply:
        u = data.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        pdetails = u.get("prompt_tokens_details") or {}
        reply = Reply(
            message=data["choices"][0].get("message") or {},
            model=data.get("model", model),
            prompt_tokens=int(u.get("prompt_tokens") or 0),
            completion_tokens=int(u.get("completion_tokens") or 0),
            reasoning_tokens=int(details.get("reasoning_tokens") or 0),
            cached_tokens=int(pdetails.get("cached_tokens") or 0),
            cost=float(u.get("cost") or 0.0),
            latency_s=latency,
            first_token_s=first,
            raw=data,
        )
        self._record(
            model,
            t0,
            tag,
            meta,
            served=reply.model,
            prompt_tokens=reply.prompt_tokens,
            completion_tokens=reply.completion_tokens,
            reasoning_tokens=reply.reasoning_tokens,
            cached_tokens=reply.cached_tokens,
            cost=reply.cost,
            latency_s=round(latency, 3),
            first_token_s=None if first is None else round(first, 3),
        )
        return reply


def image_part(png_or_jpeg: bytes, mime: str = "image/png", detail: str | None = None) -> dict[str, Any]:
    """An image content part for a user message, as a base64 data URL."""
    import base64

    url = f"data:{mime};base64,{base64.b64encode(png_or_jpeg).decode()}"
    part: dict[str, Any] = {"type": "image_url", "image_url": {"url": url}}
    if detail:
        part["image_url"]["detail"] = detail
    return part
