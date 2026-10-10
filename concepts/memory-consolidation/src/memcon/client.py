"""The one model client, plus embeddings.

Every chat call goes through demos_common.LLM. OpenRouter also serves embedding
models, which the shared client has no method for, so `Client` adds `embed`
on top of it: the same connection pool, key handling, retries and ledger, one
ledger row per call with OpenRouter's reported cost. (Flagged for the lead to
fold into common/.)
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx
from demos_common import LLM, LLMError
from demos_common.llm import RETRY_STATUS

EMBED_MODEL = "voyageai/voyage-4"


@dataclass
class Embedded:
    vectors: list[list[float]]
    latency_s: float
    cost: float
    prompt_tokens: int


class Client(LLM):
    async def embed(
        self,
        texts: list[str],
        *,
        model: str = EMBED_MODEL,
        tag: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Embedded:
        body = {"model": model, "input": texts}
        async with self._sem:
            for attempt in range(self.max_retries + 1):
                t0 = time.time()
                try:
                    r = await self._client.post("/embeddings", headers=self._headers(), json=body)
                    latency = time.time() - t0
                    if r.status_code in RETRY_STATUS and attempt < self.max_retries:
                        self._record(model, t0, tag, meta, error=f"HTTP {r.status_code}", latency_s=round(latency, 3), attempt=attempt)
                        await asyncio.sleep(min(60, 2**attempt + random.random()))
                        continue
                    data = r.json()
                    if r.status_code != 200 or "data" not in data:
                        err = data.get("error") if isinstance(data, dict) else None
                        self._record(model, t0, tag, meta, error=f"HTTP {r.status_code}: {json.dumps(err)[:300]}", latency_s=round(latency, 3), attempt=attempt)
                        if attempt < self.max_retries and r.status_code >= 500:
                            await asyncio.sleep(min(60, 2**attempt + random.random()))
                            continue
                        raise LLMError(f"{model}: HTTP {r.status_code}: {json.dumps(err)[:300]}")
                    u = data.get("usage") or {}
                    tokens = int(u.get("prompt_tokens") or u.get("total_tokens") or 0)
                    cost = float(u.get("cost") or 0.0)
                    self._record(
                        model, t0, tag, meta,
                        served=data.get("model", model),
                        prompt_tokens=tokens, completion_tokens=0, reasoning_tokens=0, cached_tokens=0,
                        cost=cost, latency_s=round(latency, 3),
                    )
                    rows = sorted(data["data"], key=lambda d: d.get("index", 0))
                    if len(rows) != len(texts):
                        raise LLMError(f"{model}: {len(rows)} embeddings for {len(texts)} texts")
                    return Embedded([row["embedding"] for row in rows], latency, cost, tokens)
                except (httpx.TransportError, json.JSONDecodeError) as e:
                    latency = time.time() - t0
                    self._record(model, t0, tag, meta, error=f"{type(e).__name__}: {e}"[:300], latency_s=round(latency, 3), attempt=attempt)
                    if attempt >= self.max_retries:
                        raise LLMError(f"{model}: {type(e).__name__}: {e}") from e
                    await asyncio.sleep(min(60, 2**attempt + random.random()))
        raise LLMError(f"{model}: retries exhausted")
