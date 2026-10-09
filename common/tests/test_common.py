import asyncio
import json

import httpx
import pytest

from demos_common import LLM, LLMError
from demos_common.stats import bootstrap_diff, fisher_exact, quantile, summary, wilson


def test_wilson_known_values():
    lo, hi = wilson(20, 20)
    assert hi == pytest.approx(1.0)
    assert lo == pytest.approx(0.8389, abs=1e-3)
    lo, hi = wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12)
    assert hi == pytest.approx(0.2775, abs=1e-3)


def test_fisher_exact_matches_scipy_reference():
    # scipy.stats.fisher_exact([[4, 16], [20, 0]]) two-sided p = 1.45e-07 (approximately)
    assert fisher_exact(4, 20, 20, 20) == pytest.approx(1.5e-7, rel=0.1)
    assert fisher_exact(5, 10, 5, 10) == pytest.approx(1.0)


def test_quantile_and_summary():
    xs = [1, 2, 3, 4, 5]
    assert quantile(xs, 0.5) == 3
    assert summary(xs)["mean"] == 3
    assert summary(xs)["p10"] == pytest.approx(1.4)


def test_bootstrap_diff_paired_and_zero_effect():
    a = {f"i{k}": [0.0, 0.0, 1.0] for k in range(20)}
    b = {f"i{k}": [1.0, 1.0, 1.0] for k in range(20)}
    r = bootstrap_diff(a, b, iters=2000)
    assert r["diff"] == pytest.approx(2 / 3)
    assert r["lo"] == pytest.approx(2 / 3) and r["hi"] == pytest.approx(2 / 3)
    same = bootstrap_diff(a, a, iters=500)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0


def _client(tmp_path, handler, **kw):
    llm = LLM(tmp_path / "ledger.jsonl", "test", max_retries=2, **kw)
    llm._client = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(handler))
    return llm


def _ok(content="hello", cost=0.001):
    return {
        "model": "x/served",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": cost,
                  "completion_tokens_details": {"reasoning_tokens": 1}, "prompt_tokens_details": {"cached_tokens": 3}},
    }


def test_chat_parses_usage_and_writes_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers["authorization"]
        return httpx.Response(200, json=_ok())

    async def go():
        async with _client(tmp_path, handler) as llm:
            return await llm.chat("x/model", [{"role": "user", "content": "hi"}], temperature=0, meta={"item": "a"})

    r = asyncio.run(go())
    assert r.text == "hello" and r.cost == 0.001 and r.prompt_tokens == 10 and r.reasoning_tokens == 1 and r.cached_tokens == 3
    assert seen["body"]["usage"] == {"include": True} and seen["body"]["temperature"] == 0
    assert seen["auth"] == "Bearer test-key-not-real"
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["cost"] == 0.001 and rows[0]["meta"] == {"item": "a"}
    assert "test-key-not-real" not in (tmp_path / "ledger.jsonl").read_text()


def test_chat_retries_then_succeeds_and_logs_each_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}}) if calls["n"] < 3 else httpx.Response(200, json=_ok())

    async def go():
        async with _client(tmp_path, handler) as llm:
            return await llm.chat("x/model", [{"role": "user", "content": "hi"}])

    r = asyncio.run(go())
    assert r.text == "hello" and calls["n"] == 3
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [("error" in r) for r in rows] == [True, True, False]


def test_chat_raises_on_client_error_without_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async def go():
        async with _client(tmp_path, handler) as llm:
            await llm.chat("x/model", [{"role": "user", "content": "hi"}])

    with pytest.raises(LLMError):
        asyncio.run(go())
    assert calls["n"] == 1


def test_stream_measures_first_token(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    sse = "".join(
        "data: " + json.dumps(c) + "\n\n"
        for c in [
            {"model": "x/served", "choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Let me "}}]},
            {"choices": [{"delta": {"content": "check."}}]},
            {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.0002}},
        ]
    ) + "data: [DONE]\n\n"

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    async def go():
        async with _client(tmp_path, handler) as llm:
            return await llm.chat_stream("x/model", [{"role": "user", "content": "hi"}])

    r = asyncio.run(go())
    assert r.text == "Let me check." and r.first_token_s is not None and r.cost == 0.0002


def test_missing_key_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    async def go():
        async with _client(tmp_path, lambda r: httpx.Response(200, json=_ok())) as llm:
            await llm.chat("x/model", [{"role": "user", "content": "hi"}])

    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        asyncio.run(go())


async def _no_sleep(*_a, **_k):
    return None
