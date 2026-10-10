"""A scripted stand-in for the model, and a tiny generated condition."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from execution_coordinates import data as D
from execution_coordinates import runner as R


@dataclass
class FakeReply:
    message: dict
    model: str = "stub/model"
    prompt_tokens: int = 100
    completion_tokens: int = 10
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.001
    latency_s: float = 0.01


@dataclass
class FakeLLM:
    """policy(messages, tools) -> assistant message dict."""

    policy: Callable[[list[dict], list[dict]], dict]
    seen: list[dict] = field(default_factory=list)

    async def chat(self, model, messages, **kw):
        self.seen.append({"model": model, "messages": messages, **kw})
        return FakeReply(self.policy(messages, kw.get("tools") or []))

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


def tool_call(name: str, args: dict, i: int = 0) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{name}_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


TINY = {"n_rows": 600, "n_customers": 60, "load_s": 0.0, "score_s": 0.0, "requests": 6}


@pytest.fixture
def tiny(tmp_path, monkeypatch):
    """A tiny condition generated into tmp_path, with runs written there too."""
    monkeypatch.setitem(D.CONDITIONS, "tiny", TINY)
    monkeypatch.setattr(D, "DATA", tmp_path / "data")
    monkeypatch.setattr(R, "RUNS", tmp_path / "runs")
    spec = D.generate("tiny", 1, tmp_path / "data")
    return spec


class Analyst:
    """A scripted analyst that answers every request correctly from the
    session's state, with switchable habits:

    reuse      reuse `df`, `rates` and `scores` if the interpreter already has them (else load)
    mutate     do the what-if by changing `rates`, `df` or the customer set in place
    mode       how it calls run_python in the coordinate arms: stateful in a session named after
               the client, or with the what-if read-only
    session    the session name it uses in the coordinate arm (a fixed name makes clients collide)
    """

    def __init__(self, spec: dict, reuse=True, mutate=False, whatif_mode="stateful", session=None):
        self.by_text = {r["text"]: (c, r) for c in spec["clients"] for r in c["requests"]}
        self.reuse, self.mutate, self.whatif_mode, self.session = reuse, mutate, whatif_mode, session

    def code(self, c: dict, r: dict) -> str:
        p = c["params"]
        ckw = {k: c[k] for k in ("idx", "id", "name", "person", "locale", "rep", "currencies", "rates", "prefix",
                                  "regions")}
        reuse = "True" if self.reuse else "False"
        lines = [
            "import json",
            "from execution_coordinates import data as D",
            f"if not ({reuse} and 'df' in globals()):",
            f"    df = D.reference_parse(load_export({c['id']!r}), D.Client(**{ckw!r}))",
            f"if not ({reuse} and 'rates' in globals()):",
            f"    rates = {c['rates']!r}",
        ]
        rid = r["id"]
        pre = c["prefix"]
        if rid == "r1_net_revenue":
            lines.append(f"ans = D.net_revenue(df, rates, {pre!r}, {p['m1']!r})")
        elif rid == "r2_by_region":
            lines.append(f"ans = D.net_revenue_by_region(df, rates, {pre!r}, {p['m1']!r})")
        elif rid.startswith("r3_"):
            kind = r["whatif"]
            if kind == "rate":
                cur, alt = _alt_rate(r["text"])
                if self.mutate:
                    lines.append(f"rates[{cur!r}] = {alt}")
                    lines.append(f"ans = D.net_revenue(df, rates, {pre!r}, {p['m1']!r})")
                else:
                    lines.append(f"ans = D.net_revenue(df, {{**rates, {cur!r}: {alt}}}, {pre!r}, {p['m1']!r})")
            elif kind == "pending":
                if self.mutate:
                    lines.append("df.loc[df['status'] == 'pending', 'status'] = 'settled'")
                    lines.append(f"ans = D.net_revenue(df, rates, {pre!r}, {p['m1']!r})")
                else:
                    lines.append(f"ans = D.net_revenue(df, rates, {pre!r}, {p['m1']!r}, statuses=('settled', 'pending'))")
            else:
                n = int(r["text"].split("without our ")[1].split()[0])
                lines.append(f"top = D.top_customers(df, rates, {pre!r}, {p['m1']!r}, {n})")
                if self.mutate:
                    lines.append("df.drop(df.index[df['customer_id'].isin(top)], inplace=True)")
                    lines.append(f"ans = D.net_revenue(df, rates, {pre!r}, {p['m1']!r})")
                else:
                    lines.append(f"ans = D.net_revenue(df, rates, {pre!r}, {p['m1']!r}, exclude=top)")
        elif rid == "r4_avg_card":
            lines.append(f"ans = D.avg_card_charge(df, rates, {pre!r}, {p['m2']!r})")
        elif rid == "r5_risk_count":
            lines.append(f"if not ({reuse} and 'scores' in globals()):\n    scores = score_customers({c['id']!r})")
            lines.append(f"ans = len(D.high_risk(scores, {pre!r}, {p['thr']}))")
        elif rid == "r6_risk_revenue":
            lines.append(f"if not ({reuse} and 'scores' in globals()):\n    scores = score_customers({c['id']!r})")
            lines.append(f"ans = D.net_revenue(df, rates, {pre!r}, {p['m3']!r}, "
                         f"customers=D.high_risk(scores, {pre!r}, {p['thr']}))")
        lines.append("json.dumps(ans)")
        return "\n".join(lines)

    def __call__(self, messages, tools):
        names = {t["function"]["name"] for t in tools}
        last = messages[-1]
        if last["role"] == "tool":
            out = json.loads(last["content"])
            assert out.get("error") is None, out
            return tool_call("final_answer", {"answer": ast.literal_eval(out["value"])})
        user = [m for m in messages if m["role"] == "user"][-1]
        text = user["content"] if isinstance(user["content"], str) else user["content"][0]["text"]
        c, r = self.by_text[text]
        args = {"code": self.code(c, r)}
        if "list_sessions" in names:  # the coordinate tool
            ro = r["id"].startswith("r3_") and self.whatif_mode == "read_only"
            args.update(mode="read_only" if ro else "stateful", session=self.session or c["id"])
        elif any("mode" in t["function"]["parameters"]["properties"] for t in tools):
            ro = r["id"].startswith("r3_") and self.whatif_mode == "read_only"
            args.update(mode="read_only" if ro else "stateful")
        return tool_call("run_python", args)


def _alt_rate(text: str) -> tuple[str, float]:
    # "...if EUR had been at 0.9493 GBP per EUR instead..."
    after = text.split("if ")[1]
    cur = after.split()[0]
    alt = float(after.split("had been at ")[1].split()[0])
    return cur, alt
