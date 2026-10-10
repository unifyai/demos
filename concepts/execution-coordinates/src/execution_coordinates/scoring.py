"""Correctness and the attribution of wrong answers, shared by the runner and
the analysis."""

from __future__ import annotations

from typing import Any

ABS_TOL = 0.05  # money is asked for rounded to 2 decimals


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= ABS_TOL + 1e-9 * abs(float(b))


def _norm(k: str) -> str:
    return " ".join(str(k).lower().replace("&", "and").split())


def matches(answer: Any, target: Any, answer_type: str) -> bool:
    if answer is None or target is None:
        return False
    if answer_type == "regions" or isinstance(target, dict):
        if not isinstance(answer, dict) or not isinstance(target, dict):
            return False
        a = {_norm(k): v for k, v in answer.items()}
        t = {_norm(k): v for k, v in target.items()}
        return set(a) == set(t) and all(close(a[k], t[k]) for k in t)
    if isinstance(answer, dict):
        return False
    if answer_type == "count":
        return abs(float(answer) - float(target)) < 0.5
    return close(answer, target)


def correct(req: dict, answer: Any) -> bool:
    return matches(answer, req["truth"], req["answer_type"])


def signature_causes(req: dict, answer: Any, clients_in_run: set[str]) -> list[str]:
    """For a wrong answer, which leftover-or-clobbered-state signatures it equals."""
    if answer is None or correct(req, answer):
        return []
    out = []
    for s in req.get("signatures", []):
        cause = s["cause"]
        if cause.startswith("leak:") and cause[5:] not in clients_in_run:
            continue
        if matches(answer, s["value"], req["answer_type"]):
            out.append(f"{cause} ({s['how']})")
    return out
