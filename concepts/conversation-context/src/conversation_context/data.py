"""Items: a scenario, a kind and a conversation, at a length, for a seed.

A seed draws the filler exchanges and where the settling exchange sits; the
settled, open and complete versions of one scenario at one length and seed
share both, so they differ only in the settling exchange and the last message.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

from .scenarios import COMPLETE_IDS, SCENARIOS

DATA = Path(__file__).resolve().parents[2] / "data"
LENGTHS = {"short": 2, "medium": 24, "long": 100, "xlong": 250}  # filler exchanges
KINDS = ("settled", "open", "complete")


@lru_cache(maxsize=1)
def pool() -> list[list[dict]]:
    return json.loads((DATA / "filler_exchanges.json").read_text())["exchanges"]


def _msgs(exchange: list[dict]) -> list[dict]:
    return [{"role": "user" if m["from"] == "Maya" else "assistant", "content": m["text"]} for m in exchange]


def conversation(scenario: dict, kind: str, length: str, seed: int) -> list[dict]:
    """The person's conversation with the assistant, ending with her question."""
    n = LENGTHS[length]
    rng = random.Random(f"{scenario['id']}|{length}|{seed}")
    filler = rng.sample(pool(), n)
    k = n if length == "short" else rng.randint(0, int(0.8 * n))
    middle = scenario["open"] if kind == "open" else scenario["settle"]
    exchanges = filler[:k] + [[{"from": "Maya", "text": middle[0]}, {"from": "Assistant", "text": middle[1]}]] + filler[k:]
    msgs = [m for ex in exchanges for m in _msgs(ex)]
    question = scenario["question_complete"] if kind == "complete" else scenario["question"]
    return msgs + [{"role": "user", "content": question}]


def settle_distance(scenario: dict, length: str, seed: int) -> int:
    """Messages between the settling exchange's first message and the question."""
    conv = conversation(scenario, "settled", length, seed)
    for i, m in enumerate(conv):
        if m["role"] == "user" and m["content"] == scenario["settle"][0]:
            return len(conv) - 1 - i
    raise ValueError("settling exchange not found")


def present(scenario_index: int, seed: int) -> bool:
    """Whether Maya answers if asked: half the runs of each item, the same for every arm."""
    return (scenario_index + seed) % 2 == 0


def items(kinds: tuple[str, ...] = KINDS, scenario_ids: list[str] | None = None) -> list[dict]:
    out = []
    for idx, s in enumerate(SCENARIOS):
        if scenario_ids and s["id"] not in scenario_ids:
            continue
        for kind in kinds:
            if kind == "complete" and s["id"] not in COMPLETE_IDS:
                continue
            out.append({"id": f"{s['id']}-{kind}", "scenario": s["id"], "index": idx, "kind": kind})
    return out
