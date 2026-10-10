"""The explicit timing model.

A model call started at t0 shows its first visible token at
    t0 + a + b * prompt_ktokens + reasoning_tokens / rr
and finishes at
    t0 + a + b * prompt_ktokens + reasoning_tokens / rr + visible_tokens / r
(rr defaults to r; it was added in Plan amendment 3 for the era models, which reason). a, b and r per model come from
calibrate.py and are stored in data/timing.json. Speech, end-of-turn detection
and barge-in use the constants below, the same for every arm.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelTiming:
    a: float  # seconds to the first token for an empty prompt (network, queueing, prefill floor)
    b: float  # extra seconds per 1000 prompt tokens
    r: float  # visible output tokens per second
    rr: float | None = None  # reasoning tokens per second (None: the same as r)


@dataclass
class Timing:
    models: dict[str, ModelTiming] = field(default_factory=dict)
    tts: float = 0.25  # text to first audio
    agent_spw: float = 0.40  # agent seconds per word (150 words a minute)
    person_spw: float = 0.35  # Sam's seconds per word
    endpoint: float = 0.5  # end-of-turn detection after Sam stops
    barge_reaction: float = 0.2  # the agent stops this long after Sam starts over it
    delay: dict[str, float] = field(default_factory=dict)  # extra seconds to the first token, by model (the slow-brain sweep)

    def call_times(self, model: str, prompt_tokens: int, completion_tokens: int, reasoning_tokens: int,
                   prefix_tokens: int = 0) -> tuple[float, float]:
        """(seconds to the first visible token, seconds to the end) after the call starts.
        prefix_tokens: visible tokens that come before the spoken text (a label)."""
        m = self.models[model]
        rr = m.rr or m.r
        base = m.a + self.delay.get(model, 0.0) + m.b * prompt_tokens / 1000.0
        first = base + reasoning_tokens / rr + prefix_tokens / m.r
        done = base + reasoning_tokens / rr + max(completion_tokens - reasoning_tokens, prefix_tokens) / m.r
        return first, done

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Timing":
        d = dict(d)
        d["models"] = {k: ModelTiming(**v) for k, v in d.get("models", {}).items()}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> "Timing":
        return cls.from_json(json.loads(Path(path).read_text()))


def token_weights(text: str) -> list[tuple[str, float]]:
    """Each whitespace token with its length in words: a token with three or more
    digits takes half a word per digit (312-555-0147 is five words)."""
    out = []
    for tok in text.split():
        d = sum(ch.isdigit() for ch in tok)
        out.append((tok, max(1.0, d / 2.0) if d >= 3 else 1.0))
    return out


def speech_seconds(text: str, spw: float) -> float:
    return sum(w for _, w in token_weights(text)) * spw


def heard_prefix(text: str, spw: float, seconds: float) -> tuple[str, str]:
    """Split text into what was heard in the first `seconds` of speaking (whole
    tokens only) and the rest."""
    heard, t = [], 0.0
    toks = token_weights(text)
    for i, (tok, w) in enumerate(toks):
        t += w * spw
        if t > seconds + 1e-9:
            return " ".join(heard), " ".join(x for x, _ in toks[i:])
        heard.append(tok)
    return " ".join(heard), ""


_SENT_END = re.compile(r"(?<=[.!?])\s+")


def resume_from_sentence(text: str, n_heard_tokens: int) -> str:
    """The rest of a cut line, from the start of the sentence in which it was cut."""
    toks = list(re.finditer(r"\S+", text))
    if n_heard_tokens <= 0 or not toks:
        return text.strip()
    if n_heard_tokens >= len(toks):
        return ""
    cut_at = toks[n_heard_tokens].start()
    starts = [0] + [m.end() for m in _SENT_END.finditer(text)]
    start = max(s for s in starts if s <= cut_at)
    return text[start:].strip()
