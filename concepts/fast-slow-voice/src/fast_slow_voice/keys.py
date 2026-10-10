"""Was an answer heard? Is a line only a holding line? Shared by the simulator
(Sam's triggers) and the measures, so both use the same rules."""

from __future__ import annotations

import re

_AIRLINES = {"UNITED": "UA", "AMERICAN": "AA", "DELTA": "DL", "SOUTHWEST": "WN", "ALASKA": "AS"}
_LONG_SPAN = re.compile(r"\d[\d\s().,\-]*\d|\d")
_SHORT_RUN = re.compile(r"\d[\d.,\-]*\d|\d")


def key_in(text: str, key: str, key_type: str) -> bool:
    """Whether `key` (normalised) is said in `text`."""
    if not text:
        return False
    if key_type == "digits":
        if len(key) >= 7:
            return any(key in re.sub(r"\D", "", m.group()) for m in _LONG_SPAN.finditer(text))
        return any(re.sub(r"\D", "", m.group()) == key for m in _SHORT_RUN.finditer(text))
    if key_type == "code":
        up = text.upper()
        for name, code in _AIRLINES.items():
            up = re.sub(rf"\b{name}(\s+AIRLINES|\s+AIR\s+LINES)?(\s+FLIGHT)?\b", code, up)
        norm = re.sub(r"[\s\-]", "", up)
        return key.upper() in norm
    if key_type == "time":
        t = text.replace(".", ":")
        return re.search(rf"(?<!\d){re.escape(key)}(?!\d)", t) is not None
    if key_type == "email":
        t = text.lower()
        t = re.sub(r"\s+dot\s+", ".", t)
        t = re.sub(r"\s+at\s+", "@", t)
        t = re.sub(r"\s+(dash|hyphen)\s+", "-", t)
        return key.lower() in t or key.lower() in t.replace(" ", "")
    if key_type == "tokens":
        return re.search(rf"(?<![a-z0-9]){re.escape(key.lower())}(?![a-z0-9])", text.lower()) is not None
    raise ValueError(key_type)


def answer_in(text: str, keys: list[str], key_type: str) -> bool:
    """All of an item's keys said in one piece of text."""
    if key_type == "tokens":
        return all(key_in(text, k, "tokens") for k in keys)
    if key_type == "time":
        return any(key_in(text, k, "time") for k in keys)  # 3:20 or 15:20
    return bool(keys) and all(key_in(text, k, key_type) for k in keys)


_HOLD = re.compile(
    r"\b(one sec(ond)?|a sec(ond)?|one moment|a moment|just a moment|bear with me|hold on|hang on|"
    r"let me (check|look|see|find|pull|get|grab|dig)|checking|looking (that|it|this) up|pulling (that|it|this) up|"
    r"give me a (sec|second|moment)|on it|i'?ll (check|look|find|get|pull)|let me take a look|"
    r"mm|hmm|right away)\b", re.I)


def is_holding(text: str) -> bool:
    """A short line that holds the floor and gives no information."""
    t = text.strip()
    if not t:
        return True
    if any(ch.isdigit() for ch in t):
        return False
    return len(t.split()) <= 12 and _HOLD.search(t) is not None
