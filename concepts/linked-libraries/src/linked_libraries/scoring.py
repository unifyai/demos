"""Scoring: which rule each reported item is, and which version of a changed
rule a text states.

Matching a reported rule to a rule of the dataset is by title: exact after
normalising, then one rule title contained in the other, then a close fuzzy
match; what is still unmatched goes to the judge. A changed rule's version is
judged from the rule's old and new fact.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from typing import Any

JUDGE_MODEL = "google/gemini-3.8-flash"

VERSION_PROMPT = """A company changed one of its rules.

Rule: {title}
Before the change: {old}
After the change: {new}

Below is a text that may state this rule, possibly among other rules. Ignore the other rules, and ignore the parts of this rule that did not change.

TEXT:
\"\"\"
{text}
\"\"\"

Answer with JSON only: {{"states_new": true or false, "states_old": true or false}}
- states_new: the text states this rule in its changed form ({new}).
- states_old: the text still states the old form ({old}) somewhere, so that a reader could follow the old rule."""

MAP_PROMPT = """These are the rules in a company's library, by id and title:
{titles}

An assistant reported this rule:
Title: {title}
Text: {text}

Which of the rules above is it? Answer with JSON only: {{"rule": "<the id>"}}, or {{"rule": null}} if it is none of them."""


def named_procedures(function: Any, procedures: dict) -> set[str]:
    """The library's procedures that an answer's "function" names, as whole identifiers
    (so 'billing/scripts/create_invoice.py' and 'create_invoice()' both name create_invoice)."""
    if not isinstance(function, str):
        return set()
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", function)) & set(procedures)


def function_found(function: Any, procedures: dict, target: str) -> bool:
    return named_procedures(function, procedures) == {target}


def h(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def norm_title(t: Any) -> str:
    t = str(t or "").lower()
    t = re.sub(r"^\s*(the\s+)?rule\s*(on|about|:)\s*", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def reported_items(answer: dict | None) -> list[dict[str, str]]:
    """The answer's rules as [{title, text}], tolerating strings and other shapes."""
    rules = (answer or {}).get("rules") or []
    if isinstance(rules, (str, dict)):
        rules = [rules]
    out = []
    for r in rules:
        if isinstance(r, dict):
            out.append({"title": str(r.get("title") or r.get("name") or ""), "text": str(r.get("text") or r.get("rule") or r.get("content") or "")})
        elif isinstance(r, str):
            out.append({"title": r.split(":")[0] if ":" in r[:80] else "", "text": r})
    return out


def match_title(title: str, rules: dict[str, dict]) -> str | None:
    """The rule id a reported title names, or None if it is not clear."""
    t = norm_title(title)
    if not t:
        return None
    titles = {rid: norm_title(r["title"]) for rid, r in rules.items()}
    exact = [rid for rid, rt in titles.items() if rt == t]
    if len(exact) == 1:
        return exact[0]
    contained = [rid for rid, rt in titles.items() if len(rt) >= 8 and (rt in t or (len(t) >= 8 and t in rt))]
    if len(contained) == 1:
        return contained[0]
    best, score = None, 0.0
    for rid, rt in titles.items():
        s = difflib.SequenceMatcher(None, rt, t).ratio()
        if s > score:
            best, score = rid, s
    return best if score >= 0.85 else None


def classify(states_new: bool, states_old: bool) -> str:
    if states_new and not states_old:
        return "current"
    if states_old and not states_new:
        return "stale"
    if states_new and states_old:
        return "conflict"
    return "vague"


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def version_key(rid: str, text: str) -> str:
    return f"version|{rid}|{h(text)}"


def map_key(size: str, item: dict) -> str:
    body = item["title"] + "\n" + item["text"]
    return f"map|{size}|{h(body)}"


def joined(items: list[dict]) -> str:
    return "\n".join(f"- {i['title']}: {i['text']}" for i in items)


def score_task(rec: dict, ds: dict, judgments: dict[str, dict]) -> dict:
    """Score one task record against its expected rules, given the judgments
    (missing judgments leave the rule 'unjudged')."""
    rules = ds["rules"]
    changed = {c["rule"] for c in ds["changes"]}
    items = reported_items(rec.get("answer"))
    mapped: dict[str, list[dict]] = {}
    unknown = 0
    for it in items:
        rid = match_title(it["title"], rules)
        if rid is None:
            j = judgments.get(map_key(ds["size"], it))
            rid = (j or {}).get("rule") if j else None
            if rid not in rules:
                rid = None
        if rid is None:
            unknown += 1
            continue
        mapped.setdefault(rid, []).append(it)
    per_rule = {}
    for rid in rec["rules_expected"]:
        got = mapped.get(rid)
        if not got:
            per_rule[rid] = "missing"
        elif rid in changed:
            j = judgments.get(version_key(rid, joined(got)))
            per_rule[rid] = classify(j["states_new"], j["states_old"]) if j else "unjudged"
        else:
            per_rule[rid] = "current"
    extras = sorted(set(mapped) - set(rec["rules_expected"]))
    all_current = all(v == "current" for v in per_rule.values())
    found = function_found((rec.get("answer") or {}).get("function"), ds["procedures"], rec["procedure"]) \
        if "procedure" in rec else bool(rec.get("function_found"))
    return {"rules": per_rule, "extras": extras, "unknown_items": unknown, "n_items": len(items),
            "function_found": found, "fully_correct": found and all_current,
            "exact": found and all_current and not extras and not unknown}


def judgment_jobs_for_task(rec: dict, ds: dict) -> list[tuple[str, str, dict]]:
    """(key, kind, payload) of the judgments a task record needs, mapping first."""
    rules = ds["rules"]
    jobs = []
    for it in reported_items(rec.get("answer")):
        if match_title(it["title"], rules) is None:
            jobs.append((map_key(ds["size"], it), "map", {"size": ds["size"], "item": it}))
    return jobs


def version_jobs_for_task(rec: dict, ds: dict, judgments: dict[str, dict]) -> list[tuple[str, str, dict]]:
    rules = ds["rules"]
    changed = {c["rule"] for c in ds["changes"]}
    mapped: dict[str, list[dict]] = {}
    for it in reported_items(rec.get("answer")):
        rid = match_title(it["title"], rules)
        if rid is None:
            rid = (judgments.get(map_key(ds["size"], it)) or {}).get("rule")
        if rid in rules:
            mapped.setdefault(rid, []).append(it)
    jobs = []
    for rid in rec["rules_expected"]:
        if rid in changed and rid in mapped:
            text = joined(mapped[rid])
            jobs.append((version_key(rid, text), "version", {"rule": rid, "text": text}))
    return jobs


def version_prompt(rule: dict, text: str) -> str:
    ch = rule["change"]
    return VERSION_PROMPT.format(title=rule["title"], old=ch["old"], new=ch["new"], text=text.strip() or "(empty)")


def map_prompt(ds: dict, item: dict) -> str:
    titles = "\n".join(f"{rid}: {r['title']}" for rid, r in ds["rules"].items())
    return MAP_PROMPT.format(titles=titles, title=item["title"], text=item["text"])
