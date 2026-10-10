"""Check that the ground truth can be read off what every arm sees.

A capable model of another family reads (a) the whole catalog and each task,
and names the procedure; (b) every rule's "applies to" list and each
procedure's signature and docstring, and names the rules that apply. Every
disagreement with the catalog is written out for review by hand.

    ../../bin/with-openrouter-key uv run python validate_data.py --ledger runs/ledger-laptop.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from demos_common import LLM

from linked_libraries import catalog
from linked_libraries.library import rule_body, signature
from linked_libraries.scoring import parse_json

HERE = Path(__file__).parent
MODEL = "openai/gpt-6.1-sol"


def catalog_text() -> str:
    parts = []
    for f, fd in catalog.FOLDERS.items():
        parts.append(f"### {f}: {fd['description']}")
        for p in catalog.PROCEDURES:
            if p["folder"] == f:
                parts.append(f"- `{signature(p['name'], p)}`: {p['doc']}")
        parts.append("")
    return "\n".join(parts)


def rules_text() -> str:
    return "\n\n".join(f"{rid} {r['title']}\n{rule_body(r)}" for rid, r in catalog.RULES.items())


async def main_async(ledger: str) -> None:
    out = {"model": MODEL, "procedure": [], "rules": []}
    procs = catalog.PROCEDURES
    async with LLM(ledger, "linked-libraries", max_concurrency=8) as llm:

        async def tasks_batch(batch):
            listing = "\n".join(f"{i}: {p['task']}" for i, p in enumerate(batch))
            prompt = (f"This is a company assistant's library of procedures, grouped by skill:\n\n{catalog_text()}\n\n"
                      f"For each task below, name the one procedure in the library that the task should call.\n\n{listing}\n\n"
                      'Answer with JSON only: {"0": "<procedure name>", "1": "...", ...}')
            r = await llm.chat(MODEL, [{"role": "user", "content": prompt}], tag="validate", meta={"kind": "procedure"}, max_tokens=8000)
            got = parse_json(r.text)
            return [(p, got.get(str(i))) for i, p in enumerate(batch)]

        async def rules_batch(batch):
            listing = "\n".join(f"- {p['name']} (skill {p['folder']}): `{signature(p['name'], p)}`. {p['doc']}" for p in batch)
            prompt = (f"These are a company's rules. Each says what kinds of work it applies to.\n\n{rules_text()}\n\n"
                      f"For each procedure below, list the id of every rule that applies to it: every rule whose "
                      f"'Applies to' covers the kind of work the procedure does. Many procedures fall under no rule.\n\n{listing}\n\n"
                      'Answer with JSON only: {"<procedure name>": ["R01", ...], ...}')
            r = await llm.chat(MODEL, [{"role": "user", "content": prompt}], tag="validate", meta={"kind": "rules"}, max_tokens=8000)
            got = parse_json(r.text)
            return [(p, got.get(p["name"])) for p in batch]

        tb = [procs[i:i + 12] for i in range(0, len(procs), 12)]
        rb = [procs[i:i + 16] for i in range(0, len(procs), 16)]
        res_t, res_r = await asyncio.gather(asyncio.gather(*(tasks_batch(b) for b in tb)),
                                            asyncio.gather(*(rules_batch(b) for b in rb)))
    agree_t = agree_r = 0
    for batch in res_t:
        for p, got in batch:
            ok = got == p["name"]
            agree_t += ok
            if not ok:
                out["procedure"].append({"procedure": p["name"], "task": p["task"], "model_says": got})
    for batch in res_r:
        for p, got in batch:
            got = sorted(got) if isinstance(got, list) else got
            ok = got == sorted(p["rules"])
            agree_r += ok
            if not ok:
                out["rules"].append({"procedure": p["name"], "doc": p["doc"], "catalog": sorted(p["rules"]), "model_says": got})
    out["summary"] = {"procedures": len(procs), "procedure_agree": agree_t, "rules_agree": agree_r}
    (HERE / "data" / "validation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(out["summary"]))
    for d in out["procedure"]:
        print("PROC", d)
    for d in out["rules"]:
        print("RULES", d["procedure"], "catalog", d["catalog"], "model", d["model_says"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(HERE / "runs" / "ledger.jsonl"))
    asyncio.run(main_async(ap.parse_args().ledger))


if __name__ == "__main__":
    main()
