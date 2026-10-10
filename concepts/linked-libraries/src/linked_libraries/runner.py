"""Run the change phase, the state read and the task phase for one model and size.

For each change arm, condition and seed (a "chain"): build a fresh library,
apply the changes in order (one agent loop each), record the text that governs
each procedure afterwards, then run every task arm that reads this library on
every task, read-only. Chains run concurrently; one shared LLM client bounds
the number of calls in flight.

Records (JSON lines under runs/<run>/): changes, state, texts, tasks, chains
(a row when a chain's changes are complete). A chain whose changes did not all
complete is rebuilt from scratch, and its partial records are moved to
discarded/ with the reason.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .agent import run_loop
from .arms import CHANGE_ARMS, SOURCE, TASK_ARMS, library_path, system_prompt, toolbox
from .library import build_folders, build_linked, guidance_for, load_linked, save_linked
from .scoring import function_found

CONDITIONS = ("direct", "passing")
MAX_STEPS_CHANGE = 40
MAX_STEPS_TASK = 25


def model_slug(model: str) -> str:
    return model.replace("/", "_")


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_function(name: Any) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    n = name.strip().strip("`").removesuffix("()")
    n = n.split("(")[0]
    n = n.removeprefix("scripts/").removesuffix(".py")
    n = n.split("/")[-1].split(".")[-1]
    return n.strip()


class Records:
    """Append-only JSON lines files for one run, safe for one process."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.jsonl"

    def read(self, name: str) -> list[dict]:
        p = self.path(name)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    def write(self, name: str, row: dict) -> None:
        with self.path(name).open("a") as f:
            f.write(json.dumps({**row, "written_at": round(time.time(), 1)}, ensure_ascii=False) + "\n")

    def rewrite(self, name: str, rows: list[dict]) -> None:
        tmp = self.path(name).with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.replace(self.path(name))

    def discard(self, name: str, keep: Any, reason: str) -> int:
        """Move rows of `name` for which keep(row) is false to discarded/<name>.jsonl, with the reason."""
        rows = self.read(name)
        bad = [r for r in rows if not keep(r)]
        if bad:
            d = self.root / "discarded"
            d.mkdir(exist_ok=True)
            with (d / f"{name}.jsonl").open("a") as f:
                for r in bad:
                    f.write(json.dumps({**r, "discard_reason": reason}, ensure_ascii=False) + "\n")
            self.rewrite(name, [r for r in rows if keep(r)])
        return len(bad)


def chain_key(model: str, size: str, arm: str, cond: str, seed: int) -> str:
    return f"{model}|{size}|{arm}|{cond}|s{seed}"


def governing_texts(ds: dict, arm: str, lib: Path) -> dict[tuple[str, str], str]:
    """For every rule and procedure it governs (the original links), the text
    that governs the procedure after the changes."""
    out: dict[tuple[str, str], str] = {}
    if arm == "linked":
        g = load_linked(lib)
        for rid in ds["rules"]:
            for n, p in ds["procedures"].items():
                if rid in p["rules"]:
                    out[(rid, n)] = "\n\n".join(f"[{gid}] {e['title']}\n{e['content']}" for gid, e in guidance_for(g, n))
        return out
    if arm == "pasted":
        for rid in ds["rules"]:
            for n, p in ds["procedures"].items():
                if rid in p["rules"]:
                    f = lib / p["folder"] / "SKILL.md"
                    out[(rid, n)] = f.read_text() if f.exists() else ""
        return out
    if arm == "rule-skills":
        texts = []
        for f in sorted(lib.glob("**/*.md")):
            top = f.relative_to(lib).parts[0]
            if top not in ds["folders"]:
                texts.append(f"# file {f.relative_to(lib)}\n{f.read_text()}")
        whole = "\n\n".join(texts)
    else:  # rules-file: RULES.md and any other markdown file outside the procedure skills
        texts = []
        for f in sorted(lib.glob("**/*.md")):
            top = f.relative_to(lib).parts[0]
            if top not in ds["folders"]:
                texts.append(f"# file {f.relative_to(lib)}\n{f.read_text()}")
        whole = "\n\n".join(texts)
    for rid in ds["rules"]:
        for n, p in ds["procedures"].items():
            if rid in p["rules"]:
                out[(rid, n)] = whole
    return out


class Runner:
    def __init__(self, llm, records: Records, ds: dict, model: str, *, lib_root: Path,
                 reasoning: dict[str, Any] | None = None, max_tokens: int = 4000) -> None:
        self.llm = llm
        self.rec = records
        self.ds = ds
        self.model = model
        self.size = ds["size"]
        self.lib_root = lib_root
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.settings = {"reasoning": reasoning, "max_tokens": max_tokens}

    def lib_dir(self, arm: str, cond: str, seed: int) -> Path:
        return self.lib_root / model_slug(self.model) / self.size / arm / cond / f"s{seed}"

    def base(self, arm: str, cond: str, seed: int) -> dict:
        return {"model": self.model, "size": self.size, "arm": arm, "condition": cond, "seed": seed,
                "settings": self.settings}

    # ------------------------------------------------------------------ changes
    async def change_chain(self, arm: str, cond: str, seed: int) -> Path:
        key = chain_key(self.model, self.size, arm, cond, seed)
        lib = library_path(self.lib_dir(arm, cond, seed), arm)
        done = {r["chain"] for r in self.rec.read("chains")}
        if key in done:
            return lib
        n = self.rec.discard("changes", lambda r: r.get("chain") != key, "chain did not complete; rebuilt and re-run")
        if n:
            print(f"[{key}] discarded {n} partial change records", flush=True)
        self.rec.discard("state", lambda r: r.get("chain") != key, "chain did not complete; rebuilt and re-run")
        d = self.lib_dir(arm, cond, seed)
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        if arm == "linked":
            save_linked(build_linked(self.ds), lib)
        else:
            build_folders(self.ds, arm, lib)
        for ch in self.ds["changes"]:
            message = ch["direct_message"] if cond == "direct" else ch["passing_message"]
            system = system_prompt(arm, self.ds, lib)  # rebuilt each change: the rules file and listing may have changed
            tools = toolbox(arm, lib, write=True)
            res = await run_loop(self.llm, self.model, system, message, tools, max_steps=MAX_STEPS_CHANGE,
                                 tag=f"{arm}|change|{self.size}|{cond}",
                                 meta={"seed": seed, "change": ch["rule"]},
                                 max_tokens=self.max_tokens, reasoning=self.reasoning)
            if res["error"]:
                raise RuntimeError(f"[{key}] change {ch['rule']} failed: {res['error']}")
            self.rec.write("changes", {**self.base(arm, cond, seed), "chain": key, "change": ch["rule"],
                                       "order": ch["order"], "message": message,
                                       "passing_procedure": ch["passing_procedure"] if cond == "passing" else None,
                                       **res})
            print(f"[{key}] change {ch['rule']}: {res['n_tool_calls']} tool calls, {len(res['writes'])} writes, "
                  f"${res['cost']:.4f}", flush=True)
        self.write_state(arm, cond, seed, lib, key)
        self.rec.write("chains", {**self.base(arm, cond, seed), "chain": key})
        return lib

    def write_state(self, arm: str, cond: str, seed: int, lib: Path, key: str) -> None:
        changed = {c["rule"] for c in self.ds["changes"]}
        known = {r["hash"] for r in self.rec.read("texts")}
        for (rid, n), text in governing_texts(self.ds, arm, lib).items():
            h = hashlib.sha1(text.encode()).hexdigest()[:16]
            if h not in known:
                self.rec.write("texts", {"hash": h, "text": text})
                known.add(h)
            intact = norm_ws(self.ds["rules"][rid]["text"]) in norm_ws(text)
            self.rec.write("state", {**self.base(arm, cond, seed), "chain": key, "rule": rid, "procedure": n,
                                     "changed": rid in changed, "hash": h, "original_intact": intact})

    # ------------------------------------------------------------------ tasks
    async def task_one(self, arm: str, cond: str, seed: int, lib: Path, task: dict, done: set[str]) -> None:
        key = f"{chain_key(self.model, self.size, arm, cond, seed)}|{task['id']}"
        if key in done:
            return
        system = system_prompt(arm, self.ds, lib)
        tools = toolbox(arm, lib, write=False)
        res = await run_loop(self.llm, self.model, system, task["text"], tools, max_steps=MAX_STEPS_TASK,
                             tag=f"{arm}|task|{self.size}|{cond}", meta={"seed": seed, "task": task["id"]},
                             max_tokens=self.max_tokens, reasoning=self.reasoning)
        ans = res["answer"] or {}
        found = function_found(ans.get("function"), self.ds["procedures"], task["procedure"])
        self.rec.write("tasks", {**self.base(arm, cond, seed), "key": key, "task": task["id"],
                                 "procedure": task["procedure"], "rules_expected": task["rules"],
                                 "changed_expected": task["changed_rules"], "hidden": task["hidden"],
                                 "function_found": found, **res})

    async def chain(self, source: str, cond: str, seed: int, task_arms: list[str]) -> None:
        lib = await self.change_chain(source, cond, seed)
        done = {r["key"] for r in self.rec.read("tasks")}
        jobs = []
        for arm in task_arms:
            if SOURCE[arm] != source:
                continue
            arm_lib = lib
            for t in self.ds["tasks"]:
                jobs.append(self.task_one(arm, cond, seed, arm_lib, t, done))
        await asyncio.gather(*jobs)
        print(f"[{chain_key(self.model, self.size, source, cond, seed)}] tasks done for "
              f"{[a for a in task_arms if SOURCE[a] == source]}", flush=True)


async def run(llm, records: Records, ds: dict, model: str, *, arms: list[str], conditions: list[str],
              seeds: list[int], lib_root: Path, reasoning: dict[str, Any] | None = None,
              max_tokens: int = 4000, tasks: list[str] | None = None) -> None:
    unknown = [a for a in arms if a not in TASK_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    if tasks:
        missing = set(tasks) - {t["id"] for t in ds["tasks"]}
        if missing:
            raise ValueError(f"unknown tasks {sorted(missing)}")
        ds = {**ds, "tasks": [t for t in ds["tasks"] if t["id"] in set(tasks)]}
    # a task record that failed on an API error is discarded and run again
    records.discard("tasks", lambda r: not r.get("error"), "API error in the loop; re-run")
    runner = Runner(llm, records, ds, model, lib_root=lib_root, reasoning=reasoning, max_tokens=max_tokens)
    sources = sorted({SOURCE[a] for a in arms}, key=CHANGE_ARMS.index)
    chains = [(src, cond, seed) for src in sources for cond in conditions for seed in seeds]
    results = await asyncio.gather(*(runner.chain(src, cond, seed, arms) for src, cond, seed in chains),
                                   return_exceptions=True)
    failed = [(c, r) for c, r in zip(chains, results) if isinstance(r, BaseException)]
    for c, r in failed:
        print(f"chain {c} failed: {type(r).__name__}: {r}", flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} chains failed; re-run to resume")
