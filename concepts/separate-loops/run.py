"""separate-loops: generate data, run batches, read transcripts, evaluate.

    uv run python run.py gen
    ../../bin/with-openrouter-key uv run python run.py run --condition realistic --batch realistic-haiku \\
        --model anthropic/claude-haiku-5.5 --parallel 6 --max-concurrency 6
    uv run python run.py show --batch realistic-haiku --run 'realistic-haiku|realistic-A-s0|handle'
    uv run python run.py eval --batch realistic-haiku

A batch's records go to runs/<batch>/records.jsonl, one JSON line per run, and
every model call to runs/ledger.jsonl. A batch can be re-run: runs already
recorded are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

from demos_common import LLM  # noqa: E402

from separate_loops.checks import evaluate  # noqa: E402
from separate_loops.scenarios import SEEDS, generate  # noqa: E402

ARMS = ["flat", "blocking", "background", "handle"]
CONDITIONS = ["sanity", "small", "medium", "realistic", "dual"]


def load_instances(condition: str) -> dict[str, dict]:
    path = HERE / "data" / f"{condition}.json"
    return {i["id"]: i for i in json.loads(path.read_text())}


def records(batch: str) -> list[dict]:
    p = HERE / "runs" / batch / "records.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def cmd_gen(args: argparse.Namespace) -> None:
    (HERE / "data").mkdir(exist_ok=True)
    for c in CONDITIONS:
        insts = generate(c)
        (HERE / "data" / f"{c}.json").write_text(json.dumps(insts, indent=1))
        print(f"data/{c}.json: {len(insts)} instances (seeds {SEEDS[c]})")


async def _run_batch(args: argparse.Namespace) -> None:
    from separate_loops.runner import run_one

    insts = load_instances(args.condition)
    fams = set(args.families.split(","))
    seeds = None if args.seeds is None else {int(s) for s in args.seeds.split(",")}
    arms = args.arms.split(",")
    out_dir = HERE / "runs" / args.batch
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "records.jsonl"
    done = {r["run_id"] for r in records(args.batch)}
    jobs = []
    for inst in insts.values():
        if inst["family"] not in fams or (seeds is not None and inst["seed"] not in seeds):
            continue
        for arm in arms:
            rid = f"{args.batch}|{inst['id']}|{arm}"
            if rid not in done:
                jobs.append((rid, inst, arm))
    print(f"{len(jobs)} runs to do ({len(done)} already recorded) in {args.batch}", flush=True)
    sem = asyncio.Semaphore(args.parallel)
    t0 = time.time()
    async with LLM(HERE / args.ledger, "separate-loops", max_concurrency=args.max_concurrency) as llm:
        def spent() -> float:
            total = args.spent_elsewhere
            for path in args.budget_ledgers.split(","):
                f = HERE / path
                if f.exists():
                    total += sum(json.loads(x).get("cost") or 0.0 for x in f.read_text().splitlines() if x.strip())
            return total

        async def one(rid: str, inst: dict, arm: str) -> None:
            async with sem:
                if args.stop_at and spent() >= args.stop_at:
                    print(f"[budget] not starting {rid}: ${spent():.2f} spent, stop at ${args.stop_at}", flush=True)
                    return
                rec = await run_one(inst, arm, args.model, llm, run_id=rid, cap_s=args.cap, reasoning=args.reasoning)
                with open(out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                ev = evaluate(rec, inst)
                print(f"[{time.time() - t0:7.0f}s] {rid}: {rec['end']['reason']} at {rec['end']['t']:.0f}s, "
                      f"correct={ev['correct']} {ev['checks']} stale={ev['stale_total']} cost=${ev['cost']:.4f} "
                      f"errors={ev['errors']}", flush=True)
        await asyncio.gather(*(one(*j) for j in jobs))


def cmd_run(args: argparse.Namespace) -> None:
    asyncio.run(_run_batch(args))


def _fmt(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def cmd_show(args: argparse.Namespace) -> None:
    rec = next(r for r in records(args.batch) if r["run_id"] == args.run or r["run_id"].endswith(args.run))
    inst = load_instances(rec["condition"])[rec["instance"]]
    ev = []
    for m in rec["person"]:
        ev.append((m["t"], f"PERSON [{m['kind']}{'/' + m['how'] if m['how'] else ''}] {m['text']}"))
    for m in rec["agent"]:
        ev.append((m["t"], f"AGENT  ({m['loop']}, seen {m['seen']}) {m['text']}"))
    for a in rec["actions"]:
        if a["t_start"] is None:
            continue
        res = json.dumps(a["result"])[:args.width] if a["result"] is not None else "(cancelled)"
        ev.append((a["t_start"], f"   act {a['worker']:>4} {a['tool']}({json.dumps(a['args'])[:args.width]}) -> {res}"))
    for c in rec["calls"]:
        if args.calls and c["loop"] != "person":
            ev.append((c["t0"], f"      call {c['loop']} {c['t1'] - c['t0']:.1f}s tools={c['tools']} "
                                f"{'DISCARDED ' if c['discarded'] else ''}{(c['text'] or '')[:args.width]}"))
    for t in rec["tasks"]:
        ev.append((t["t_start"], f" TASK {t['id']} started: {t['description'][:args.width * 2]}"))
        for i in t["interjections"]:
            ev.append((i["t"], f" TASK {t['id']} <- {i['text']}"))
        for c in t["controls"]:
            ev.append((c["t"], f" TASK {t['id']} {c['op']}: {c['result']}"))
        for q in t["questions"]:
            ev.append((q["t"], f" TASK {t['id']} asks: {q['text']}"))
        if t["t_end"] is not None:
            ev.append((t["t_end"], f" TASK {t['id']} {t['state']}: {t['result']}"))
    for t, line in sorted(ev, key=lambda x: x[0]):
        print(f"{_fmt(t)} {line}")
    print(f"END {rec['end']}")
    if rec["errors"]:
        print("ERRORS:", *rec["errors"], sep="\n")
    r = evaluate(rec, inst)
    print(json.dumps({k: v for k, v in r.items() if k != "messages"}, indent=1))
    for m in r["messages"]:
        print(m)


def judged(batch: str) -> dict[str, dict]:
    p = HERE / "runs" / batch / "judge.jsonl"
    if not p.exists():
        return {}
    return {j["run_id"]: j for j in (json.loads(x) for x in p.read_text().splitlines() if x.strip())}


async def _judge_batch(args: argparse.Namespace) -> None:
    from separate_loops.judge import judge_run

    done = judged(args.batch)
    todo = [r for r in records(args.batch) if r["run_id"] not in done]
    print(f"{len(todo)} runs to judge in {args.batch}", flush=True)
    sem = asyncio.Semaphore(args.parallel)
    out = HERE / "runs" / args.batch / "judge.jsonl"
    async with LLM(HERE / args.ledger, "separate-loops", max_concurrency=args.parallel) as llm:
        async def one(rec: dict) -> None:
            async with sem:
                inst = load_instances(rec["condition"])[rec["instance"]]
                j = await judge_run(rec, inst, llm)
                with open(out, "a") as f:
                    f.write(json.dumps(j) + "\n")
        await asyncio.gather(*(one(r) for r in todo))


def cmd_judge(args: argparse.Namespace) -> None:
    asyncio.run(_judge_batch(args))


def cmd_judgecheck(args: argparse.Namespace) -> None:
    """Print a seeded sample of the judge's verdicts with the evidence it saw, to check by hand."""
    import random

    from separate_loops.judge import action_log, state_summary

    recs = {r["run_id"]: r for r in records(args.batch)}
    js = judged(args.batch)
    items = [(rid, qid) for rid, j in js.items() for qid in j["questions"]] + [(rid, "outcome") for rid in js]
    rng = random.Random(args.seed)
    for rid, qid in rng.sample(items, min(args.n, len(items))):
        rec = recs[rid]
        inst = load_instances(rec["condition"])[rec["instance"]]
        sites = {inst["main"]["site"]} | ({inst["second"]["site"]} if inst.get("second") else set())
        print("=" * 100)
        if qid == "outcome":
            o = js[rid]["outcome"]
            print(f"{rid} OUTCOME judge: reported={o['reported']} accurate={o['accurate']} | {o['notes']}")
            print(state_summary(rec, inst)[:1500])
            last = max((a["t_end"] or a["t_start"] for a in rec["actions"] if a["site"] in sites and a["t_start"]), default=0)
            for m in [m for m in rec["agent"] if m["t"] >= last] or rec["agent"][-3:]:
                print(f"  AGENT {_fmt(m['t'])} {m['text'][:600]}")
            continue
        q = next(m for m in rec["person"] if str(m["id"]) == qid)
        v = js[rid]["questions"][qid]
        after = [m for m in rec["agent"] if q["t"] <= m["t"] <= q["t"] + 900][:6]
        print(f"{rid} Q{qid} at {_fmt(q['t'])}: {q['text']}")
        print(f"  judge: first_answer={v['first_answer']} first_accurate={v['first_accurate']} | {v['notes']}")
        log = action_log(rec, sites, after[-1]["t"] if after else q["t"]).splitlines()
        for line in log[-args.log:]:
            print("   ", line[:170])
        for i, m in enumerate(after):
            print(f"  {i + 1}. AGENT {_fmt(m['t'])} {m['text'][:500]}")


def cmd_eval(args: argparse.Namespace) -> None:
    rows = []
    for rec in records(args.batch):
        inst = load_instances(rec["condition"])[rec["instance"]]
        rows.append(evaluate(rec, inst))
    by = {}
    for r in rows:
        by.setdefault(r["arm"], []).append(r)
    for arm in ARMS:
        rs = by.get(arm, [])
        if not rs:
            continue
        qs = [m for r in rs for m in r["messages"] if m["kind"] in ("status", "status2", "status_both", "unrelated")
              and m["while_working"]]
        rep = sorted(m["reply_s"] for m in qs if m["reply_s"] is not None)
        med = rep[len(rep) // 2] if rep else None
        print(f"{arm:>10}: runs {len(rs)}, correct {sum(r['correct'] for r in rs)}, complete {sum(r['complete'] for r in rs)}, "
              f"stale {sum(r['stale_total'] for r in rs)}, reported_auto {sum(r['reported_auto'] for r in rs)}, "
              f"reply median {med}s (n={len(rep)}/{len(qs)}), cost ${sum(r['cost'] for r in rs):.3f}, "
              f"end {[r['end_reason'] for r in rs]}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen")
    r = sub.add_parser("run")
    r.add_argument("--condition", required=True, choices=CONDITIONS)
    r.add_argument("--batch", required=True)
    r.add_argument("--model", default="anthropic/claude-haiku-5.5")
    r.add_argument("--arms", default=",".join(ARMS))
    r.add_argument("--families", default="A,B,C")
    r.add_argument("--seeds", default=None)
    r.add_argument("--parallel", type=int, default=6)
    r.add_argument("--max-concurrency", type=int, default=6)
    r.add_argument("--cap", type=float, default=None)
    r.add_argument("--ledger", default="runs/ledger.jsonl", help="runs/ledger-vm.jsonl on the cloud machine")
    r.add_argument("--reasoning", default=None, help="reasoning effort for every agent call (top tier)")
    r.add_argument("--stop-at", type=float, default=None, help="start no run once this much has been spent, in $")
    r.add_argument("--budget-ledgers", default="runs/ledger.jsonl,runs/ledger-vm.jsonl")
    r.add_argument("--spent-elsewhere", type=float, default=0.0, help="spend in ledgers not on this machine, in $")
    s = sub.add_parser("show")
    s.add_argument("--batch", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--width", type=int, default=160)
    s.add_argument("--calls", action="store_true")
    j = sub.add_parser("judge")
    j.add_argument("--batch", required=True)
    j.add_argument("--parallel", type=int, default=8)
    j.add_argument("--ledger", default="runs/ledger.jsonl")
    jc = sub.add_parser("judgecheck")
    jc.add_argument("--batch", required=True)
    jc.add_argument("--n", type=int, default=10)
    jc.add_argument("--seed", type=int, default=0)
    jc.add_argument("--log", type=int, default=25)
    e = sub.add_parser("eval")
    e.add_argument("--batch", required=True)
    e.add_argument("--out", default=None)
    a = ap.parse_args()
    {"gen": cmd_gen, "run": cmd_run, "show": cmd_show, "judge": cmd_judge, "judgecheck": cmd_judgecheck, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
