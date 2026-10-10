"""Run calls: every (arm, script, seed) of a condition, resumable.

  ../../bin/with-openrouter-key uv run python run.py --size realistic --pair anthropic --delay 0 \
      --arms relay,slow_speaks,slow_only,fast_only,fast_context,slow_filler --scripts 16 --seeds 0,1,2

Records go to runs/<condition>.jsonl (one line per call, with the measures),
failures to runs/<condition>.errors.jsonl. A (arm, script, seed) already
recorded is skipped, so a re-run continues where it stopped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import traceback
from pathlib import Path

from demos_common import LLM

from fast_slow_voice.arms import Models, make_arm
from fast_slow_voice.config import PAIRS, PRESETS
from fast_slow_voice.metrics import call_metrics
from fast_slow_voice.script import Script, build_sanity, generate, scripts_hash
from fast_slow_voice.sim import Sim
from fast_slow_voice.timing import Timing

HERE = Path(__file__).parent
BASE_SEED = 1


def load_scripts(n: int, sanity: bool) -> list[Script]:
    if sanity:
        return [build_sanity(f"sanity-{k}", 900 + k) for k in range(n)]
    path = HERE / "data" / f"scripts-{BASE_SEED}.json"
    if path.exists():
        scripts = [Script.from_json(d) for d in json.loads(path.read_text())]
    else:
        scripts = generate(BASE_SEED, 16)
        path.write_text(json.dumps([s.to_json() for s in scripts], indent=0, default=list))
        (HERE / "data" / f"scripts-{BASE_SEED}.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return scripts[:n]


def condition_name(a: argparse.Namespace) -> str:
    if a.name:
        return a.name
    base = f"{a.pair}-{'sanity' if a.sanity else a.size}-d{a.delay:g}"
    return base


def done_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r["arm"], r["script"], r["seed"]))
    return out


async def one(llm: LLM, arm_name: str, script: Script, size: str, seed: int, models: Models, timing: Timing,
              cond: str, out: Path, err: Path, lock: asyncio.Lock) -> None:
    world = script.world(size)
    meta = {"condition": cond, "arm": arm_name, "script": script.id, "seed": seed, "size": size}
    arm = make_arm(arm_name, models)
    sim = Sim(script, world, arm, timing, llm, run_seed=seed, tag=f"fast-slow-voice/{cond}/{arm_name}", meta=meta)
    t0 = time.time()
    try:
        await sim.run()
        rec = sim.record()
        rec.update({"fast_model": models.fast.name, "slow_model": models.slow.name,
                    "models": {"fast": models.fast.__dict__, "slow": models.slow.__dict__, "fast_think": models.fast_think.__dict__},
                    "wall_s": round(time.time() - t0, 1),
                    "world": world.fingerprint(), "delay": timing.delay})
        rec["metrics"] = call_metrics(rec, script, world)
        if hasattr(arm, "parse_failures"):
            rec["metrics"]["label_parse_failures"] = arm.parse_failures
            rec["metrics"]["replay_fallbacks"] = arm.replay_fallbacks
        async with lock:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        m = rec["metrics"]
        print(f"{cond} {arm_name:13s} {script.id} s{seed}: correct {m['n_correct']}/{m['n_wanted']} stale {m['n_stale']} "
              f"invented {len(m['invented'])} cost ${m['cost']:.3f} wall {rec['wall_s']}s flags {m['flags']}", flush=True)
    except Exception as e:  # noqa: BLE001
        async with lock:
            with open(err, "a") as f:
                f.write(json.dumps({**meta, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:],
                                    "t": time.time()}) + "\n")
        print(f"ERROR {cond} {arm_name} {script.id} s{seed}: {type(e).__name__}: {e}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="relay,slow_speaks,slow_only,fast_only,fast_context")
    ap.add_argument("--size", default="realistic")
    ap.add_argument("--pair", default="anthropic", choices=sorted(PAIRS))
    ap.add_argument("--delay", type=float, default=0.0, help="seconds added to the slow model's first token")
    ap.add_argument("--scripts", type=int, default=16)
    ap.add_argument("--first-script", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--concurrency", type=int, default=12, help="model calls in flight (shared key: 12 or less)")
    ap.add_argument("--parallel", type=int, default=16, help="calls simulated at once")
    ap.add_argument("--ledger", default="runs/ledger.jsonl", help="runs/ledger-vm.jsonl on the cloud machine")
    a = ap.parse_args()
    cond = condition_name(a)
    size = "sanity" if a.sanity else a.size
    scripts = load_scripts(a.first_script + a.scripts, a.sanity)[a.first_script:]
    fast, slow, think = PAIRS[a.pair]
    models = Models(PRESETS[fast], PRESETS[slow], PRESETS[think])
    timing = Timing.load(HERE / "data" / "timing.json")
    if a.delay:
        timing.delay[models.slow.name] = a.delay
    out = HERE / "runs" / f"{cond}.jsonl"
    err = HERE / "runs" / f"{cond}.errors.jsonl"
    done = done_keys(out)
    jobs = [(arm, s, seed) for seed in [int(x) for x in a.seeds.split(",")] for s in scripts for arm in a.arms.split(",")
            if (arm, s.id, seed) not in done]
    print(f"{cond}: {len(jobs)} calls to run ({len(done)} already recorded); scripts {scripts_hash(scripts)}", flush=True)
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(a.parallel)
    async with LLM(HERE / a.ledger, f"fast-slow-voice/{cond}", max_concurrency=min(12, a.concurrency)) as llm:
        async def guarded(job):
            async with sem:
                await one(llm, job[0], job[1], size, job[2], models, timing, cond, out, err, lock)
        await asyncio.gather(*(guarded(j) for j in jobs))


if __name__ == "__main__":
    asyncio.run(main())
