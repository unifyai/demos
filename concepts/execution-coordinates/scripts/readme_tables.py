"""Print the README's results tables from results.json (no number is typed by hand).

    uv run python scripts/readme_tables.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ARMS = ["fresh", "shared", "per_task", "coordinate", "long_shared", "long_coordinate", "per_task_modes",
        "per_task@initials", "coordinate@initials"]
NAMES = {"anthropic/claude-haiku-5.5": "Claude Haiku 5.5", "openai/gpt-6-luna": "GPT-6 luna",
         "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5", "anthropic/claude-opus-5.5": "Claude Opus 5.5 (high)"}


def pct(x):
    return f"{100 * x:.1f}"


TIERS = [("cheap", "anthropic/claude-haiku-5.5"), ("cheap", "openai/gpt-6-luna"),
         ("mid", "anthropic/claude-sonnet-5.5"), ("top", "anthropic/claude-opus-5.5")]


def tiers(res):
    """One row per model and size: each arm's accuracy and state-caused wrong answers, and the paired
    differences that decide the idea."""
    groups, comps = res["groups"], res["comparisons"]
    print("\n**By tier** (accuracy %, with the number of wrong answers caused by another client's state in "
          "brackets; then coordinate minus per_task and coordinate minus shared, in points of accuracy with "
          "95% intervals; loads of the export per client for fresh / shared / per_task / coordinate)\n")
    print("| tier | model | K | clients | fresh | shared | per_task | coordinate | long_shared | long_coordinate | "
          "coordinate - per_task | coordinate - shared | loads per client |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tier, m in TIERS:
        for k in (1, 2, 4, 8):
            row = []
            gs = {a: groups.get(f"{m}|{a}|k{k}") for a in ARMS[:6]}
            if not gs.get("coordinate"):
                continue
            for a in ARMS[:6]:
                g = gs[a]
                row.append("-" if not g else f"{pct(g['accuracy'])} ({g['wrong_by_cause']['leak']})")
            def diff(name):
                c = comps.get(f"{m}|k{k}|{name}")
                if not c:
                    return "-"
                a = c["accuracy"]
                return f"{100 * a['diff']:+.1f} [{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}]"
            loads = " / ".join("-" if not gs[a] else f"{gs[a]['loads_per_client']:.1f}"
                               for a in ("fresh", "shared", "per_task", "coordinate"))
            n = gs["coordinate"]["requests"] // 6
            print(f"| {tier} | {NAMES.get(m, m)} | {k} | {n} | " + " | ".join(row) +
                  f" | {diff('coordinate-per_task')} | {diff('coordinate-shared')} | {loads} |")


def main():
    import sys

    res = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json").read_text())
    tiers(res)
    groups = res["groups"]
    models = []
    for key in groups:
        m = key.split("|")[0]
        if m not in models:
            models.append(m)
    for m in models:
        print(f"\n**{NAMES.get(m, m)}** (requests correct, accuracy with 95% Wilson interval; wrong answers caused by "
              "state, all of them leaks from another client unless marked; loads of the export per client; agent "
              "minutes per run; cost per request; prompt tokens per request)\n")
        print("| K | arm | correct | accuracy % [95% CI] | wrong from state | other wrong | loads per client | "
              "agent min per run | $ per 1,000 requests | prompt tokens per request |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for k in (1, 2, 4, 8):
            for arm in ARMS:
                g = groups.get(f"{m}|{arm}|k{k}")
                if not g:
                    continue
                w = g["wrong_by_cause"]
                state = f"{g['state_caused_wrong']}" + (f" ({w['own_whatif']} own what-if)" if w["own_whatif"] else "")
                print(f"| {k} | {arm} | {g['correct']}/{g['requests']} | {pct(g['accuracy'])} "
                      f"[{pct(g['accuracy_ci'][0])}, {pct(g['accuracy_ci'][1])}] | {state} | {w['other']} | "
                      f"{g['loads_per_client']:.2f} | {g['agent_minutes_per_run']['mean']:.1f} | "
                      f"{1000 * g['cost_per_request']:.2f} | {g['prompt_tokens_per_request'] / 1000:.1f}k |")
    print("\n**Paired differences** (second arm minus first; bootstrap over clients, 95% interval)\n")
    print("| model | K | difference | accuracy (points) | wrong from state (points) | reloads per client | "
          "agent s per request | $ per 1,000 requests |")
    print("|---|---|---|---|---|---|---|---|")
    want = ["coordinate-per_task", "coordinate-fresh", "coordinate-shared", "per_task-shared", "long_shared-shared",
            "long_coordinate-long_shared", "long_coordinate-per_task", "per_task_modes-per_task",
            "coordinate@initials-per_task@initials"]
    for key, c in res["comparisons"].items():
        m, k, pair = key.split("|")
        if pair not in want:
            continue
        a, s, rl, t, co = (c["accuracy"], c["state_caused_rate"], c["reloads_per_client"], c["agent_s_per_request"],
                           c["cost_per_request"])
        b, base = pair.split("-")
        print(f"| {NAMES.get(m, m)} | {k[1:]} | {b} vs {base} | {100 * a['diff']:+.1f} [{100 * a['lo']:+.1f}, "
              f"{100 * a['hi']:+.1f}] | {100 * s['diff']:+.1f} [{100 * s['lo']:+.1f}, {100 * s['hi']:+.1f}] | "
              f"{rl['diff']:+.2f} [{rl['lo']:+.2f}, {rl['hi']:+.2f}] | {t['diff']:+.1f} [{t['lo']:+.1f}, "
              f"{t['hi']:+.1f}] | {1000 * co['diff']:+.2f} [{1000 * co['lo']:+.2f}, {1000 * co['hi']:+.2f}] |")
    print("\n**What the coordinate arms chose**\n")
    print("| model | arm | K | code calls by mode | listing / inspection calls | what-if calls by mode | "
          "sessions per client | sessions shared by two clients |")
    print("|---|---|---|---|---|---|---|---|")
    for key, g in groups.items():
        if "choices" not in g:
            continue
        m, arm, k = key.split("|")
        c = g["choices"]
        modes = ", ".join(f"{v} {kk}" for kk, v in sorted(c["run_python_by_mode"].items(), key=lambda x: -x[1]))
        wm = ", ".join(f"{v} {kk}" for kk, v in sorted(c["whatif_calls_by_mode"].items(), key=lambda x: -x[1]))
        lists = c["tool_calls"].get("list_sessions", 0)
        insp = c["tool_calls"].get("inspect_session", 0)
        print(f"| {NAMES.get(m, m)} | {arm} | {k[1:]} | {modes} | {lists} / {insp} | {wm} | "
              f"{c['mean_sessions_per_client']:.2f} | {c['sessions_used_by_more_than_one_client']} |")
    decisions_table(res)
    blocks = res.get("top_tier_seed_blocks") or {}
    if blocks:
        print("\n**Top tier at K = 4 by block of seeds** (correct / requests; $ per 1,000 requests; agent seconds "
              "per request)\n")
        arms = [a for a in ARMS[:7] if any(a in b for b in blocks.values())]
        print("| seeds | " + " | ".join(arms) + " |")
        print("|---|" + "---|" * len(arms))
        for name, b in blocks.items():
            cells = [("-" if a not in b else f"{b[a]['correct']}/{b[a]['requests']}; "
                      f"{1000 * b[a]['cost_per_request']:.1f}; {b[a]['agent_s_per_request']:.1f}") for a in arms]
            print(f"| {name.replace('seeds_', '').replace('_', ' to ')} | " + " | ".join(cells) + " |")
    led = res.get("ledger") or {}
    if led:
        print("\n**Cost from the ledgers**\n")
        for k, d in sorted(led["by_model"].items()):
            m = k.split(" | ")[-1]
            old = " (superseded, out of the analysis)" if m in ("anthropic/claude-opus-4.5", "openai/gpt-5.2") else ""
            print(f"- {m}{old}: {int(d.get('calls', 0)):,} calls, {int(d.get('errors', 0))} errors, "
                  f"${d.get('cost', 0):.2f}")
        print(f"- today's models: ${led['todays_models']:.2f}; superseded January-2026 models: "
              f"${led['superseded_models']:.2f}; total: ${led['total']:.2f}")
    full = res.get("full_design_cost") or {}
    if full:
        print("\n**What the cheap tier's full design would cost on the mid and top tiers** (the six main arms at "
              "K = 1, 2, 4, 8 and per_task_modes at K = 4, ten seeds; from the measured cost per request)\n")
        print("| model | requests run | cost of those | full design requests | full design, estimated | the rest |")
        print("|---|---|---|---|---|---|")
        for m, d in full.items():
            print(f"| {NAMES.get(m, m)} | {d['requests_run']:,} | ${d['cost_run']:.2f} | {d['full_requests']:,} | "
                  f"${d['full_cost_estimate']:.0f} | ${d['rest_cost_estimate']:.0f} |")


def decisions_table(res):
    dec = res.get("decisions") or {}
    if not dec:
        return
    print("\n**The plan's rules for the idea winning** (1: reloads less than fresh; 2: fewer wrong answers from "
          "state than the shared interpreter; 3: accuracy no more than 5 points below the competent default, "
          "lower end of the paired interval; 4: pulls ahead, by accuracy or by agent time or cost at matched "
          "accuracy. The long arms take long_shared as both baselines.)\n")
    print("| model | K | arm | vs | 1 forgetting | 2 leaking | 3 holds | 4 pulls ahead | accuracy diff (points) | "
          "agent s per request diff | $ per 1,000 requests diff |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    yn = lambda v: "-" if v is None else ("yes" if v is True else "no" if v is False else "not shown")  # noqa: E731
    for m, by_k in dec.items():
        for k, row in sorted(by_k.items(), key=lambda kv: int(kv[0])):
            for idea, r in row.items():
                a, t, c = r["accuracy_diff_vs_default"], r["agent_s_diff_vs_default"], r["cost_diff_vs_default"]
                print(f"| {NAMES.get(m, m)} | {k} | {idea} | {r['baseline']} | {yn(r.get('1_avoids_forgetting'))} | "
                      f"{yn(r.get('2_avoids_leaking'))} | {yn(r['3_holds_against_default'])} | "
                      f"{yn(r['4_pulls_ahead'])} | {100 * a['diff']:+.1f} [{100 * a['lo']:+.1f}, {100 * a['hi']:+.1f}] | "
                      f"{t['diff']:+.1f} [{t['lo']:+.1f}, {t['hi']:+.1f}] | "
                      f"{1000 * c['diff']:+.2f} [{1000 * c['lo']:+.2f}, {1000 * c['hi']:+.2f}] |")


if __name__ == "__main__":
    main()
