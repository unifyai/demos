# demos

Minimal, clean implementations of the ideas behind unify.ai's design-history
posts, each tested against the strongest simple alternatives on data that
reflects the conditions the idea is for.

Why this repo exists (Dan, 9 October 2026): the first experiments behind the
posts ran the original `unify-agent` code at the posts' dates
(`landing-page/experiments/`). Most of what they measured was that code's bugs
and toy settings in which a simpler design was bound to do as well. Here each
idea is rebuilt from scratch, small and bug-free, so the only question is
whether the idea itself holds. Which posts get written, and how they are framed
and dated, is decided later from these results; they may well link to this
code rather than to the archive.

## Rules

1. **One idea per folder,** `concepts/<name>/`: the idea's minimal
   implementation, the baselines, the data generator, the runner, the
   analysis, tests and a README. A concept's code imports nothing from another
   concept, only from `common/`.
2. **Minimal and correct.** The smallest code that implements the idea fully.
   Unit tests cover the mechanism and every baseline. A sanity condition, in
   which every arm should succeed, runs first. Read the transcripts of a sample
   of runs, in every arm, before trusting a number. A bug found in any arm is
   fixed and every affected run re-run; discarded runs are kept in a labelled
   file with the reason.
3. **Strong baselines.** Each idea is compared with what a competent engineer
   would build instead, today, with today's models. That includes "put
   everything in one long context", which million-token contexts have made
   strong. Arms use the same model, prompts and data except for the mechanism.
4. **Conditions written down first.** Before any experiment run, the README
   gets a timestamped "Plan": the conditions the idea is built for (from its
   own reasoning), the arms, the measures, the items and seeds, and what result
   counts as the idea winning. It changes only for a documented fault. Every
   plan includes a small setting and the realistic one, ideally a sweep of
   scale, so the results show where the idea pulls ahead, or that it does not.
5. **Report everything as it comes out.** Effects with intervals
   (`common/src/demos_common/stats.py`), and accuracy, cost and latency side by
   side. If the idea loses, the README says so plainly. Never tune a setting
   until the idea wins.
6. **Models: today's, with strength as an axis.** Only models someone would
   reasonably choose today, on the cost and quality frontier; no older models,
   and no backdating (Dan, 9 October 2026). Stronger models need thinner
   harnesses, so each idea is tested at three tiers (below): its full design on
   the cheap tier, on two families, and its headline comparison on the mid and
   top tiers, showing whether its advantage holds or fades as models get
   stronger. Every arm in a condition uses the same model. A judge comes from a
   different family from the model under test where possible, and is checked
   by hand on a sample.
7. **Every model call goes through `demos_common.LLM`,** which writes each call
   to the concept's ledger with its exact cost. Never print, log or store the
   key.
8. **Budget per concept:** aim under $50 of model spend; stop and report before
   $80.

## Layout

```
common/                       shared, tested: demos_common.LLM (OpenRouter
                              client + cost ledger + retries + time to first
                              token), demos_common.stats, demos_common.ledger
concepts/<name>/              one idea: pyproject.toml (own dependencies, with
                              demos-common as a path dependency), src/, tests/,
                              data/, runs/ (ledger and raw records), results.json,
                              README.md (idea, plan, results, caveats, cost),
                              runs.tar.zst (runs/ archived for git)
bin/with-openrouter-key       runs a command with the key in its environment
```

Run records are too large for git loose (one concept wrote 180 MB in 20,000
files), so `runs/` is ignored and each concept commits `runs.tar.zst` instead,
about 70 times smaller: make it with `tar -cf - runs | zstd -19 -T0 -o runs.tar.zst`
and restore it with `zstd -dc runs.tar.zst | tar -xf -`.

A new concept's `pyproject.toml` depends on the shared package like this:

```toml
[project]
dependencies = ["demos-common"]
[tool.uv.sources]
demos-common = { path = "../../common", editable = true }
```

## Running

On the laptop: `cd concepts/<name> && ../../bin/with-openrouter-key uv run python run.py ...`
(the key comes from Secret Manager as the team service account).

On the cloud machine (Google Cloud VM `blog-exp-1` in `saas-368716`, 32 vCPU,
125 GB, `ssh ubuntu@34.67.145.107`; the company's cloud accounts are listed in
brain's `docs/operations/cloud-compute.md`): copy a concept over with
`rsync -az --exclude .venv --exclude __pycache__ concepts/<name>/ ubuntu@34.67.145.107:demos/concepts/<name>/`,
run the same command there under `nohup` or `tmux` (the key comes from
`~/.experiments.env`), and copy `runs/` and results back after every batch. The
machine is temporary: nothing may live only there.

## Models

OpenRouter list prices per million tokens on 9 October 2026 (all take images
and about a million tokens of context):

| Tier | Models | Input | Output |
|---|---|---|---|
| Cheap | `anthropic/claude-haiku-5.5`, `openai/gpt-6-luna` | $0.10 | $0.50 |
| Mid | `anthropic/claude-sonnet-5.5`, `openai/gpt-6.1-sol` | $2 | $10 |
| Top | `anthropic/claude-opus-5.5`, reasoning effort high | $4 | $20 |

Older generations (GPT-5.x, Claude 4.x, Gemini 2.5 and earlier) are left out:
nobody would choose them today at their price. The fast models in a spoken
agent come from the cheap tier.

## Concepts

| Folder | The idea | Post it came from |
|---|---|---|
| `skills-as-functions` | An agent that keeps what it learns as callable, tested functions beats one that keeps notes, as tasks recur and the world changes | Storing skills as functions |
| `separate-loops` | Talking and acting in separate loops, joined by a steerable handle, keeps a conversation responsive and lets corrections reach work in flight | Talking and acting in separate loops |
| `conversation-context` | Delegated work that carries the caller's conversation, read-only, settles what a request leaves out, through several handovers | Requests that carry their conversation |
| `memory-consolidation` | Offline, hierarchical consolidation of a long history answers questions at every time scale within a fixed budget, without slowing the conversation | Offline hierarchical memory consolidation |
| `linked-libraries` | Procedures as functions and rules as guidance, linked both ways, keep each rule in one place as a library grows | Functions and guidance, linked both ways |
| `demonstration-by-reference` | A demonstration is an ordinary request whose frames travel by reference, so long sessions stay cheap and accurate | Demonstration without a mode |
| `execution-coordinates` | One execution tool whose calls choose what survives, where code runs and which session it joins avoids both forgetting and leaking | Execution as a coordinate |
| `fast-slow-voice` | A spoken agent split into a fast model that talks and a slow one that thinks; which split works best | A fast brain and a slow brain, and its summer redesign |
