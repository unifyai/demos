# Execution coordinates

**The idea.** An agent that acts by writing code gets one execution tool, and
each call chooses its coordinates: what survives the call (stateful, stateless
or read-only), which named session it joins, and which environment it runs in.
A fresh sandbox per call forgets, so expensive state (a slow load, a long
computation) is rebuilt again and again; one interpreter kept for good leaks,
so one task's names and in-place changes reach another task that uses the same
everyday names (`df`, `result`, `totals`). Choosing the coordinates per call is
meant to avoid both, and a read-only call is meant to make an expensive session
safe to explore. Background: the post `execution-as-a-coordinate.mdx` and the
earlier experiment on the old code (`landing-page/experiments/execution-as-a-coordinate/`),
which found that one shared interpreter did as well as the model's choice on
four self-contained requests, that the model never chose read-only, that the
old read-only was a shallow copy (28.5% of steps still changed the session),
that names bound inside `if`/`for`/`try` blocks were lost, and that inspection
could not see the session it listed.

## Plan (written 2026-10-09 22:20 UTC, before any experiment run)

### Conditions the idea is built for

From the idea's own reasoning, the coordinate tool should matter when all of
these hold, and not otherwise:

1. **Several tasks share one agent at once.** Requests from several clients
   arrive interleaved, and each client's work uses the same everyday names
   (`df`, `rates`, `scores`, `result`) for different data. This is where one
   shared interpreter can leak.
2. **State is expensive to rebuild.** Each client's export takes 20 seconds to
   load, and a risk model takes another 20 seconds. This is where a fresh
   sandbox forgets at a cost.
3. **State must survive across requests.** A client's later requests build on
   the earlier ones ("split that by region", "at the same rates", "those
   high-risk customers"), so the data, the rates and the scores are needed
   again requests later.
4. **Some steps should only read.** Each client asks one what-if in the middle
   of the series, whose natural code changes state (a rate, a status column,
   the set of customers), and the next requests go back to the real numbers.
   This is what read-only is for.

The environment axis (which interpreter or virtual environment a session runs
in) is implemented and unit-tested, but not offered to the model in the
experiment: its benefit, packages that cannot share one interpreter, is equally
available to any harness that assigns a task its environment, so it does not
separate the arms, and the tasks below need one environment.

### Workload (generated, seeded: `src/execution_coordinates/data.py`)

Per seed, eight fictional client companies, each with its own payments export
of 100,000 rows (January to April 2025, about 3,000 customers and 20 internal
test accounts), all columns text and in the same schema (`txn_id, created_at,
customer_id, region, type, method, status, currency, amount`), as exports from
one platform are. Clients differ in locale (amounts `1,234.50`, `1.234,50`,
`1 234,50` or `1234.50`; dates month-first, day-first or ISO), currencies (two
or three of EUR, USD, GBP, CHF, SEK, CAD) and reporting currency, test-account
prefix and regions, so one client's data or helper used for another gives a
silently wrong number, not an error. Each client also has risk scores behind
`score_customers(client_id)`.

Each client sends six requests, in order, every one answerable from scratch:

1. Load the export and give net revenue for month M in the reporting currency
   (settled charges minus settled refunds, test accounts excluded, at stated
   rates; the export's formats are described).
2. Split that by region (same rules and rates).
3. A what-if, one of three kinds chosen by the seed: without the n largest
   customers; at a different rate for one currency; with pending transactions
   counted as settled. "Just a what-if."
4. Back to the real numbers: the average settled card charge in month M2, at
   the same rates.
5. Run the risk model (`score_customers`, 20 seconds) and count non-test
   customers scoring above a threshold.
6. Net revenue in month M3 from those high-risk customers, at the same rates.

The clients' requests are merged into one arrival order by a uniformly random
interleaving (seeded), each client's six in order, and handled one at a time.
True answers come from the generator's own numbers and are checked against an
independent parse of the written CSVs. For each request the generator also
computes the answers that leftover or clobbered state would give (another
client's data, rates or scores; the client's own what-if left in place), used
to attribute wrong answers.

**Sizes.** The number of interleaved clients K is the sweep: K = 1, 2, 4, 8
(6, 12, 24 and 48 requests per run). K = 1 is the small setting, where nothing
can leak between clients; K = 4 is the realistic headline; K = 8 is the large
end. Loads are 20 seconds (real sleeps) at every K. A **sanity condition**
precedes everything: K = 1, three requests (1, 2 and 4, no what-if, no risk
model), 2,000 rows, loads of 1 second, two seeds, in which every arm should
answer everything.

### Arms

Same model, system prompt, requests and data in every arm; only the execution
mechanism (and, for the last two, the context design) differs. Every arm's
code runs in the same executor (`executor.py`): a session is a Python worker
process; a fresh call is a new process; read-only forks the session's process.

Context design **threads**: each client's requests are one conversation, so a
request sees that client's earlier requests with the agent's own code, outputs
and answers, and nothing of other clients. Context design **inbox**: one
conversation holds every request from every client in arrival order.

| arm | context | execution |
|---|---|---|
| `fresh` | threads | every call runs in a fresh interpreter and working directory; nothing survives |
| `shared` | threads | one interpreter for the whole run, shared by every client's requests (the tool says so) |
| `per_task` | threads | one interpreter per client, chosen by the harness from the request's client, kept across that client's requests: the competent engineer's default and the strong baseline |
| `coordinate` | threads | the idea: `run_python(code, mode, session)` with mode stateful, stateless or read-only and a named session, plus `list_sessions` and `inspect_session`; the model chooses on every call |
| `long_shared` | inbox | one shared interpreter, with everything in one long context, so the model can see every client's names and changes |
| `long_coordinate` | inbox | the coordinate tool with everything in one long context |
| `per_task_modes` | threads | ablation, K = 4 only: the harness picks the client's interpreter and the model picks stateful, read-only or stateless, which separates the mode axis from the session axis |

Tool descriptions state each arm's policy truthfully (for `shared`, that other
clients' requests run in the same interpreter); their exact texts are in
`tools.py`. The coordinate tool's description gives the one line of guidance
per mode from the post and no advice on naming.

The implementation must fix the old code's faults, each with a test: read-only
is truly read-only (a forked process, so names, in-place changes to objects,
module state and the working directory's files all stay as they were); a name
bound inside any block (`if`, `for`, `while`, `try`, `with`) persists like any
other; the last expression's value is shown; inspection shows every session
the listing shows; invalid combinations return a structured error with a
suggestion.

### Measures

Per request: correct or not (money within 0.05 after rounding to 2 decimals,
counts exact, every region for breakdowns); for wrong answers, the cause
(another client's state, the client's own what-if left in place, or other),
from the executor's record of which request last changed each name the code
read, the generator's signatures, and a reading of every wrong answer at K = 4
and a sample elsewhere; loads of the export and runs of the risk model
(reloads = beyond the first per client); agent time (model latency plus
execution time, excluding queueing for the shared client and the harness's
measurement); model calls, prompt, completion and cached tokens, and cost from
the ledger; for the coordinate arms, the modes and sessions chosen.

Statistics: accuracy with Wilson intervals; differences between arms with a
paired bootstrap over clients (an item is one client's six requests in one
seed, the same in every arm), 10,000 resamples (`demos_common.stats`).

### Models, seeds and budget

- **Primary:** `anthropic/claude-haiku-5.5`, every arm at K = 1, 2, 4, 8,
  seeds 1 to 5, and `per_task_modes` at K = 4, seeds 1 to 5.
- **Second family:** `openai/gpt-6-luna`, the six main arms at K = 4, seeds 1
  to 3.
- **Capable model**, only if spend after the two above is under $25:
  `anthropic/claude-sonnet-5.5`, `fresh`, `shared`, `per_task` and
  `coordinate` at K = 4, seeds 1 to 3.
- Provider-default sampling and reasoning, recorded. Prompt caching on
  (explicit breakpoints for Anthropic models, automatic for OpenAI), the same
  in every arm, as a deployment would use it. At most 25 model calls per
  request, 12 concurrent calls.
- Budget: aim under $50, stop and report before $80.

### What counts as the idea winning

At K = 4 and K = 8, primary model:

1. **Avoids forgetting:** `coordinate` loads less per client than `fresh`
   (interval of the difference excludes 0).
2. **Avoids leaking:** `coordinate` has fewer wrong answers caused by leftover
   or clobbered state than `shared` (interval excludes 0). If `shared` does not
   leak, the leak half of the claim is reported as not shown.
3. **Holds against the competent default:** `coordinate`'s accuracy is no more
   than 5 points below `per_task`'s (lower end of the paired interval above
   -0.05).
4. **Pulls ahead** only if `coordinate`'s accuracy beats `per_task`'s
   (interval above 0), or matches it within 5 points at lower agent time or
   cost (interval excludes 0).

The same rules compare `long_coordinate` with `long_shared`. Anything else is
reported as the idea not holding in that respect, with what we think the reason
is. Expectation before running, for the record: `per_task` at least as accurate
as `coordinate` and cheaper; `fresh` slowest; `shared` leaking more as K grows
unless the model namespaces its names; read-only rarely chosen.

### Order of work

Build and unit-test the executor, the baselines, the agent loop (model
stubbed) and the generator; one-line stack probes of each model; the sanity
condition; a smoke run at K = 2, seed 1, every arm, reading whole transcripts
from each; then the runs above on the cloud machine, copying `runs/` back after
every batch. Discarded runs go to `runs/discarded/` with the reason.

### Amendment (2026-10-09 22:50 UTC, after the sanity and smoke runs, before any main run)

The smoke run cost $0.0004 to $0.0006 per request with Haiku (prompt caching
on), about an eighth of the plan's estimate, so the same budget buys more data.
Decided before any main run, for every arm alike:

1. **Primary model:** seeds 1 to 10 instead of 1 to 5, at every K (40 clients
   at K = 4 and 80 at K = 8 for the paired bootstrap).
2. **Second family** (`openai/gpt-6-luna`): the six main arms at every K,
   seeds 1 to 5 (was K = 4, seeds 1 to 3).
3. **Capable model** (`anthropic/claude-sonnet-5.5`), if spend after 1 and 2
   is under $15: the six main arms at K = 4 and K = 8, and `per_task_modes`
   at K = 4, seeds 1 to 3 (was four arms at K = 4). Stopped if the total
   reaches $45.

Nothing else changes.

### Amendment 2 (superseded by Amendment 3): model strength as an axis (lead's, received 2026-10-10 00:15 UTC, before anything below was run)

Reason (Dan, through the lead): smarter models need thinner harnesses, so the
fair test of a design is on the models it was built for. Model strength becomes
an axis, with every arm in a condition on the same model:

1. **Headline: the models of the design's own time, January 2026.**
   `anthropic/claude-opus-4.5` (the original code's default; $5 in, $25 out
   per million tokens, 200k context) is the primary model, and
   `openai/gpt-5.2` ($1.75 in, $14 out, 400k) the second family, both at
   provider-default reasoning, as before.
2. **Today's tier.** The runs above on today's models (Haiku 5.5 primary,
   GPT-6 luna, Sonnet 5.5) stay, reported as a labelled "today's tier"
   condition, not the headline. Runs of that tier already started finish as
   planned; the added shared-initials check runs on that tier as written.
3. **Today's frontier, if budget allows after the headline:**
   `openai/gpt-6.1-sol` (cheaper than `anthropic/claude-opus-5.5` at high
   reasoning effort) on the headline comparison only (`per_task`,
   `coordinate`, `shared` and `fresh` at K = 4), seeds 1 to 3.
4. **Budget:** a hard stop at $80 for the whole concept, including the $14.92
   spent when this was received. Opus 4.5 is about 2.5 times Sonnet 5.5's
   price; at the measured Sonnet token use the full planned sweep (six arms,
   K = 1, 2, 4, 8, ten seeds) would cost Opus 4.5 roughly $120, which does not
   fit. So the headline runs in this order, each step only if the projected
   total (from the cost per request measured so far) stays under $75, and the
   README reports what the full size would cost:
   a. sanity condition and smoke run (K = 2, seed 1, every arm) for both
      models, transcripts read;
   b. Opus 4.5, the six main arms at K = 4 (the realistic size), seeds 1 to 5;
   c. GPT-5.2, the six main arms at K = 4, seeds 1 to 5;
   d. Opus 4.5, the six main arms at K = 1 and 2, seeds 1 to 5 (the small end);
   e. Opus 4.5, the six main arms at K = 8, seeds 1 to 3 (the large end);
   f. the frontier runs in 3;
   g. Opus 4.5 `per_task_modes` at K = 4, seeds 1 to 5, and GPT-5.2 at K = 1,
      2 and 8, as budget remains.
   The decision rules in "What counts as the idea winning" apply to the
   headline models (Opus 4.5 primary) at the sizes that ran.

   **Order revised after the January models' sanity and smoke runs
   (2026-10-10 00:30 UTC, before any headline run), from their measured
   cost:** Opus 4.5 cost $0.042 a request at K = 2 (it writes about 600 to
   800 output tokens a request, at $25 per million, and caches less, since
   short prompts fall under Anthropic's minimum cacheable length) and GPT-5.2
   $0.012. With $22.61 spent, the steps become: b. Opus 4.5, the six main arms
   and `per_task_modes` at K = 4, seeds 1 to 3 (about $23); c. GPT-5.2, the six
   main arms at K = 4, seeds 1 to 3 (about $5); d. the frontier runs in 3
   (about $2); e. Opus 4.5, the six main arms at K = 2, seeds 1 to 3 (about
   $9); then, while the projected total stays under $75, Opus 4.5 at K = 4
   seeds 4 and 5, GPT-5.2 at K = 2, and Opus 4.5 at K = 1. K = 8 does not fit
   for Opus 4.5 (about $43 for three seeds).

### Amendment 3: three tiers of today's models (lead's, received 2026-10-10 00:40 UTC; supersedes Amendment 2)

Dan replaced Amendment 2: use only today's models that someone would reasonably
choose today, on the cost and quality frontier, and no models of the design's
own era. Model strength stays an axis, in three tiers, every arm in a condition
on the same model:

- **Cheap:** `anthropic/claude-haiku-5.5` and `openai/gpt-6-luna`, the full
  design on both families (Haiku: done; Luna: the sweep already running).
- **Mid:** `anthropic/claude-sonnet-5.5` (one family is enough): the headline
  comparison, which is what ran under Amendment 1 (the six main arms at K = 4
  and 8 and `per_task_modes` at K = 4, seeds 1 to 3).
- **Top:** `anthropic/claude-opus-5.5` with reasoning effort high: the headline
  comparison, `per_task`, `coordinate` and `shared` at K = 4 (then `fresh`),
  with as many seeds as fit, after its own sanity and smoke runs.

The added shared-initials check runs on the cheap and mid tiers as written.
Budget: still the hard stop at $80 for everything; at this amendment $50.06 was
spent, $34.85 of it on the Opus 4.5 and GPT-5.2 runs of Amendment 2, which were
stopped when this arrived (Opus 4.5 had finished 15 of its 21 K = 4 runs,
GPT-5.2 none of its 18). Those runs (their sanity, smoke and K = 4 runs) are
kept, labelled, in `runs/superseded-era-models/` and left out of the analysis;
their cost stays in the ledgers and the totals. What does not fit in the top
tier is reported with its cost.

Top tier as run (2026-10-10 01:20 UTC): the four headline arms and
`per_task_modes` at K = 4 and the four headline arms at K = 8, seeds 1 to 3;
then, with $58.99 spent and a projected total of about $72, `long_shared` and
`long_coordinate` at K = 4, seeds 1 to 3, so the top tier has every main arm at
the realistic size.

Top-tier seeds after the hand-over (2026-10-10 02:05 UTC; every run above was
complete, and no process of this concept was running on the cloud machine).
The first agent was cut off by a usage limit at 01:45 UTC and a second agent
took over. The lead restated the budget at the hand-over: the $34.85 spent on
the January-2026 models stays out of the analysis and does not count against
the cap, and spend on today's models stops before $80 ($36.18 at this point,
from every ledger). Seeds 1 to 3 had been chosen to fit the earlier cap, which
counted that $34.85, and Amendment 3 gives the top tier "as many seeds as
fit". So the six main arms at K = 4 on Opus 5.5 (high) get seeds 4 to 10, the
cheap tier's primary seed count, which makes 40 clients at the realistic size:
42 runs, about $15 at the measured $0.30 to $0.43 a run, taking today's-model
spend to about $51. The number of seeds was fixed by the budget and the
primary model's design before any of these runs; the seeds 1 to 3 results had
been looked at as they came in, and every table reports all ten seeds
together. `per_task_modes` (about $2.30 more) and K = 8 (about $22 more for
the four headline arms) stay at seeds 1 to 3.

### Added check (2026-10-09 23:50 UTC, after seeing Sonnet's session names, before running it)

Sonnet 5.5 names its coordinate-arm sessions with abbreviations, often a single
letter (`t`, `p`, `s`). The sixteen fictional company names in `data.py`
happen to have sixteen different initials, so in the planned data such names
can never collide, while real client lists share initials and prefixes. This
removes, by accident of the data, a failure mode that belongs to the idea: the
coordinate tool leaves naming to the model, and two clients' requests that
choose the same session name share an interpreter. So, outside the
pre-registered comparison and labelled as such: condition `initials`, generated
exactly as `main` except that the eight clients come in four pairs sharing an
initial and a prefix (Saltmarsh and Saltash, Halden and Halford, Pinecrest and
Pinewood, Marigold and Marlow), with `coordinate` and `per_task` at K = 8,
seeds 1 to 3, for Sonnet 5.5 (and, being cheap, Haiku 5.5 and GPT-6 luna).
Expectation: Haiku and Luna, which name sessions after the client's full id,
are unaffected; Sonnet's coordinate arm may put two clients in one session and
leak.

## Results (written 2026-10-10 03:00 UTC, from every complete run on today's models)

Every number below is in `results.json` (`uv run python analyze.py`), and the
tables are printed from it by `scripts/readme_tables.py`; none is typed by
hand. Accuracy intervals are Wilson; differences are paired over clients (an
item is one client's six requests in one seed, the same in every arm) with
10,000 bootstrap resamples. "Agent time" is model latency plus execution.

### Verdict

**The idea holds in the two respects it names, and does not pull ahead of the
competent default.** Choosing the coordinates per call avoided a fresh
sandbox's forgetting (about one load of each export per client against 5.8 to
6.9, and 20 to 24 seconds less agent time a request) and, at the cheap and top
tiers, the shared interpreter's leaks (no wrong answer from another client's
state, against 2 to 14 in 144 to 480 requests at K = 4 and 8 with Haiku 5.5,
GPT-6 luna and Opus 5.5). But one interpreter per client, chosen by the
harness, did exactly as well at every tier and size, with the same or less
agent time, while the coordinate arm cost 8% to 30% more. By the plan's
rules, on the primary model (Haiku 5.5) at K = 4 and K = 8 the idea passes
rules 1 to 3 (avoids forgetting, avoids leaking, holds within 5 points of
`per_task`) and fails rule 4 (pulls ahead): `coordinate` minus `per_task` is +0.4 points [+0.0,
+1.2] at K = 4 and +0.0 at K = 8, at +$0.08 [+0.06, +0.11] and +$0.09 [+0.08,
+0.10] per 1,000 requests (about 20% more) and +0.2 [-0.0, +0.4] and +0.4
[+0.3, +0.5] seconds a request. GPT-6 luna and Opus 5.5 give the same result
at K = 4; at K = 8 their shared interpreters leaked too rarely (3 of 240 and 2
of 144 requests) for rule 2's interval to exclude 0.

**Where the model chooses the session, naming becomes the failure.** At the
mid tier the coordinate arm did worse than `per_task`: Sonnet 5.5 named
sessions with one letter (`s`, `p`, `h`), two clients' separate conversations
chose the same name, and one client's request ran on the other's frames (5
wrong answers in the main runs; in the shared-initials check, 3 more, with all
four pairs of clients sharing a session in every seed). At K = 8 the lower end
of its difference from `per_task` (-2.8 points [-6.9, +0.0]) fails rule 3. A
harness that assigns the interpreter cannot make this mistake.

**In the one-long-context design** `long_coordinate` matched `long_shared` on
accuracy everywhere and was faster for the cheap and mid tiers (0.6 to 1.1
seconds a request for Haiku and Luna; 7 to 8 seconds for Sonnet, whose
`long_shared` reloaded exports defensively), which passes rule 4 on time, at
6% to 16% more cost; but `long_shared` itself never leaked (every model,
seeing every client's code, gave each client its own names), so the leak half
is not shown there, and both long arms cost more than `per_task`. At the top
tier `long_coordinate` was slightly slower (+0.3 [+0.1, +0.5] seconds) and
dearer.

**Across model strength** the picture does not change: the coordinate tool's
gap to `per_task` is nil in accuracy at every tier, its cost premium is
similar (8% to 30%), and only the weak points move. Fresh's forgetting costs
the same 20 to 24 seconds a request at every tier. The declared shared
interpreter leaked least with Opus 5.5 (2.5% of requests at K = 4, against
4.6% for Haiku, 5.0% for Luna and 8.3% for Sonnet), mostly because its code
read another client's names in fewer requests (13%, against 24% for Haiku);
at every tier most of the requests that did read another client's names (70%
to 95%) still answered right, the model having noticed or the value being
harmless. The tool's own risk, colliding session names, appeared only with
Sonnet 5.5. Opus 5.5 used
the tool exactly as designed (it listed the sessions, named them after the
client and its contact once others existed, and went read-only for all 40
what-ifs and many later looks, 40% of its code calls) and still only matched
`per_task`, at 20% more cost.

**Read-only never changed an answer.** Haiku 5.5 almost never chose it (2 of
2,868 code calls in its coordinate arms), Sonnet 5.5 seldom (2% to 12%), GPT-6
luna and Opus 5.5 for 36% to 48% of code calls (looks and what-ifs); but no
model's what-if in any arm changed state that a later request then used
wrongly, so the mode had nothing to protect here (see the caveats).

### Rules of the plan, model by model

- **Haiku 5.5 (primary, 10 seeds; 40 clients at K = 4, 80 at K = 8):**
  rules 1 to 3 pass at K = 4 and 8, rule 4 fails. `long_coordinate` against
  `long_shared`: rule 2 not shown, rule 3 passes, rule 4 passes on agent time
  (-0.8 [-1.3, -0.4] and -1.1 [-1.4, -0.7] seconds a request) at higher cost.
- **GPT-6 luna (second family, 5 seeds):** rules 1 and 3 pass, rule 2 passes
  at K = 4 (shared 6 leaks of 120) but not at K = 8 (3 of 240, interval
  includes 0), rule 4 fails. Long arms as for Haiku.
- **Sonnet 5.5 (mid, 3 seeds):** rule 1 passes; rule 2 fails (the coordinate
  arm leaked too); rule 3 passes at K = 4 and fails at K = 8; rule 4 fails.
  Long arms: rule 4 passes on agent time (-8.0 and -6.8 seconds a request).
- **Opus 5.5, reasoning high (top, 10 seeds at K = 4, 3 at K = 8):** rules 1
  to 3 pass at K = 4 and rule 2 fails at K = 8 (shared 2 leaks of 144), rule 4
  fails: +0.0 points, +0.7 [+0.3, +1.1] seconds and +$2.48 [+1.77, +3.15] per
  1,000 requests against `per_task`. Long arms: no difference in accuracy;
  `long_coordinate` 0.3 [0.1, 0.5] seconds a request slower and $2.24 [1.64,
  2.88] per 1,000 requests dearer, so rule 4 fails.

### The shared-initials check (added, outside the pre-registered comparison)

K = 8, seeds 1 to 3, eight clients in four pairs sharing an initial and a
prefix. Haiku 5.5 and GPT-6 luna named every session after the client's full
id (Luna sometimes adding the contact's name): 144 of 144 in both arms, no
session shared. Sonnet 5.5 put each pair in one session (`h`, `m`, `p`, `s`)
in all three seeds, 12 shared sessions in all, and answered 141 of 144
against `per_task`'s 144 of 144 (-2.1 points [-5.6, +0.0]); its coordinate
arm loaded 1.92 exports per client against 1.00, took 5.4 [2.9, 8.3] more
seconds a request and cost $3.67 [2.74, 4.65] more per 1,000 requests. The
three wrong answers were Pinecrest and Pinewood reading each other's frames
and scores in session `p` (seed 3).

### The top tier's added seeds

Seeds 4 to 10 were run at the hand-over, after seeds 1 to 3 had been seen;
the table "Top tier at K = 4 by block of seeds" shows the two blocks apart.
They agree: every arm but `shared` answered everything in both blocks, and
`shared` leaked 3 times in each.

### Wrong answers, read

64 wrong answers in all (`review.json`): 62 from another client's state (54 in
the shared arms: Haiku 28, Luna 10, Sonnet 8, Opus 8; 8 in Sonnet's coordinate
arm, 5 in the main runs and 3 in the shared-initials check), 2 from slips in
`per_task` with no state involved (Haiku added refunds instead of subtracting
them; Luna copied a total from a display in scientific notation), and none
from a client's own what-if. For every wrong answer the answering cell was
read with the record of who had set each name it read; about a third were
read in full, including all 8 of the top tier's and the 3 of the
shared-initials check.

### Cost

From every ledger (the laptop's `runs/ledger.jsonl` and the cloud machine's
`runs/ledger-vm*.jsonl`, none counted twice): **$51.30 on today's models**
(Haiku 5.5 $3.67, GPT-6 luna $1.20, Sonnet 5.5 $13.05, Opus 5.5 $33.38, of
which $15.12 was the top tier's seeds 4 to 10), under the $80 cap; plus $34.85
on the superseded January-2026 runs (Opus 4.5, GPT-5.2), which are out of the
analysis and, as the lead set at the hand-over, out of the cap; $86.16 in all.
The cheap tier's full design (six main arms at K = 1, 2, 4, 8 and
`per_task_modes` at K = 4, ten seeds) would cost about $41 on Sonnet 5.5 ($31
more than ran) and $91 on Opus 5.5 ($59 more); the largest missing pieces on
the top tier are K = 8 at seeds 4 to 10 for the four headline arms (about $22)
and `per_task_modes` at seeds 4 to 10 (about $2.30).

### Tables

**By tier** (accuracy %, with the number of wrong answers caused by another client's state in brackets; then coordinate minus per_task and coordinate minus shared, in points of accuracy with 95% intervals; loads of the export per client for fresh / shared / per_task / coordinate)

| tier | model | K | clients | fresh | shared | per_task | coordinate | long_shared | long_coordinate | coordinate - per_task | coordinate - shared | loads per client |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cheap | Claude Haiku 5.5 | 1 | 10 | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 6.9 / 1.0 / 1.0 / 1.0 |
| cheap | Claude Haiku 5.5 | 2 | 20 | 100.0 (0) | 97.5 (3) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.0 [+0.0, +0.0] | +2.5 [+0.0, +5.0] | 6.8 / 1.9 / 1.0 / 1.1 |
| cheap | Claude Haiku 5.5 | 4 | 40 | 100.0 (0) | 95.4 (11) | 99.6 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.4 [+0.0, +1.2] | +4.6 [+1.7, +7.9] | 6.8 / 3.3 / 1.0 / 1.0 |
| cheap | Claude Haiku 5.5 | 8 | 80 | 100.0 (0) | 97.1 (14) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.0 [+0.0, +0.0] | +2.9 [+0.6, +5.6] | 6.8 / 4.0 / 1.0 / 1.0 |
| cheap | GPT-6 luna | 1 | 5 | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | 6.2 / 1.0 / 1.0 / 1.0 |
| cheap | GPT-6 luna | 2 | 10 | 100.0 (0) | 98.3 (1) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.0 [+0.0, +0.0] | +1.7 [+0.0, +5.0] | 5.8 / 2.4 / 1.0 / 1.0 |
| cheap | GPT-6 luna | 4 | 20 | 100.0 (0) | 95.0 (6) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.0 [+0.0, +0.0] | +5.0 [+0.8, +10.8] | 5.8 / 2.8 / 1.0 / 1.0 |
| cheap | GPT-6 luna | 8 | 40 | 100.0 (0) | 98.8 (3) | 99.6 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.4 [+0.0, +1.2] | +1.2 [+0.0, +2.9] | 6.1 / 4.1 / 1.0 / 1.0 |
| mid | Claude Sonnet 5.5 | 4 | 12 | 100.0 (0) | 91.7 (6) | 100.0 (0) | 98.6 (1) | 100.0 (0) | 100.0 (0) | -1.4 [-4.2, +0.0] | +6.9 [-1.4, +18.1] | 6.4 / 2.0 / 1.0 / 1.2 |
| mid | Claude Sonnet 5.5 | 8 | 24 | 100.0 (0) | 98.6 (2) | 100.0 (0) | 97.2 (4) | 100.0 (0) | 100.0 (0) | -2.8 [-6.9, +0.0] | -1.4 [-4.2, +0.0] | 6.4 / 2.7 / 1.0 / 1.1 |
| top | Claude Opus 5.5 (high) | 4 | 40 | 100.0 (0) | 97.5 (6) | 100.0 (0) | 100.0 (0) | 100.0 (0) | 100.0 (0) | +0.0 [+0.0, +0.0] | +2.5 [+0.8, +5.0] | 6.7 / 1.1 / 1.0 / 1.0 |
| top | Claude Opus 5.5 (high) | 8 | 24 | 100.0 (0) | 98.6 (2) | 100.0 (0) | 100.0 (0) | - | - | +0.0 [+0.0, +0.0] | +1.4 [+0.0, +3.5] | 6.8 / 1.1 / 1.0 / 1.0 |

**Claude Haiku 5.5** (requests correct, accuracy with 95% Wilson interval; wrong answers caused by state, all of them leaks from another client unless marked; loads of the export per client; agent minutes per run; cost per request; prompt tokens per request)

| K | arm | correct | accuracy % [95% CI] | wrong from state | other wrong | loads per client | agent min per run | $ per 1,000 requests | prompt tokens per request |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fresh | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 6.90 | 3.5 | 0.56 | 11.4k |
| 1 | shared | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 1.1 | 0.43 | 9.5k |
| 1 | per_task | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 1.1 | 0.39 | 9.1k |
| 1 | coordinate | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 1.2 | 0.51 | 12.0k |
| 1 | long_shared | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 1.1 | 0.41 | 9.4k |
| 1 | long_coordinate | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 1.1 | 0.49 | 11.7k |
| 2 | fresh | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 6.80 | 7.0 | 0.57 | 11.6k |
| 2 | shared | 117/120 | 97.5 [92.9, 99.1] | 3 | 0 | 1.90 | 3.2 | 0.59 | 13.6k |
| 2 | per_task | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.00 | 2.1 | 0.41 | 9.3k |
| 2 | coordinate | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.05 | 2.2 | 0.51 | 11.7k |
| 2 | long_shared | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.00 | 2.2 | 0.48 | 15.1k |
| 2 | long_coordinate | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.00 | 2.1 | 0.55 | 18.2k |
| 4 | fresh | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 6.85 | 14.1 | 0.58 | 12.0k |
| 4 | shared | 229/240 | 95.4 [92.0, 97.4] | 11 | 0 | 3.27 | 9.0 | 0.71 | 15.8k |
| 4 | per_task | 239/240 | 99.6 [97.7, 99.9] | 0 | 1 | 1.00 | 4.3 | 0.41 | 9.4k |
| 4 | coordinate | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 4.4 | 0.50 | 11.6k |
| 4 | long_shared | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 4.6 | 0.58 | 25.8k |
| 4 | long_coordinate | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 4.3 | 0.65 | 30.0k |
| 4 | per_task_modes | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 4.3 | 0.47 | 10.8k |
| 8 | fresh | 480/480 | 100.0 [99.2, 100.0] | 0 | 0 | 6.81 | 28.1 | 0.65 | 11.7k |
| 8 | shared | 466/480 | 97.1 [95.2, 98.3] | 14 | 0 | 3.96 | 20.5 | 0.82 | 17.1k |
| 8 | per_task | 480/480 | 100.0 [99.2, 100.0] | 0 | 0 | 1.00 | 8.5 | 0.42 | 9.4k |
| 8 | coordinate | 480/480 | 100.0 [99.2, 100.0] | 0 | 0 | 1.01 | 8.8 | 0.51 | 11.7k |
| 8 | long_shared | 480/480 | 100.0 [99.2, 100.0] | 0 | 0 | 1.00 | 9.4 | 0.83 | 50.4k |
| 8 | long_coordinate | 480/480 | 100.0 [99.2, 100.0] | 0 | 0 | 1.00 | 8.5 | 0.88 | 53.8k |
| 8 | per_task@initials | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 8.6 | 0.43 | 9.8k |
| 8 | coordinate@initials | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 8.8 | 0.50 | 11.8k |

**Claude Opus 5.5 (high)** (requests correct, accuracy with 95% Wilson interval; wrong answers caused by state, all of them leaks from another client unless marked; loads of the export per client; agent minutes per run; cost per request; prompt tokens per request)

| K | arm | correct | accuracy % [95% CI] | wrong from state | other wrong | loads per client | agent min per run | $ per 1,000 requests | prompt tokens per request |
|---|---|---|---|---|---|---|---|---|---|
| 4 | fresh | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 6.70 | 15.2 | 16.96 | 10.1k |
| 4 | shared | 234/240 | 97.5 [94.7, 98.8] | 6 | 0 | 1.10 | 5.9 | 15.03 | 10.4k |
| 4 | per_task | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 5.4 | 12.38 | 9.1k |
| 4 | coordinate | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 5.7 | 14.85 | 11.3k |
| 4 | long_shared | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 5.4 | 14.40 | 22.1k |
| 4 | long_coordinate | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 5.5 | 16.63 | 25.9k |
| 4 | per_task_modes | 72/72 | 100.0 [94.9, 100.0] | 0 | 0 | 1.00 | 5.6 | 13.80 | 10.5k |
| 8 | fresh | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 6.75 | 30.3 | 19.39 | 9.9k |
| 8 | shared | 142/144 | 98.6 [95.1, 99.6] | 2 | 0 | 1.08 | 12.4 | 17.71 | 11.1k |
| 8 | per_task | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 11.1 | 12.98 | 9.3k |
| 8 | coordinate | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 11.6 | 16.07 | 11.7k |

**Claude Sonnet 5.5** (requests correct, accuracy with 95% Wilson interval; wrong answers caused by state, all of them leaks from another client unless marked; loads of the export per client; agent minutes per run; cost per request; prompt tokens per request)

| K | arm | correct | accuracy % [95% CI] | wrong from state | other wrong | loads per client | agent min per run | $ per 1,000 requests | prompt tokens per request |
|---|---|---|---|---|---|---|---|---|---|
| 4 | fresh | 72/72 | 100.0 [94.9, 100.0] | 0 | 0 | 6.42 | 13.8 | 7.29 | 9.2k |
| 4 | shared | 66/72 | 91.7 [83.0, 96.1] | 6 | 0 | 2.00 | 6.9 | 6.92 | 10.1k |
| 4 | per_task | 72/72 | 100.0 [94.9, 100.0] | 0 | 0 | 1.00 | 4.8 | 5.21 | 7.9k |
| 4 | coordinate | 71/72 | 98.6 [92.5, 99.8] | 1 | 0 | 1.17 | 5.3 | 6.46 | 9.8k |
| 4 | long_shared | 72/72 | 100.0 [94.9, 100.0] | 0 | 0 | 2.75 | 8.3 | 6.79 | 20.3k |
| 4 | long_coordinate | 72/72 | 100.0 [94.9, 100.0] | 0 | 0 | 1.00 | 5.1 | 7.85 | 24.7k |
| 4 | per_task_modes | 72/72 | 100.0 [94.9, 100.0] | 0 | 0 | 1.00 | 4.8 | 5.81 | 8.5k |
| 8 | fresh | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 6.42 | 27.8 | 8.74 | 9.2k |
| 8 | shared | 142/144 | 98.6 [95.1, 99.6] | 2 | 0 | 2.67 | 16.2 | 8.57 | 11.3k |
| 8 | per_task | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 9.4 | 5.36 | 7.8k |
| 8 | coordinate | 140/144 | 97.2 [93.1, 98.9] | 4 | 0 | 1.08 | 11.1 | 6.95 | 9.8k |
| 8 | long_shared | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 2.75 | 15.5 | 8.54 | 37.5k |
| 8 | long_coordinate | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 10.1 | 9.40 | 41.2k |
| 8 | per_task@initials | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 9.0 | 5.19 | 7.5k |
| 8 | coordinate@initials | 141/144 | 97.9 [94.1, 99.3] | 3 | 0 | 1.92 | 13.3 | 8.86 | 12.9k |

**GPT-6 luna** (requests correct, accuracy with 95% Wilson interval; wrong answers caused by state, all of them leaks from another client unless marked; loads of the export per client; agent minutes per run; cost per request; prompt tokens per request)

| K | arm | correct | accuracy % [95% CI] | wrong from state | other wrong | loads per client | agent min per run | $ per 1,000 requests | prompt tokens per request |
|---|---|---|---|---|---|---|---|---|---|
| 1 | fresh | 30/30 | 100.0 [88.6, 100.0] | 0 | 0 | 6.20 | 3.4 | 0.35 | 6.1k |
| 1 | shared | 30/30 | 100.0 [88.6, 100.0] | 0 | 0 | 1.00 | 1.2 | 0.29 | 5.3k |
| 1 | per_task | 30/30 | 100.0 [88.6, 100.0] | 0 | 0 | 1.00 | 1.2 | 0.27 | 5.2k |
| 1 | coordinate | 30/30 | 100.0 [88.6, 100.0] | 0 | 0 | 1.00 | 1.2 | 0.33 | 6.7k |
| 1 | long_shared | 30/30 | 100.0 [88.6, 100.0] | 0 | 0 | 1.00 | 1.2 | 0.28 | 5.3k |
| 1 | long_coordinate | 30/30 | 100.0 [88.6, 100.0] | 0 | 0 | 1.00 | 1.2 | 0.34 | 7.4k |
| 2 | fresh | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 5.80 | 6.7 | 0.33 | 5.3k |
| 2 | shared | 59/60 | 98.3 [91.1, 99.7] | 1 | 0 | 2.40 | 4.3 | 0.44 | 9.1k |
| 2 | per_task | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 2.4 | 0.28 | 5.3k |
| 2 | coordinate | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 2.5 | 0.33 | 6.7k |
| 2 | long_shared | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 2.5 | 0.32 | 9.2k |
| 2 | long_coordinate | 60/60 | 100.0 [94.0, 100.0] | 0 | 0 | 1.00 | 2.4 | 0.32 | 10.3k |
| 4 | fresh | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 5.75 | 12.8 | 0.32 | 5.2k |
| 4 | shared | 114/120 | 95.0 [89.5, 97.7] | 6 | 0 | 2.80 | 8.6 | 0.42 | 8.6k |
| 4 | per_task | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.00 | 4.8 | 0.28 | 5.4k |
| 4 | coordinate | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.00 | 4.7 | 0.31 | 6.5k |
| 4 | long_shared | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.00 | 4.9 | 0.37 | 16.9k |
| 4 | long_coordinate | 120/120 | 100.0 [96.9, 100.0] | 0 | 0 | 1.00 | 4.6 | 0.40 | 18.5k |
| 8 | fresh | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 6.08 | 27.2 | 0.33 | 5.6k |
| 8 | shared | 237/240 | 98.8 [96.4, 99.6] | 3 | 0 | 4.10 | 21.9 | 0.43 | 9.1k |
| 8 | per_task | 239/240 | 99.6 [97.7, 99.9] | 0 | 1 | 1.00 | 9.8 | 0.28 | 5.4k |
| 8 | coordinate | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 9.7 | 0.30 | 6.4k |
| 8 | long_shared | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 10.9 | 0.50 | 30.5k |
| 8 | long_coordinate | 240/240 | 100.0 [98.4, 100.0] | 0 | 0 | 1.00 | 10.1 | 0.56 | 34.9k |
| 8 | per_task@initials | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 9.4 | 0.29 | 5.5k |
| 8 | coordinate@initials | 144/144 | 100.0 [97.4, 100.0] | 0 | 0 | 1.00 | 9.5 | 0.31 | 6.5k |

**Paired differences** (second arm minus first; bootstrap over clients, 95% interval)

| model | K | difference | accuracy (points) | wrong from state (points) | reloads per client | agent s per request | $ per 1,000 requests |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 1 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.10 [+0.00, +0.30] | +0.9 [+0.2, +1.8] | +0.12 [+0.06, +0.16] |
| Claude Haiku 5.5 | 1 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.80 [-7.00, -6.50] | -23.8 [-24.7, -22.7] | -0.05 [-0.11, +0.00] |
| Claude Haiku 5.5 | 1 | coordinate vs shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.10 [+0.00, +0.30] | +0.7 [-0.1, +1.8] | +0.08 [+0.01, +0.15] |
| Claude Haiku 5.5 | 1 | per_task vs shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.2 [-0.5, +0.1] | -0.04 [-0.06, -0.01] |
| Claude Haiku 5.5 | 1 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.3 [-0.0, +0.5] | +0.08 [+0.05, +0.12] |
| Claude Haiku 5.5 | 1 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.3 [-0.0, +0.7] | +0.10 [+0.07, +0.14] |
| Claude Haiku 5.5 | 1 | long_shared vs shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.1 [-0.4, +0.1] | -0.02 [-0.06, +0.01] |
| Claude Haiku 5.5 | 2 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.05 [+0.00, +0.15] | +0.5 [+0.2, +0.9] | +0.10 [+0.06, +0.14] |
| Claude Haiku 5.5 | 2 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.80 [-7.15, -6.45] | -23.9 [-25.2, -22.6] | -0.06 [-0.09, -0.03] |
| Claude Haiku 5.5 | 2 | coordinate vs shared | +2.5 [+0.0, +5.0] | -2.5 [-5.0, +0.0] | -1.15 [-1.80, -0.55] | -4.8 [-7.4, -2.4] | -0.09 [-0.17, -0.00] |
| Claude Haiku 5.5 | 2 | per_task vs shared | +2.5 [+0.0, +5.0] | -2.5 [-5.0, +0.0] | -1.20 [-1.85, -0.65] | -5.3 [-7.7, -3.0] | -0.18 [-0.26, -0.11] |
| Claude Haiku 5.5 | 2 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.10 [-0.25, +0.00] | -0.2 [-0.8, +0.2] | +0.07 [+0.04, +0.10] |
| Claude Haiku 5.5 | 2 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.2 [+0.0, +0.4] | +0.14 [+0.12, +0.17] |
| Claude Haiku 5.5 | 2 | long_shared vs shared | +2.5 [+0.0, +5.0] | -2.5 [-5.0, +0.0] | -1.10 [-1.80, -0.45] | -4.9 [-7.5, -2.4] | -0.11 [-0.19, -0.03] |
| Claude Haiku 5.5 | 4 | coordinate vs per_task | +0.4 [+0.0, +1.2] | +0.0 [+0.0, +0.0] | -0.03 [-0.07, +0.00] | +0.2 [-0.0, +0.4] | +0.08 [+0.06, +0.11] |
| Claude Haiku 5.5 | 4 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.92 [-7.20, -6.65] | -24.4 [-25.4, -23.4] | -0.08 [-0.10, -0.07] |
| Claude Haiku 5.5 | 4 | coordinate vs shared | +4.6 [+1.7, +7.9] | -4.6 [-7.9, -1.7] | -2.95 [-3.52, -2.35] | -11.5 [-13.6, -9.3] | -0.21 [-0.26, -0.16] |
| Claude Haiku 5.5 | 4 | per_task vs shared | +4.2 [+1.3, +7.5] | -4.6 [-7.9, -1.7] | -2.93 [-3.50, -2.33] | -11.7 [-13.8, -9.5] | -0.29 [-0.34, -0.24] |
| Claude Haiku 5.5 | 4 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.30 [-0.45, -0.17] | -0.8 [-1.3, -0.4] | +0.07 [+0.04, +0.10] |
| Claude Haiku 5.5 | 4 | long_coordinate vs per_task | +0.4 [+0.0, +1.2] | +0.0 [+0.0, +0.0] | -0.03 [-0.07, +0.00] | -0.0 [-0.3, +0.2] | +0.23 [+0.21, +0.26] |
| Claude Haiku 5.5 | 4 | per_task_modes vs per_task | +0.4 [+0.0, +1.2] | +0.0 [+0.0, +0.0] | -0.03 [-0.07, +0.00] | +0.2 [-0.1, +0.4] | +0.05 [+0.03, +0.07] |
| Claude Haiku 5.5 | 4 | long_shared vs shared | +4.6 [+1.7, +7.9] | -4.6 [-7.9, -1.7] | -2.65 [-3.20, -2.10] | -10.9 [-12.9, -8.8] | -0.13 [-0.18, -0.07] |
| Claude Haiku 5.5 | 8 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.01 [+0.00, +0.04] | +0.4 [+0.3, +0.5] | +0.09 [+0.08, +0.10] |
| Claude Haiku 5.5 | 8 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.83 [-6.99, -6.65] | -24.0 [-24.7, -23.4] | -0.15 [-0.17, -0.13] |
| Claude Haiku 5.5 | 8 | coordinate vs shared | +2.9 [+0.6, +5.6] | -2.9 [-5.6, -0.6] | -3.84 [-4.19, -3.49] | -14.6 [-15.9, -13.4] | -0.31 [-0.36, -0.27] |
| Claude Haiku 5.5 | 8 | per_task vs shared | +2.9 [+0.6, +5.6] | -2.9 [-5.6, -0.6] | -3.85 [-4.20, -3.50] | -15.0 [-16.3, -13.8] | -0.40 [-0.45, -0.35] |
| Claude Haiku 5.5 | 8 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.33 [-0.42, -0.23] | -1.1 [-1.4, -0.7] | +0.05 [+0.03, +0.06] |
| Claude Haiku 5.5 | 8 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.0 [-0.1, +0.1] | +0.46 [+0.44, +0.48] |
| Claude Haiku 5.5 | 8 | long_shared vs shared | +2.9 [+0.6, +5.6] | -2.9 [-5.6, -0.6] | -3.52 [-3.89, -3.16] | -14.0 [-15.2, -12.7] | +0.01 [-0.04, +0.06] |
| Claude Haiku 5.5 | 8 | coordinate@initials vs per_task@initials | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.3 [+0.0, +0.6] | +0.07 [+0.04, +0.11] |
| Claude Opus 5.5 (high) | 4 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.7 [+0.3, +1.1] | +2.48 [+1.77, +3.15] |
| Claude Opus 5.5 (high) | 4 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.78 [-7.00, -6.55] | -23.7 [-24.6, -22.7] | -2.11 [-2.98, -1.24] |
| Claude Opus 5.5 (high) | 4 | coordinate vs shared | +2.5 [+0.8, +5.0] | -2.5 [-5.0, -0.8] | -0.12 [-0.25, -0.03] | -0.5 [-1.2, +0.1] | -0.18 [-1.09, +0.74] |
| Claude Opus 5.5 (high) | 4 | per_task vs shared | +2.5 [+0.8, +5.0] | -2.5 [-5.0, -0.8] | -0.12 [-0.25, -0.03] | -1.2 [-2.0, -0.6] | -2.65 [-3.66, -1.66] |
| Claude Opus 5.5 (high) | 4 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.3 [+0.1, +0.5] | +2.24 [+1.64, +2.88] |
| Claude Opus 5.5 (high) | 4 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.2 [-0.2, +0.5] | +4.25 [+3.66, +4.86] |
| Claude Opus 5.5 (high) | 4 | per_task_modes vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.4 [+0.1, +0.7] | +1.53 [+0.90, +2.15] |
| Claude Opus 5.5 (high) | 4 | long_shared vs shared | +2.5 [+0.8, +5.0] | -2.5 [-5.0, -0.8] | -0.12 [-0.25, -0.03] | -1.3 [-2.1, -0.7] | -0.64 [-1.57, +0.29] |
| Claude Opus 5.5 (high) | 8 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.6 [+0.0, +1.2] | +3.09 [+2.06, +4.15] |
| Claude Opus 5.5 (high) | 8 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.79 [-7.17, -6.42] | -23.3 [-24.9, -21.8] | -3.31 [-4.47, -2.15] |
| Claude Opus 5.5 (high) | 8 | coordinate vs shared | +1.4 [+0.0, +3.5] | -1.4 [-3.5, +0.0] | -0.08 [-0.21, +0.00] | -1.0 [-1.9, -0.1] | -1.64 [-3.14, -0.13] |
| Claude Opus 5.5 (high) | 8 | per_task vs shared | +1.4 [+0.0, +3.5] | -1.4 [-3.5, +0.0] | -0.08 [-0.21, +0.00] | -1.6 [-2.4, -0.9] | -4.73 [-6.07, -3.37] |
| Claude Sonnet 5.5 | 4 | coordinate vs per_task | -1.4 [-4.2, +0.0] | +1.4 [+0.0, +4.2] | +0.25 [+0.00, +0.75] | +1.1 [-0.6, +3.9] | +1.25 [+0.44, +2.47] |
| Claude Sonnet 5.5 | 4 | coordinate vs fresh | -1.4 [-4.2, +0.0] | +1.4 [+0.0, +4.2] | -6.17 [-6.67, -5.67] | -21.4 [-23.8, -18.6] | -0.83 [-1.60, +0.43] |
| Claude Sonnet 5.5 | 4 | coordinate vs shared | +6.9 [-1.4, +18.1] | -6.9 [-18.1, +1.4] | -1.00 [-1.67, -0.25] | -3.9 [-7.0, -0.5] | -0.46 [-1.57, +0.81] |
| Claude Sonnet 5.5 | 4 | per_task vs shared | +8.3 [+0.0, +18.1] | -8.3 [-18.1, +0.0] | -1.25 [-1.75, -0.75] | -5.1 [-7.0, -3.2] | -1.70 [-2.38, -1.13] |
| Claude Sonnet 5.5 | 4 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -2.50 [-3.58, -1.42] | -8.0 [-11.7, -4.5] | +1.06 [+0.76, +1.36] |
| Claude Sonnet 5.5 | 4 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.08 [+0.00, +0.25] | +0.6 [-0.2, +1.5] | +2.63 [+2.14, +3.08] |
| Claude Sonnet 5.5 | 4 | per_task_modes vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.1 [-0.7, +0.6] | +0.59 [+0.29, +0.90] |
| Claude Sonnet 5.5 | 4 | long_shared vs shared | +8.3 [+0.0, +18.1] | -8.3 [-18.1, +0.0] | +1.33 [+0.33, +2.33] | +3.5 [+0.2, +6.9] | -0.13 [-0.77, +0.44] |
| Claude Sonnet 5.5 | 8 | coordinate vs per_task | -2.8 [-6.9, +0.0] | +2.8 [+0.0, +6.9] | +0.38 [+0.00, +1.04] | +2.1 [+0.2, +5.1] | +1.59 [+0.93, +2.42] |
| Claude Sonnet 5.5 | 8 | coordinate vs fresh | -2.8 [-6.9, +0.0] | +2.8 [+0.0, +6.9] | -6.04 [-6.54, -5.33] | -20.9 [-23.1, -17.6] | -1.78 [-2.56, -0.85] |
| Claude Sonnet 5.5 | 8 | coordinate vs shared | -1.4 [-4.2, +0.0] | +1.4 [+0.0, +4.2] | -1.54 [-2.38, -0.54] | -6.4 [-10.0, -2.1] | -1.62 [-2.59, -0.45] |
| Claude Sonnet 5.5 | 8 | per_task vs shared | +1.4 [+0.0, +4.2] | -1.4 [-4.2, +0.0] | -1.92 [-2.50, -1.42] | -8.6 [-10.9, -6.3] | -3.21 [-3.95, -2.47] |
| Claude Sonnet 5.5 | 8 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -2.12 [-2.92, -1.38] | -6.8 [-9.8, -4.0] | +0.86 [+0.44, +1.26] |
| Claude Sonnet 5.5 | 8 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.9 [+0.3, +1.3] | +4.04 [+3.68, +4.38] |
| Claude Sonnet 5.5 | 8 | long_shared vs shared | +1.4 [+0.0, +4.2] | -1.4 [-4.2, +0.0] | +0.21 [-0.79, +1.25] | -0.9 [-4.6, +2.8] | -0.03 [-0.82, +0.71] |
| Claude Sonnet 5.5 | 8 | coordinate@initials vs per_task@initials | -2.1 [-5.6, +0.0] | +2.1 [+0.0, +5.6] | +1.12 [+0.50, +1.88] | +5.4 [+2.9, +8.3] | +3.67 [+2.74, +4.65] |
| GPT-6 luna | 1 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.3 [+0.1, +0.6] | +0.06 [+0.03, +0.09] |
| GPT-6 luna | 1 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.60 [-7.40, -6.00] | -22.4 [-25.6, -19.7] | -0.02 [-0.06, +0.01] |
| GPT-6 luna | 1 | coordinate vs shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.5 [-0.2, +1.4] | +0.04 [-0.01, +0.09] |
| GPT-6 luna | 1 | per_task vs shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.2 [-0.3, +0.8] | -0.02 [-0.05, +0.02] |
| GPT-6 luna | 1 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.2 [-0.1, +0.4] | +0.06 [+0.03, +0.09] |
| GPT-6 luna | 1 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.4 [+0.1, +0.6] | +0.08 [+0.06, +0.09] |
| GPT-6 luna | 1 | long_shared vs shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.4 [-0.0, +0.9] | -0.01 [-0.04, +0.03] |
| GPT-6 luna | 2 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.6 [+0.1, +1.0] | +0.05 [+0.02, +0.07] |
| GPT-6 luna | 2 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.20 [-6.60, -5.80] | -21.1 [-22.1, -20.0] | +0.00 [-0.02, +0.03] |
| GPT-6 luna | 2 | coordinate vs shared | +1.7 [+0.0, +5.0] | -1.7 [-5.0, +0.0] | -2.10 [-3.30, -1.00] | -9.0 [-13.9, -4.1] | -0.11 [-0.20, -0.02] |
| GPT-6 luna | 2 | per_task vs shared | +1.7 [+0.0, +5.0] | -1.7 [-5.0, +0.0] | -2.10 [-3.30, -1.00] | -9.5 [-14.5, -4.5] | -0.16 [-0.26, -0.06] |
| GPT-6 luna | 2 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.5 [-1.1, +0.1] | +0.01 [-0.02, +0.03] |
| GPT-6 luna | 2 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.0 [-0.7, +0.6] | +0.04 [-0.00, +0.08] |
| GPT-6 luna | 2 | long_shared vs shared | +1.7 [+0.0, +5.0] | -1.7 [-5.0, +0.0] | -2.10 [-3.30, -1.00] | -9.0 [-13.6, -4.5] | -0.13 [-0.20, -0.05] |
| GPT-6 luna | 4 | coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.2 [-0.5, +0.2] | +0.03 [+0.01, +0.05] |
| GPT-6 luna | 4 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -5.90 [-6.30, -5.50] | -20.2 [-22.0, -18.4] | -0.01 [-0.03, +0.01] |
| GPT-6 luna | 4 | coordinate vs shared | +5.0 [+0.8, +10.8] | -5.0 [-10.8, -0.8] | -2.35 [-3.30, -1.40] | -9.7 [-13.4, -6.0] | -0.12 [-0.17, -0.06] |
| GPT-6 luna | 4 | per_task vs shared | +5.0 [+0.8, +10.8] | -5.0 [-10.8, -0.8] | -2.35 [-3.30, -1.40] | -9.5 [-13.3, -5.8] | -0.14 [-0.19, -0.09] |
| GPT-6 luna | 4 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.6 [-0.9, -0.4] | +0.03 [+0.00, +0.06] |
| GPT-6 luna | 4 | long_coordinate vs per_task | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.4 [-0.7, -0.0] | +0.12 [+0.10, +0.15] |
| GPT-6 luna | 4 | long_shared vs shared | +5.0 [+0.8, +10.8] | -5.0 [-10.8, -0.8] | -2.35 [-3.30, -1.40] | -9.3 [-12.9, -5.6] | -0.05 [-0.10, -0.00] |
| GPT-6 luna | 8 | coordinate vs per_task | +0.4 [+0.0, +1.2] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -0.2 [-0.5, +0.2] | +0.02 [+0.01, +0.04] |
| GPT-6 luna | 8 | coordinate vs fresh | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -6.33 [-6.53, -6.15] | -21.9 [-22.7, -21.2] | -0.02 [-0.04, -0.01] |
| GPT-6 luna | 8 | coordinate vs shared | +1.2 [+0.0, +2.9] | -1.2 [-2.9, +0.0] | -3.90 [-4.45, -3.30] | -15.2 [-17.3, -12.9] | -0.13 [-0.16, -0.10] |
| GPT-6 luna | 8 | per_task vs shared | +0.8 [-0.8, +2.5] | -1.2 [-2.9, +0.0] | -3.90 [-4.45, -3.30] | -15.1 [-17.1, -12.8] | -0.15 [-0.18, -0.13] |
| GPT-6 luna | 8 | long_coordinate vs long_shared | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | -1.0 [-1.3, -0.7] | +0.06 [+0.05, +0.07] |
| GPT-6 luna | 8 | long_coordinate vs per_task | +0.4 [+0.0, +1.2] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.3 [-0.0, +0.6] | +0.28 [+0.25, +0.30] |
| GPT-6 luna | 8 | long_shared vs shared | +1.2 [+0.0, +2.9] | -1.2 [-2.9, +0.0] | -3.90 [-4.45, -3.30] | -13.8 [-15.9, -11.5] | +0.07 [+0.04, +0.10] |
| GPT-6 luna | 8 | coordinate@initials vs per_task@initials | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.00 [+0.00, +0.00] | +0.1 [-0.2, +0.4] | +0.03 [+0.01, +0.04] |

**What the coordinate arms chose**

| model | arm | K | code calls by mode | listing / inspection calls | what-if calls by mode | sessions per client | sessions shared by two clients |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | per_task | 1 | 82 stateful | 0 / 0 | 10 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | coordinate | 1 | 84 stateful | 0 / 0 | 10 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | long_coordinate | 1 | 83 stateful | 0 / 0 | 11 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | per_task | 2 | 158 stateful | 0 / 0 | 20 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | coordinate | 2 | 162 stateful, 2 read_only | 0 / 0 | 20 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | long_coordinate | 2 | 154 stateful | 0 / 0 | 20 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | per_task | 4 | 324 stateful | 0 / 0 | 40 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | coordinate | 4 | 322 stateful | 0 / 0 | 40 stateful | 1.02 | 0 |
| Claude Haiku 5.5 | long_coordinate | 4 | 301 stateful | 0 / 0 | 40 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | per_task_modes | 4 | 332 stateful | 0 / 0 | 39 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | per_task | 8 | 650 stateful | 0 / 0 | 80 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | coordinate | 8 | 648 stateful | 0 / 0 | 81 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | long_coordinate | 8 | 584 stateful | 0 / 0 | 80 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | per_task@initials | 8 | 199 stateful | 0 / 0 | 24 stateful | 1.00 | 0 |
| Claude Haiku 5.5 | coordinate@initials | 8 | 196 stateful | 0 / 0 | 24 stateful | 1.00 | 0 |
| Claude Opus 5.5 (high) | per_task | 4 | 342 stateful | 0 / 0 | 40 stateful | 1.00 | 0 |
| Claude Opus 5.5 (high) | coordinate | 4 | 192 stateful, 140 read_only | 24 / 0 | 40 read_only | 1.00 | 0 |
| Claude Opus 5.5 (high) | long_coordinate | 4 | 167 stateful, 126 read_only | 4 / 0 | 44 read_only | 1.00 | 0 |
| Claude Opus 5.5 (high) | per_task_modes | 4 | 58 stateful, 53 read_only | 0 / 0 | 12 read_only | 1.00 | 0 |
| Claude Opus 5.5 (high) | per_task | 8 | 208 stateful | 0 / 0 | 24 stateful | 1.00 | 0 |
| Claude Opus 5.5 (high) | coordinate | 8 | 124 stateful, 73 read_only | 16 / 0 | 24 read_only | 1.00 | 0 |
| Claude Sonnet 5.5 | per_task | 4 | 99 stateful | 0 / 0 | 12 stateful | 1.00 | 0 |
| Claude Sonnet 5.5 | coordinate | 4 | 95 stateful, 2 stateless | 0 / 0 | 12 stateful | 1.00 | 1 |
| Claude Sonnet 5.5 | long_coordinate | 4 | 88 stateful, 1 read_only | 0 / 0 | 12 stateful | 1.00 | 0 |
| Claude Sonnet 5.5 | per_task_modes | 4 | 89 stateful, 3 read_only | 0 / 0 | 11 stateful, 1 read_only | 1.00 | 0 |
| Claude Sonnet 5.5 | per_task | 8 | 190 stateful | 0 / 0 | 24 stateful | 1.00 | 0 |
| Claude Sonnet 5.5 | coordinate | 8 | 189 stateful, 5 read_only, 2 stateless | 0 / 0 | 23 stateful, 1 read_only | 1.04 | 1 |
| Claude Sonnet 5.5 | long_coordinate | 8 | 137 stateful, 30 read_only, 1 stateless | 0 / 0 | 16 stateful, 8 read_only | 1.00 | 0 |
| Claude Sonnet 5.5 | per_task@initials | 8 | 186 stateful | 0 / 0 | 24 stateful | 1.00 | 0 |
| Claude Sonnet 5.5 | coordinate@initials | 8 | 199 stateful, 19 stateless, 6 read_only | 4 / 3 | 22 stateful, 4 stateless, 2 read_only | 1.33 | 12 |
| GPT-6 luna | per_task | 1 | 42 stateful | 0 / 0 | 5 stateful | 1.00 | 0 |
| GPT-6 luna | coordinate | 1 | 38 stateful, 7 read_only | 0 / 0 | 4 stateful, 1 read_only | 1.00 | 0 |
| GPT-6 luna | long_coordinate | 1 | 44 stateful, 8 read_only | 0 / 0 | 4 stateful, 1 read_only | 1.00 | 0 |
| GPT-6 luna | per_task | 2 | 81 stateful | 0 / 0 | 10 stateful | 1.00 | 0 |
| GPT-6 luna | coordinate | 2 | 55 stateful, 36 read_only | 0 / 0 | 7 read_only, 5 stateful | 1.00 | 0 |
| GPT-6 luna | long_coordinate | 2 | 64 stateful, 22 read_only | 0 / 0 | 6 stateful, 4 read_only | 1.00 | 0 |
| GPT-6 luna | per_task | 4 | 177 stateful | 0 / 0 | 20 stateful | 1.00 | 0 |
| GPT-6 luna | coordinate | 4 | 109 stateful, 70 read_only | 0 / 0 | 13 read_only, 8 stateful | 1.00 | 0 |
| GPT-6 luna | long_coordinate | 4 | 101 read_only, 81 stateful | 0 / 0 | 18 read_only, 7 stateful | 1.00 | 1 |
| GPT-6 luna | per_task | 8 | 355 stateful | 0 / 0 | 41 stateful | 1.00 | 0 |
| GPT-6 luna | coordinate | 8 | 227 stateful, 132 read_only | 0 / 0 | 21 read_only, 20 stateful | 1.00 | 0 |
| GPT-6 luna | long_coordinate | 8 | 169 read_only, 163 stateful | 0 / 0 | 32 read_only, 8 stateful | 1.00 | 0 |
| GPT-6 luna | per_task@initials | 8 | 211 stateful | 0 / 0 | 25 stateful | 1.00 | 0 |
| GPT-6 luna | coordinate@initials | 8 | 139 stateful, 78 read_only | 0 / 0 | 14 stateful, 10 read_only | 1.00 | 0 |

**The plan's rules for the idea winning** (1: reloads less than fresh; 2: fewer wrong answers from state than the shared interpreter; 3: accuracy no more than 5 points below the competent default, lower end of the paired interval; 4: pulls ahead, by accuracy or by agent time or cost at matched accuracy. The long arms take long_shared as both baselines.)

| model | K | arm | vs | 1 forgetting | 2 leaking | 3 holds | 4 pulls ahead | accuracy diff (points) | agent s per request diff | $ per 1,000 requests diff |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | 1 | coordinate | per_task | yes | not shown | yes | no | +0.0 [+0.0, +0.0] | +0.9 [+0.2, +1.8] | +0.12 [+0.06, +0.16] |
| Claude Haiku 5.5 | 1 | long_coordinate | long_shared | - | not shown | yes | no | +0.0 [+0.0, +0.0] | +0.3 [-0.0, +0.5] | +0.08 [+0.05, +0.12] |
| Claude Haiku 5.5 | 2 | coordinate | per_task | yes | no | yes | no | +0.0 [+0.0, +0.0] | +0.5 [+0.2, +0.9] | +0.10 [+0.06, +0.14] |
| Claude Haiku 5.5 | 2 | long_coordinate | long_shared | - | not shown | yes | no | +0.0 [+0.0, +0.0] | -0.2 [-0.8, +0.2] | +0.07 [+0.04, +0.10] |
| Claude Haiku 5.5 | 4 | coordinate | per_task | yes | yes | yes | no | +0.4 [+0.0, +1.2] | +0.2 [-0.0, +0.4] | +0.08 [+0.06, +0.11] |
| Claude Haiku 5.5 | 4 | long_coordinate | long_shared | - | not shown | yes | yes | +0.0 [+0.0, +0.0] | -0.8 [-1.3, -0.4] | +0.07 [+0.04, +0.10] |
| Claude Haiku 5.5 | 8 | coordinate | per_task | yes | yes | yes | no | +0.0 [+0.0, +0.0] | +0.4 [+0.3, +0.5] | +0.09 [+0.08, +0.10] |
| Claude Haiku 5.5 | 8 | long_coordinate | long_shared | - | not shown | yes | yes | +0.0 [+0.0, +0.0] | -1.1 [-1.4, -0.7] | +0.05 [+0.03, +0.06] |
| Claude Opus 5.5 (high) | 4 | coordinate | per_task | yes | yes | yes | no | +0.0 [+0.0, +0.0] | +0.7 [+0.3, +1.1] | +2.48 [+1.77, +3.15] |
| Claude Opus 5.5 (high) | 4 | long_coordinate | long_shared | - | not shown | yes | no | +0.0 [+0.0, +0.0] | +0.3 [+0.1, +0.5] | +2.24 [+1.64, +2.88] |
| Claude Opus 5.5 (high) | 8 | coordinate | per_task | yes | no | yes | no | +0.0 [+0.0, +0.0] | +0.6 [+0.0, +1.2] | +3.09 [+2.06, +4.15] |
| Claude Sonnet 5.5 | 4 | coordinate | per_task | yes | no | yes | no | -1.4 [-4.2, +0.0] | +1.1 [-0.6, +3.9] | +1.25 [+0.44, +2.47] |
| Claude Sonnet 5.5 | 4 | long_coordinate | long_shared | - | not shown | yes | yes | +0.0 [+0.0, +0.0] | -8.0 [-11.7, -4.5] | +1.06 [+0.76, +1.36] |
| Claude Sonnet 5.5 | 8 | coordinate | per_task | yes | no | no | no | -2.8 [-6.9, +0.0] | +2.1 [+0.2, +5.1] | +1.59 [+0.93, +2.42] |
| Claude Sonnet 5.5 | 8 | long_coordinate | long_shared | - | not shown | yes | yes | +0.0 [+0.0, +0.0] | -6.8 [-9.8, -4.0] | +0.86 [+0.44, +1.26] |
| GPT-6 luna | 1 | coordinate | per_task | yes | not shown | yes | no | +0.0 [+0.0, +0.0] | +0.3 [+0.1, +0.6] | +0.06 [+0.03, +0.09] |
| GPT-6 luna | 1 | long_coordinate | long_shared | - | not shown | yes | no | +0.0 [+0.0, +0.0] | +0.2 [-0.1, +0.4] | +0.06 [+0.03, +0.09] |
| GPT-6 luna | 2 | coordinate | per_task | yes | no | yes | no | +0.0 [+0.0, +0.0] | +0.6 [+0.1, +1.0] | +0.05 [+0.02, +0.07] |
| GPT-6 luna | 2 | long_coordinate | long_shared | - | not shown | yes | no | +0.0 [+0.0, +0.0] | -0.5 [-1.1, +0.1] | +0.01 [-0.02, +0.03] |
| GPT-6 luna | 4 | coordinate | per_task | yes | yes | yes | no | +0.0 [+0.0, +0.0] | -0.2 [-0.5, +0.2] | +0.03 [+0.01, +0.05] |
| GPT-6 luna | 4 | long_coordinate | long_shared | - | not shown | yes | yes | +0.0 [+0.0, +0.0] | -0.6 [-0.9, -0.4] | +0.03 [+0.00, +0.06] |
| GPT-6 luna | 8 | coordinate | per_task | yes | no | yes | no | +0.4 [+0.0, +1.2] | -0.2 [-0.5, +0.2] | +0.02 [+0.01, +0.04] |
| GPT-6 luna | 8 | long_coordinate | long_shared | - | not shown | yes | yes | +0.0 [+0.0, +0.0] | -1.0 [-1.3, -0.7] | +0.06 [+0.05, +0.07] |

**Top tier at K = 4 by block of seeds** (correct / requests; $ per 1,000 requests; agent seconds per request)

| seeds | fresh | shared | per_task | coordinate | long_shared | long_coordinate | per_task_modes |
|---|---|---|---|---|---|---|---|
| 1 to 3 | 72/72; 17.9; 38.3 | 69/72; 14.9; 14.3 | 72/72; 12.3; 13.5 | 72/72; 14.6; 14.1 | 72/72; 14.8; 13.5 | 72/72; 16.3; 13.7 | 72/72; 13.8; 13.9 |
| 4 to 10 | 168/168; 16.6; 37.7 | 165/168; 15.1; 15.0 | 168/168; 12.4; 13.5 | 168/168; 14.9; 14.3 | 168/168; 14.2; 13.4 | 168/168; 16.8; 13.7 | - |

**Cost from the ledgers**

- anthropic/claude-haiku-5.5: 14,747 calls, 0 errors, $3.67
- anthropic/claude-opus-4.5 (superseded, out of the analysis): 1,697 calls, 0 errors, $28.87
- anthropic/claude-opus-5.5: 5,194 calls, 0 errors, $33.38
- anthropic/claude-sonnet-5.5: 4,205 calls, 0 errors, $13.05
- openai/gpt-5.2 (superseded, out of the analysis): 1,260 calls, 0 errors, $5.99
- openai/gpt-6-luna: 8,357 calls, 0 errors, $1.20
- today's models: $51.30; superseded January-2026 models: $34.85; total: $86.16

**What the cheap tier's full design would cost on the mid and top tiers** (the six main arms at K = 1, 2, 4, 8 and per_task_modes at K = 4, ten seeds; from the measured cost per request)

| model | requests run | cost of those | full design requests | full design, estimated | the rest |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 1,368 | $10.18 | 5,640 | $41 | $31 |
| Claude Opus 5.5 (high) | 2,088 | $32.18 | 5,640 | $91 | $59 |

## Checks before trusting

- **Unit tests** (`uv run pytest`, 49 tests on the laptop; the cloud
  machine ran the same runner, agent, tools, executor and generator, checked
  by checksum, with the 48 tests it had before the analysis changes listed as
  bug 9): the mechanism (names bound in every kind of block persist; the
  last expression's value is shown; read-only leaves names, objects changed in
  place, dicts, lists, module state, environment variables, the working
  directory, its files and the random state exactly as they were, and a
  read-only call that crashes, is killed or runs out of time leaves the session
  intact; stateless sees nothing and keeps nothing; sessions are isolated;
  listing and inspection see every session and every name; invalid coordinates
  return an error with a suggestion; sessions run in the environment they
  name); every arm's tools; the agent loop; the generator's truths against an
  independent parse; and an end-to-end run of every arm with a scripted model,
  including a deliberate leak in a shared interpreter, a deliberate session-name
  collision and a what-if that damages state in place, each of which the
  measurement attributes correctly (and read-only prevents).
- **Stack probes** (Haiku 5.5 and GPT-6 luna): tool calls work through
  OpenRouter, cost is reported, and caching works (Anthropic through the
  breakpoints, OpenAI by itself).
- **Sanity condition:** every arm answered every request (42 of 42). A first
  sanity run was discarded (`runs/discarded/index.jsonl`): its data was
  generated before request 1 took its load time from the condition, so it said
  "about 20 seconds" while loads took 1 second.
- **Smoke run** (K = 2, seed 1, every arm, Haiku; whole transcripts read in
  every arm): every arm 12 of 12 except `shared`, 11 of 12. Its wrong answer is
  a textbook leak: Northgate's fourth request reused `d` and `rate`, which
  Umbra's first request had rebound in the shared interpreter (both report in
  GBP and use the same columns), and answered 116.48 instead of 115.29; the
  executor's record shows the read. `long_shared`, which sees both clients'
  code, loaded Umbra's export under a new name and did not leak. Both
  coordinate arms named one session per client and only ever chose stateful;
  neither chose read-only, stateless, listing or inspection. The what-ifs were
  all written without changing state, in every arm. `fresh` loaded the export
  13 times and took 6.8 agent minutes against about 2.2 for the others, and
  once answered a what-if from totals it had printed earlier, without code.
- **Second-family and capable-model smoke runs** (K = 2, seed 1, every arm,
  GPT-6 luna and Sonnet 5.5, transcripts read before their sweeps): Luna 12 of
  12 in every arm; its coordinate arm chose read-only for 10 of its 16 code
  calls (looks and what-ifs against a loaded session), where Haiku never did.
  Sonnet 12 of 12 in every arm but `shared`, 11 of 12 (a leak through a frame
  `mm` that the other client had rebound). Sonnet's first smoke run exposed an
  agent-loop fault, below; it was discarded and re-run after the fix.
- **Agent-loop fault, fixed** (`agent.py`): Sonnet sometimes sends
  `run_python` and a placeholder `final_answer` ("0") in one message, and the
  loop recorded the placeholder and ended the request (2 of its first 27
  requests). An answer sent alongside other calls is now not recorded and the
  model is told to answer alone after seeing the results. Haiku and Luna never
  send one (0 of their requests at any point, checked over every run), so the
  fix changes none of their runs; the Haiku sweep was already running with the
  old loop, which behaves identically for it.
- **January-2026 models' sanity and smoke runs** (Opus 4.5 and GPT-5.2,
  under Amendment 2, since superseded; kept in `runs/superseded-era-models/`
  and out of the analysis; transcripts read): sanity 42 of 42 for each. Smoke (K = 2, seed 1): every arm
  12 of 12 except `shared`, 9 of 12 with Opus 4.5 and 10 of 12 with GPT-5.2.
  In one Opus 4.5 failure, Northgate's request found its parsed column missing
  from `df` (Umbra's request had rebound it), re-parsed it, noticed
  "apostrophe thousands separators" (Umbra's format), adapted its parser to the
  other client's data and answered from it. Opus 4.5 adds derived columns to
  the loaded export itself, the in-place habit the old experiment measured;
  both January models chose read-only in the coordinate arms (for looks and
  what-ifs), where Haiku 5.5 never did. No model errors, no answers sent
  alongside code, every request answered.
- **Top tier's sanity and smoke runs** (Opus 5.5, reasoning effort high,
  sent as OpenRouter's `reasoning: {effort: high}` with 8,000 tokens a call;
  transcripts read): sanity 24 of 24 on the four headline arms; smoke (K = 2,
  seed 1) 12 of 12 in every arm but `shared`, 11 of 12 (a leak through the
  other client's `rate` dict). The model returns its reasoning but uses little
  of it on these steps (at most 73 reasoning tokens a call). In the coordinate
  arm it is the only model that listed the sessions before starting, and it
  went read-only for every what-if and every later look. $0.015 a request.
- **Top tier's main runs** (read at the hand-over, before the analysis):
  whole requests from every arm (`fresh` and `per_task` at K = 8, the
  coordinate arm's whole K = 8 run, `long_shared`, `long_coordinate` and
  `per_task_modes` at K = 4, and runs from seeds 4 to 10 as they finished),
  and every top-tier wrong answer in full (8, all in `shared`: the answering
  cell used another client's rates dict, scores frame or parsed export, in two
  cases mixing the client's own series with the other client's frame, which
  pandas aligned by row index). The arms behave as described: `long_shared`
  gives each client its own one-letter frame and a `_raw` copy;
  `long_coordinate` names one session per client and prints the export's
  first timestamp before trusting a session; in `shared`, many requests that
  read another client's `rate` or `d` noticed and rebuilt their own. No model
  errors, no unanswered request, no worker errors.
- **Measurement refinement** (found reading the first main transcripts, before
  any analysis): the worker's record of which names a cell read
  over-approximates, so a loop variable such as `c` in `for c in cols:` counted
  as a read of whatever `c` another client's request had left behind. The
  analysis now confirms every recorded read with a scope-aware reading of the
  cell's code (`reads.py`, with tests) before using it to attribute a wrong
  answer. This changes no run and nothing the model saw.

## How it is built

`src/execution_coordinates/`:

- `worker.py`: one session's process. A cell is compiled as a module and run
  with the session's namespace as globals and locals, so names bound in any
  block persist; the last expression's value is shown. Read-only forks the
  process: the child runs against a copy-on-write image of the session in a
  copy of its working directory and replies; the session is untouched. The
  worker also records which request last changed each name's value (by
  identity and content), for measurement only.
- `executor.py`: the idea. `run(code, mode, session, env)` with the three
  modes, named sessions created by a stateful call, environments (a session is
  a process started from the named interpreter), `list_sessions`, `inspect`,
  and structured errors with suggestions for invalid coordinates.
- `tools.py`: each arm's tools over the same executor. The baselines fix the
  coordinates (fresh: stateless; shared: one session; per_task: the request's
  client's session); the coordinate arms expose them.
- `agent.py`: the tool-calling loop, the same in every arm (at most 25 model
  calls a request, two nudges if it stops without answering), through
  `demos_common.LLM`, with cache breakpoints for Anthropic models.
- `data.py`: the generator (clients, exports, requests, truths, signatures,
  interleavings), checked against an independent parse.
- `runner.py`, `scoring.py`, `reads.py`, `analysis.py`: runs, correctness and
  attribution, the scope-aware read analysis, and results.

## Bugs found and fixed

In order of discovery; nothing reported below ran with a bug that could change
its numbers.

1. **Read record, conditional bindings** (measurement, caught by the
   end-to-end test before any model run): the worker counted a name bound
   inside an `if` as bound for the rest of the cell, so it missed reads of
   session state after `if 'df' not in globals(): df = ...`. Fixed, then made
   scope-aware in the analysis (item 6).
2. **Read record, augmented assignment** (measurement, caught by a unit test
   before any model run): `x += 1` was not counted as reading `x`.
3. **Provenance of equal values** (measurement, before the main runs): a name
   rebound to an equal value took the new request as its source. It now keeps
   the request that made the value.
4. **Sanity request text** (data): the first sanity data said "about 20
   seconds" while loads took 1 second. Regenerated; those 14 runs discarded
   (`runs/discarded/index.jsonl`) and re-run.
5. **An answer sent alongside code** (agent loop, every arm): Sonnet 5.5
   sometimes sent `final_answer` with a placeholder in the same message as the
   code meant to compute it, and the loop recorded the placeholder. An answer
   sent with other calls is now refused and the model asked to answer alone.
   Haiku 5.5 and GPT-6 luna never did this in any run (checked over all of
   them), so their runs, some made before the fix, are unaffected; the Sonnet
   smoke run was discarded and re-run, and every Sonnet and Opus 5.5 run used
   the fix.
6. **Read record, loop variables** (measurement, found reading the first main
   transcripts): the worker's record counts `c` in `for c in cols:` as a read
   of whatever `c` another client left behind. The analysis confirms each
   recorded read with a scope-aware reading of the code (`reads.py`) before it
   attributes a wrong answer. No run changed.
7. **A data accident** (design, found reading Sonnet's runs): the sixteen
   fictional company names have sixteen different initials, so abbreviated
   session names could not collide by initial. Tested by the added
   shared-initials check rather than by changing the planned data.
8. **Operational slips on the cloud machine**, none affecting a result: one
   launch line ran a second command from the wrong directory (the ablation did
   not start until relaunched); a wait loop's `pgrep -f` matched its own
   command line (stopped by PID); the GPT-6 luna sweep was restarted at a
   higher concurrency, discarding its 12 runs in flight (listed in
   `runs/discarded/index-vm.jsonl`).
9. **Unpaired comparisons across seed sets** (analysis, found at the hand-over,
   before any result was written up): once the top tier's K = 4 arms had seeds
   1 to 10 and `per_task_modes` seeds 1 to 3, a paired difference between them
   would have mixed seed sets, since the bootstrap resamples items that only
   one arm ran within that arm. Comparisons now use only the items (K, seed,
   client) both arms ran (`common_items`, tested). No other comparison changes,
   since every other pair ran the same items. The decision rules of the plan
   are now computed in `analysis.py` (`decisions`, tested) rather than read
   off the tables, and the spend on today's models is reported apart from the
   superseded January-2026 models.

## Caveats and threats to validity

- **The harness knew each request's task.** Every request names its client,
  and in the threads design each client's requests are one conversation, so
  the right session is obvious to the model and one interpreter per client is
  easy for the harness to build. The coordinate tool's natural niche, work
  whose task boundaries the harness cannot see (one inbox, no client labels,
  tasks that split or merge), was not tested; there `per_task` cannot be built
  and the comparison would be with `long_shared`.
- **Naming is left to the model, which is the idea's own risk.** A model in
  one client's conversation cannot see the sessions another conversation
  named unless it lists them, and only Opus 5.5 did so routinely (Sonnet 5.5
  listed or inspected sessions 7 times, all in the shared-initials check;
  Haiku 5.5 and GPT-6 luna never). The company names
  happened to have distinct initials (bug 7), so the planned data understates
  collisions; the added shared-initials check measures them.
- **Read-only had nothing to protect.** No wrong answer in any arm came from
  a client's own what-if: the models computed what-ifs on a copy or a derived
  frame. In 87 later requests across the arms that keep state, the answering
  code read a name bound during the client's what-if request (for example the
  real rates, rebuilt there after another client had rebound them), and all 87
  answers were right. So the mode's protection, which the unit tests show is
  complete, never changed an answer here; a model or a codebase that mutates
  loaded data in place (as the old experiment's Opus 4.5 code did in 28.5% of
  its steps) could differ.
- **The shared interpreter was described truthfully** ("shared by every
  request you handle, for every client"). The models defended themselves by
  reloading, which cut leaks and cost time. An undeclared shared interpreter
  would likely leak more and reload less.
- **One task family** (payments analysis on synthetic exports), requests
  handled one at a time (true concurrency would add races to a shared
  interpreter), loads simulated by real 20-second sleeps, and the environment
  axis implemented and unit-tested but never offered to the models.
- **Small samples above the cheap tier:** the mid tier has three seeds (12
  clients at K = 4, 24 at K = 8) and the top tier ten seeds at K = 4 (40
  clients) but three at K = 8 (24) and for `per_task_modes` (12), so those
  intervals are wide. The top tier's seeds 4 to 10 were added after seeds 1 to
  3 had been seen (see "Top-tier seeds after the hand-over"); the two blocks
  are reported apart in the Results.
- **Agent time** is model latency plus execution, without queueing for the
  shared rate limit; runs shared one machine, which may slow execution a
  little in every arm alike.
- **Superseded runs:** the Opus 4.5 and GPT-5.2 runs of Amendment 2 are kept
  in `runs/superseded-era-models/` and left out of every number here.

## Re-run

From `~/demos/concepts/execution-coordinates`, on the laptop or the cloud
machine (where the key comes from `~/.experiments.env` through the same
wrapper). Every command below is the one that produced the reported runs; a
complete run file is skipped and a partial one is moved to `runs/discarded/`
before it is run again.

```bash
uv sync -p 3.12 && uv run pytest                     # 49 tests, no model calls
uv run python -m execution_coordinates.data --cond sanity --seeds 1-2
uv run python -m execution_coordinates.data --cond main --seeds 1-10
uv run python -m execution_coordinates.data --cond initials --seeds 1-3
K=../../bin/with-openrouter-key
SIX=fresh,shared,per_task,coordinate,long_shared,long_coordinate

# checks: stack probe, sanity condition, smoke runs (laptop)
$K uv run python scripts/probe.py anthropic/claude-haiku-5.5 openai/gpt-6-luna
for m in anthropic/claude-haiku-5.5; do
  $K uv run python run.py --phase sanity --cond sanity --ks 1 --seeds 1-2 --arms $SIX,per_task_modes --model $m
  $K uv run python run.py --phase smoke --ks 2 --seeds 1 --arms $SIX,per_task_modes --model $m
done
$K uv run python run.py --phase smoke-luna --ks 2 --seeds 1 --arms $SIX --model openai/gpt-6-luna
$K uv run python run.py --phase smoke-sonnet --ks 2 --seeds 1 --arms $SIX,per_task_modes --model anthropic/claude-sonnet-5.5
$K uv run python run.py --phase sanity --cond sanity --ks 1 --seeds 1-2 --arms fresh,shared,per_task,coordinate \
  --model anthropic/claude-opus-5.5 --reasoning high
$K uv run python run.py --phase smoke-opus55 --ks 2 --seeds 1 --arms fresh,shared,per_task,coordinate \
  --model anthropic/claude-opus-5.5 --reasoning high

# cheap tier, full design (cloud machine; at most 12 concurrent calls in all)
$K uv run python run.py --phase main --model anthropic/claude-haiku-5.5 --arms $SIX --ks 1,2,4,8 --seeds 1-10 \
  --max-runs 36 --max-concurrency 10 --ledger runs/ledger-vm.jsonl
$K uv run python run.py --phase ablation --model anthropic/claude-haiku-5.5 --arms per_task_modes --ks 4 \
  --seeds 1-10 --max-concurrency 2 --ledger runs/ledger-vm-ablation.jsonl
$K uv run python run.py --phase family2 --model openai/gpt-6-luna --arms $SIX --ks 1,2,4,8 --seeds 1-5 \
  --max-runs 30 --max-concurrency 6 --ledger runs/ledger-vm-luna.jsonl
# mid tier
$K uv run python run.py --phase capable --model anthropic/claude-sonnet-5.5 --arms $SIX --ks 4,8 --seeds 1-3 \
  --max-runs 36 --max-concurrency 8 --ledger runs/ledger-vm-sonnet.jsonl
$K uv run python run.py --phase capable --model anthropic/claude-sonnet-5.5 --arms per_task_modes --ks 4 \
  --seeds 1-3 --max-concurrency 1 --ledger runs/ledger-vm-sonnet-ablation.jsonl
# top tier
$K uv run python run.py --phase top --model anthropic/claude-opus-5.5 --reasoning high \
  --arms fresh,shared,per_task,coordinate,per_task_modes --ks 4 --seeds 1-3 --max-concurrency 4 \
  --ledger runs/ledger-vm-opus55.jsonl
$K uv run python run.py --phase top --model anthropic/claude-opus-5.5 --reasoning high \
  --arms fresh,shared,per_task,coordinate --ks 8 --seeds 1-3 --max-concurrency 4 --ledger runs/ledger-vm-opus55.jsonl
$K uv run python run.py --phase top --model anthropic/claude-opus-5.5 --reasoning high \
  --arms long_shared,long_coordinate --ks 4 --seeds 1-3 --max-concurrency 6 --ledger runs/ledger-vm-opus55-long.jsonl
$K uv run python run.py --phase top --model anthropic/claude-opus-5.5 --reasoning high \
  --arms per_task,coordinate,shared,fresh,long_shared,long_coordinate --ks 4 --seeds 4-10 --max-runs 10 \
  --max-concurrency 10 --ledger runs/ledger-vm-opus55-s4.jsonl
# added shared-initials check (cheap and mid tiers)
for m in anthropic/claude-haiku-5.5 openai/gpt-6-luna anthropic/claude-sonnet-5.5; do
  $K uv run python run.py --phase initials --cond initials --model $m --arms coordinate,per_task --ks 8 \
    --seeds 1-3 --max-concurrency 2 --ledger runs/ledger-vm-initials.jsonl
done

scripts/sync_from_vm.sh                               # copy the machine's runs back
uv run python scripts/review_add.py runs/main         # and the other folders: the review of wrong answers
uv run python analyze.py                              # results.json and the tables
uv run python scripts/readme_tables.py                # the tables in this README
uv run python scripts/full_cost.py                    # what the full design would cost per tier
uv run python -m demos_common.ledger runs/ledger*.jsonl
```

Reading transcripts: `uv run python scripts/show.py runs/<folder>/<run>.jsonl
[--req N | --wrong]`, `scripts/brief.py runs/<folder>` and `scripts/wrong.py`.
