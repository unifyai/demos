# conversation-context

Delegated work that carries the caller's conversation, read-only, nested one
layer deeper at each handover, so it can settle what a request leaves out and
ask its caller only about what the conversation leaves open.

## Plan (written 9 October 2026, 22:20 UTC; final version 22:45 UTC, before any experiment run)

### The idea

When an agent hands work on (the conversation starts a task, the task asks a
records agent, the records agent asks a searcher), the request is written by a
model from inside a conversation, and it often leaves out what that
conversation has already settled ("what's the flight number?" when the person
said twenty minutes ago that they are off to Berlin). The idea: the loop, not
the model, passes the caller's conversation down with every call as read-only
background, and at each handover it adds the caller's own exchange beneath it,
so a searcher two or three handovers down reads the person's conversation at
the root, then the task's work, then the records agent's work, then its own
request. The callee uses that background to settle what the request leaves
out, and still asks its caller about what the background leaves open. The
model never writes the background (no tool parameter carries it), so it is
never paraphrased or shortened.

### Conditions it is built for (from its own reasoning)

1. **Delegation through two or three handovers**, with questions travelling
   back up one caller at a time: each caller answers from what it knows or asks
   its own caller, and the top agent answers from the conversation or asks the
   person.
2. **Requests written the way a busy model writes them in a long
   conversation**: the deciding detail was said many turns earlier, among
   unrelated errands, and the person's last message is terse.
3. **A person who is not always there**: work runs in the background, so a
   question that reaches the person may go unanswered. Asking costs round
   trips, and a question to an absent person gets nothing.
4. **Three kinds of request**: ones the conversation settles (the request
   leaves out a detail the conversation states), ones it leaves open (nobody
   has said which is meant, so the right move is to ask, and with nobody to
   answer, to say so), and complete ones (the request names it; the
   conversation adds only cost).

### Arms

Every arm uses the same model, the same records, the same tools, and the same
prompts apart from the mechanism. Every agent below the top has
`ask_caller(question)`; the top agent has `ask_person(question)`. Every prompt
says to ask rather than guess when a request could mean more than one thing,
and to say so and give what was found for each if nobody can tell.

| arm | what a callee receives |
|---|---|
| `alone` | its request only (agents as tools, as in common SDKs); questions bubble up |
| `brief` | its request only, but every caller is told the callee sees nothing else and to write each request so it stands alone, with every detail it may need (the usual remedy) |
| `summary` | its request plus, as read-only background, a summary written at each handover by a separate summariser call (same model) from everything the caller can see, focused on the request; rolled forward at the next handover |
| `full` (the idea) | its request plus the whole conversation as read-only background, with each caller's exchange nested beneath it, one layer per handover; framed as not addressed to it and as possibly not settling the request ("where it clearly says what is meant, go ahead; where it doesn't, handle the request as you would without it: ask rather than guess") |
| `full-naive` (ablation of the framing) | the same nested background under the old header ("Broader context (read-only) ... Resolve your request in light of this") |
| `single` (one long context) | no delegation: the top agent has the search tools itself, with the whole conversation in its context |

`summary` uses the same framing as `full`, so the two differ only in what
represents the conversation. No arm lets a model write the background into a
tool parameter; `brief` is the arm in which the model writes the context into
the request itself.

### Data (seeded; `src/conversation_context/`)

One world (`world.py`): Maya Brooks, a person in London, on Tuesday 9 June 2026,
with three record sources (emails, chats, texts and call notes; documents;
calendar) and about 360 records, searched with BM25. 48 of them belong to 24
hand-written scenarios (`scenarios.py`); the rest are filler from templates.
Each scenario has a target record and a distractor that the person's terse
question matches equally well (two flights booked by the travel desk, two
quotes from the same supplier, two codes from the same friend), each with a
distinct answer string that grading matches with patterns; the target is the
later of the two in 12 scenarios and the earlier in 12. Each scenario also has
a settling exchange (Maya states the deciding detail), a neutral exchange of
the same shape for the open kind, the terse question, a complete question, a
terse request and a complete request (for the fixed-request grids), and Maya's
answer if she is asked and there.

Conversations (`data.py`) are assembled from a pool of 282 filler exchanges
written once by `anthropic/claude-sonnet-5.5` (`scripts/make_filler.py`,
saved in `data/filler_exchanges.json`): 168 with texting-length replies and
114 with fuller replies (drafts, summaries, explanations), as a long day with
an assistant has. Any exchange that matches any scenario's `avoid` fragments
(its people, places and topic words) or is out of season was dropped, so
filler neither settles nor unsettles a scenario. Per seed, the filler and the
position of the settling exchange are re-drawn; the settled, open and complete
versions of one scenario share them. Lengths (measured):

| length | filler exchanges | messages | tokens (about) | settling exchange, messages before the question |
|---|---|---|---|---|
| short | 2 | 8 | 300 | 2 |
| medium | 24 | 56 | 2,800 | median 34 (12 to 56) |
| long (realistic) | 100 | 222 | 11,500 | median 123 (46 to 210) |
| xlong | 250 | 553 | 29,000 | median 376 (114 to 548) |

Items per condition: 24 settled, 24 open, 8 complete (every third scenario) =
56 items; three seeds (0, 1, 2), so 168 runs per arm per condition. A seed
re-draws the filler, the position and Maya's availability, and samples the
models afresh. Maya answers ("I mean ...", from the scenario) in half the runs
of each item and is away in the other half (no reply), by the parity of
scenario index plus seed, the same for every arm.

### Grids

| grid | first request | length | depth | arms | models |
|---|---|---|---|---|---|
| **G1 (headline, realistic)** | written by the top agent | long | 3 | all six | haiku; luna |
| G1-short | written by the top agent | short | 3 | all six | haiku |
| G2 (the idea's condition) | fixed terse request (leaves the detail out in settled and open items, names it in complete ones) | short, medium, long, xlong | 3 | alone, summary, full, full-naive | haiku |
| G3 (depth) | fixed terse request | long | 1, 2 | alone, full | haiku |

Depth 1: the top agent asks a searcher directly. Depth 2: top agent, task
agent, searcher. Depth 3: top agent, task agent, records agent (routes to the
three source searchers), searcher. With model-written requests the top agent's
first turn is run once per item and seed with the shared prompt and reused by
`alone`, `summary`, `full` and `full-naive`, whose top agents are identical
(the mechanism is invisible to them), so those arms are paired on the very
same request; `brief` and `single` run their own first turn.

### Measures

Per run, graded on the top agent's final reply to the person:

- **Right outcome** (primary): for settled, complete and open-with-the-person-
  present items, the reply gives the target answer; for open items with the
  person away, the reply does not commit to one candidate (it gives both, or
  says it cannot tell which and asks).
- Correct, wrong (commits to the distractor), hedged (gives both without
  settling), flagged (asks or says it cannot tell, without values), none.
- **Wrong without asking**: commits to a non-target answer and no agent asked
  anyone. **Guessed**: commits to one candidate on an open item without the
  person's answer.
- **Questions**: `ask_caller` calls at each level, where each was answered,
  `ask_person` calls, and whether the person answered.
- **Cost** (dollars per run, from each call's reported cost, cross-checked
  against the ledger), **latency** (sum of model-call latencies on the run's
  path, seconds), model calls and tokens per run.
- G1 only: whether the top agent's first request named the deciding detail
  (scenario regexes), and results split by that (valid for the four paired
  arms, whose requests are drawn from the same distribution and shared).

Grading is by exact patterns for each candidate's answer; a reply holding both
answers, or neither, goes to a judge from the other family
(`openai/gpt-6.1-sol` for haiku runs, `anthropic/claude-sonnet-5.5` for luna
runs), which sees the question, the two candidate answers and the reply, never
the arm. A sample of judge labels is checked by hand.

Statistics: rates with Wilson intervals over runs; differences between arms
paired over items (each item's seeds kept together) with a 10,000-resample
bootstrap (`demos_common.stats.bootstrap_diff`).

### Models

Primary: `anthropic/claude-haiku-5.5` for every agent and the summariser
(cheap workers, as such agents are built). Second family on the headline:
`openai/gpt-6-luna`. Default temperature and reasoning settings (a probe showed
no reasoning tokens by default for either). Max 8 model calls per agent, 4 per
clarification turn.

### Order of work

1. Unit tests with the model stubbed, for the mechanism (nesting, framing,
   hidden from the model), every arm, question bubbling, the person present
   and away, grading and the data invariants.
2. Sanity condition, where every arm should succeed: complete items, short
   conversations, depth 3, person present, seed 0, every arm.
3. Smoke run: four scenarios, settled, open and complete, long, depth 3, seed
   0, every arm, model-written and fixed requests; read whole transcripts from
   every arm. Fix any bug and re-run what it touched; discarded runs go to
   `runs/discarded/` with the reason.
4. The grids on the cloud machine, `max_concurrency` 12, `runs/` copied back
   after every batch.

Budget: aim under $50; stop before $80. Before any run, $2.17 went on writing
the filler pool. Estimated from token counts and list prices: G1 haiku $10, G1
luna $10, G1-short $4, G2 $30 (of which xlong $15), G3 $5, sanity, smoke and
judge $3: about $62 in all. Grids run in this order, with the ledger checked
after each: G1 haiku, G2 long, G1 luna, G2 short, G1-short, G3, G2 medium, G2
xlong. If the smoke run's measured cost per run projects the total above $50,
cut in this order, without looking at any result: (1) `full-naive` from G2
xlong, (2) `summary` from G2 xlong, (3) G2 medium, (4) `single` on luna,
(5) G3 depth 2.

### What counts as the idea winning

- **Headline (G1, haiku, long, depth 3)**, all items pooled on the right
  outcome: the idea wins if `full` is above each of `alone`, `brief` and
  `summary`, with the paired 95% interval above zero for each, while on open
  items its wrong-without-asking rate is not above `alone`'s (point estimate
  no higher, interval upper bound under 10 points). It ties if any interval
  includes zero; then cost and latency decide and the verdict names the cheaper
  arm. It loses if an alternative is above it with an interval excluding zero.
  "Holds" needs luna to point the same way. `single` is a reference: if it
  matches `full`, the gain is in having the conversation, not in passing it.
- **Mechanism (G2)**: settled items, `full` above `alone` or equal with fewer
  questions; open items, `full` asks about as often as `alone` and gives no
  more wrong answers without asking; complete items, equal accuracy, with the
  extra cost reported. The framing works if `full` gives fewer wrong answers
  without asking than `full-naive` on open items (interval below zero) without
  losing settled accuracy.
- **Scale (G1-short, G2, G3)**: the lengths and depths at which the gain
  appears, holds or vanishes.

## Plan revision 1 (9 October 2026, 23:40 UTC, after the smoke runs, before any grid)

Two documented faults change the plan; nothing below was decided from a result.

**What is tested.** Reading the smoke transcripts found that "settled" items were not settled
at medium and long lengths: the settling line sat about 120 messages before a question that never
pointed back to it, and every arm, including `single`, rightly treated the request as open (the
summariser wrote that the Berlin mention was "possibly related, but unconfirmed"). Each question
now refers back to what Maya said ("that conference I mentioned"), and the settling and open
lines share the referent, differing only in the deciding detail ("the Berlin conference" / "the
conference"). The fixed terse requests are unchanged, so in the fixed-request grids the request
still leaves out both the detail and the reference. The smoke also found that clarifications were
answered in a side turn the caller then forgot; questions now arrive in the caller's own history.
Both are listed under "Bugs found and fixed", with the other fixes from the smoke runs.

**Cost.** Measured on the smoke runs at the long length, a run costs about 2.5 times the estimate
(agents make 12 to 20 model calls a run, and in `full` each call carries the conversation): per
run, `alone` $0.009, `brief` $0.008, `summary` $0.012, `full` $0.029, `full_naive` $0.026,
`single` $0.004. The full plan projects to about $110 and the declared cuts leave it above the
$80 stop. Revised grids, run in this order; a grid starts only if the ledger plus that grid's
projection stays under $70, and nothing runs past $78:

| order | grid | first request | length | depth | arms | model | projected |
|---|---|---|---|---|---|---|---|
| 1 | **G1 (headline)** | written by the top agent | long | 3 | alone, brief, summary, full, single | haiku | $10 |
| 2 | **G2** | fixed terse | long | 3 | alone, summary, full, full_naive | haiku | $13 |
| 3 | G2-short | fixed terse | short | 3 | alone, summary, full, full_naive | haiku | $4 |
| 4 | **G1-luna** | written by the top agent | long | 3 | alone, brief, summary, full | luna | $9 |
| 5 | G3 | fixed terse | long | 1 | alone, full | haiku | $3 |
| 6 | G2-xlong | fixed terse | xlong | 3 | alone, full; settled and open items of the 12 odd-numbered scenarios | haiku | $7 |
| 7 | G1-short | written by the top agent | short | 3 | alone, brief, summary, full, single | haiku | $4 |

Dropped: G2 medium and G3 depth 2 (declared cuts 3 and 5); `full_naive` from G1 (the framing is
tested where the request leaves the detail out, in G2 and G2-short); `single` and `full_naive` on
luna (declared cut 4); `summary` and `full_naive` at xlong (declared cuts 1 and 2), and xlong's
complete items and half its scenarios. Before G1-luna, the sanity condition runs on luna. The
winning criteria are unchanged, except that the headline comparison no longer includes
`full_naive`. Projected total with the $11.57 already spent: about $62, over the $50 aim and under
the $80 stop.

## Plan amendment 1 (10 October 2026, 00:15 UTC, from Dan through the lead; replaced by amendment 2)

**Reason (Dan's):** smarter models need thinner harnesses, so the fair test of a design is on the
models it was built for. Model strength becomes an axis.

**What changes.**

1. **Headline: the models of the design's own time (June 2025).** The delegated work (task agent,
   records agent, searchers) runs on `openai/o4-mini`, the original lookup's default; the caller
   (the top agent, which holds the conversation, writes the first request, answers questions from
   below and replies to Maya) runs on `openai/gpt-4.1`. The second family for the delegated work is
   `anthropic/claude-sonnet-4`, with the same `gpt-4.1` caller. `single` has no delegated work, so it
   runs entirely on the caller's model. A summary or a request is written by the model of the agent
   handing over (the caller's at the first handover, the worker's below). All arms in a condition
   use the same models.
2. **Today's tier.** The runs so far on `anthropic/claude-haiku-5.5` (G1 at the long length: 840
   runs, plus the sanity and smoke runs) stay as a labelled "today's tier" condition, not the
   headline. No further grids run on haiku unless budget is left at the end.
3. **Today's frontier, if budget allows after the headline:** `anthropic/claude-opus-5.5` (reasoning
   effort high) or `openai/gpt-6.1-sol` in every role, on the headline comparison only, with fewer
   seeds if need be, to show whether any advantage shrinks with stronger models.
4. **Judges** stay strong current models from a different family from the model that writes the
   graded reply (the caller): `anthropic/claude-sonnet-5.5` for the `gpt-4.1` callers,
   `openai/gpt-6.1-sol` for haiku.
5. **Budget:** a hard stop at $80 for the whole concept, including the $21.18 spent when the
   amendment arrived. The era models cost 10 to 30 times haiku per token, and a run makes 12 to 20
   model calls, so the long (realistic) length may not fit: the headline runs at the longest length
   that does, chosen from the measured cost per run of an era smoke run before any era grid, and
   the README reports what the full size would cost.

**How it is applied.** The simulated person stays a fixed reply from the scenario's note
("Maya replied: ..."), not a model: in this design the role the old experiment gave its simulated
caller, answering questions from the conversation, is played by the caller itself (`gpt-4.1`),
and a fixed reply keeps the person's answers identical across arms. The model cap per call rises
from 1,500 to 8,000 tokens, since `o4-mini`'s reasoning counts against it (haiku hit the old cap in
3 of 23,648 calls). Before the era grids: the sanity condition and a smoke run on the era models,
with transcripts read in every arm. The era grids, in order:

| order | grid | caller | delegated work | first request | depth | arms | items |
|---|---|---|---|---|---|---|---|
| 1 | **E1 (headline)** | gpt-4.1 | o4-mini | written by the caller | 3 | alone, brief, summary, full, single | all 56, 3 seeds |
| 2 | **E2 (second family)** | gpt-4.1 | claude-sonnet-4 | written by the caller | 3 | alone, full (then brief and summary if budget allows) | all 56, 3 seeds |
| 3 | E3 (frontier, if budget allows) | opus-5.5 or gpt-6.1-sol | the same | written by the caller | 3 | alone, full | fewer seeds if need be |

E1 and E2 run at the same length: the longest of long, medium and short at which E1 and E2 (with
`alone` and `full`) are projected to fit with the judge and a $5 margin under $80. If none fits with
all 56 items, complete items are dropped first, then scenarios (the odd-numbered 12 first).

## Plan amendment 2 (10 October 2026, 00:40 UTC, from Dan through the lead; replaces amendment 1)

**Instruction.** Use only today's models that someone would reasonably choose today, on the cost
and quality frontier; no era models at all. Model strength stays an axis, across three tiers
applied to the delegated work: cheap (`anthropic/claude-haiku-5.5`, `openai/gpt-6-luna`), mid
(`anthropic/claude-sonnet-5.5`, `openai/gpt-6.1-sol`) and top (`anthropic/claude-opus-5.5`,
reasoning effort high). The caller and the simulated person stay fixed across tiers, so only the
worker's strength varies. Run the full design on the cheap tier, both families; then the headline
comparison on the mid and top tiers (one family is enough there; fewer seeds or items if need be).
Anything run on an older model since amendment 1 is kept in a labelled file and left out of the
analysis. Hard stop at $80 for the whole concept; if the mid or top tier doesn't fit at full size,
run what fits and report what the rest would cost.

**How it is applied.**

1. **Excluded:** every run on gpt-4.1, o4-mini or claude-sonnet-4 ($5.85, 10 October 00:16 to
   00:38 UTC) is in `runs/era_models_excluded/`, which `analyze.py` never reads. Spend when this
   amendment arrived: $27.03, so $52.97 remains.
2. **Fixed caller and person:** the caller (the top agent) is `anthropic/claude-haiku-5.5` in every
   tier, as in G1; Maya is the fixed reply from the scenario's note, as in G1. She is not a model,
   so she cannot vary across tiers. The two changes made for the era models are undone, so every
   tier runs the design G1 ran: the asking rule loses the sentence added for o4-mini (only 6 of
   haiku's 126 questions in G1 were of the kind it addressed), and Maya is not a model. The cap per
   call stays raised (16,000 tokens, since opus-5.5's high reasoning counts against it; haiku hit
   the old cap of 1,500 in 3 of 23,648 calls, so the cap does not separate the tiers).
3. **What G1 already shows about the premise.** In G1 the haiku caller's first request named the
   deciding detail in 68 of 71 settled items (96%), because each question points back to what Maya
   said and the caller resolves it. With the caller fixed across tiers, that rate cannot change with
   the worker, so G1 on stronger workers could not show the idea's advantage where it exists. The
   comparison across tiers is therefore made where the request leaves the detail out: the fixed
   terse request (G2), `full` against `alone`, on settled and open items. G1 stays the realistic
   headline on the cheap tier, both families.
4. **Cheap tier, both families** (haiku, and gpt-6-luna, the caller staying haiku): G1 at the long
   length (luna reuses G1's shared first turns, so both families answer the very same requests;
   `single` has no worker, so it is not repeated) and G2 at the long length with `alone`,
   `summary`, `full` and `full_naive`. The rest of plan revision 1 (G2-short, G3, G2-xlong,
   G1-short) runs only if money is left after the tiers, cheap tier first; what doesn't run is
   reported with its projected cost.
5. **Mid tier** (`claude-sonnet-5.5`; one family) and **top tier** (`claude-opus-5.5`, reasoning
   high): G2, `alone` and `full`, settled and open items. Their size (length, items, seeds) is set
   from a measured cost per run (a two-run probe per tier) before either runs, as the largest that
   fits the money left after the cheap tier with a $4 margin, reducing seeds first, then items,
   then length; the README reports what the full size would cost.

Order: cheap-tier G1 (luna), cheap-tier G2 (haiku), cheap-tier G2 (luna), mid tier, top tier,
then the rest of the cheap-tier design while money lasts. Judges as before: a strong current model
from the other family from the caller, so `openai/gpt-6.1-sol` (the caller is always haiku).

### Amendment 2: sizes, from measured costs (10 October 2026, 00:48 UTC, before any of these ran)

Probe runs (fixed terse request; s01-settled, s01-open, s18-settled, s18-open; `alone` and `full`;
seed 0; kept in `runs/probes/`, outside the analysis) measured, per run:

| workers | length | `alone` | `full` |
|---|---|---|---|
| gpt-6-luna (cheap) | long | $0.005 | $0.022 |
| claude-sonnet-5.5 (mid) | short | $0.19 | $0.24 |
| claude-opus-5.5, reasoning high (top) | short | $0.69 | $0.34 |

Spend after the probes: $32.98. At the long length a mid-tier `full` run would cost about $1 and a
top-tier one about $2 (every worker call carries the 12,000-token conversation at $2 or $4 per
million input tokens), so the tier comparison runs at the short length, where `full` costs about
what `alone` does and the request still leaves the detail out. The cheap tier runs it at the same
length, so the three tiers are compared on the same items. Grids, in order, each started only if
the ledger plus its projection stays under $76:

| order | grid | workers | length | arms | items | projected |
|---|---|---|---|---|---|---|
| 1 | G1 | gpt-6-luna | long | alone, brief, summary, full (G1's shared first turns) | all 56, 3 seeds | $9 |
| 2 | G2 | haiku | long | alone, summary, full, full_naive | all 56, 3 seeds | $13 |
| 3 | G2 | gpt-6-luna | long | alone, full | all 56, 3 seeds | $4 |
| 4 | G2-short (cheap tier of the ladder) | haiku | short | alone, full | all 56, 3 seeds | $1.5 |
| 5 | G2-short (mid tier) | claude-sonnet-5.5 | short | alone, full | settled and open of s01, s04, s07, s10, s13, s16, s19, s22; seed 0 (16 items; Maya there in 8) | $7 |
| 6 | G2-short (top tier) | claude-opus-5.5, reasoning high | short | alone, full | settled and open of s01, s04, s13, s16; seed 0 (8 items; Maya there in 4) | $8 |

If money is short at step 6, the top tier drops whole scenarios from the end of its list. If money
is left after step 6, the mid tier adds seed 1, then the top tier adds seed 1, each only if it fits
whole. Full size would cost, at the measured rates: the mid tier on G2-short with all 56 items and
3 seeds, about $72; the top tier, about $173; G2 at the long length with all 56 items and 3 seeds,
about $200 (mid) and $450 (top) for `alone` and `full` alone. Not run, for lack of money: G2-short
`summary` and `full_naive`, G3, G2-xlong, G1-short, and the mid tier's second family
(gpt-6.1-sol).

### Handover and a budget ruling (10 October 2026, 02:54 UTC, before any of the runs below)

The first agent on this concept was stopped by a usage limit at 01:45 UTC; a second agent took over
at 02:01 UTC. The `amend2` driver (`scripts/run_grids.sh amend2`, PID 215386 on the cloud machine)
was still running G2 on haiku at the long length (569 of 672 runs). It was left to finish, and it
ran the rest of amendment 2 by itself (G2 on luna, G2-short on haiku, the mid tier's seed 0, the top
tier's seed 0 and the mid tier's seed 1), ending at 02:52 UTC. Nothing was restarted or run twice.

At the handover the lead ruled that the $5.85 spent on the older models stays out of the analysis
and does not count against the $80 cap. The driver's gate counted it, so it skipped the plan's last
step, the top tier's seed 1 (ledger $72.44 + $8.50 projected is not under $76). On today's models
($66.59) that step fits whole ($75.09). `scripts/run_grids.sh amend2-extra` runs it under the same
$76 gate counted on today's models only, then, in amendment 2's order ("the rest of plan revision 1
... cheap tier first") and each only if it fits whole: G2-short's `summary` arm (projected $1.50)
and its `full_naive` arm ($1.00). Both projections come from G2-short's measured cost per run
($0.0054 `alone`, $0.0053 `full`) and G2-long's ratio of `summary` to `alone` (1.5), and were set
before either ran.

## Bugs found and fixed (before the grids)

Every fix below was made before any grid ran; discarded runs are kept in `runs/discarded/` with
the reason (`runs/discarded/README.md`). Each was found by a unit test or by reading transcripts.

1. **Judge from the same family.** `grade.judge_family` was inverted, so haiku runs would have been
   judged by claude-sonnet-5.5. Caught by a unit test; Anthropic runs are now judged by
   `openai/gpt-6.1-sol` and OpenAI runs by `anthropic/claude-sonnet-5.5`.
2. **Two scenarios hid their own ambiguity.** In s16 the target never said "contract" and in s18
   neither record said "dose", so a search for the terse request found one candidate and an agent
   could answer without knowing there was a second. Caught by a data test; the records now say
   "contract" and "dose", and the stemmer maps "vets" to "vet".
3. **Answer times in filler events.** Filler calendar events could show a calendar scenario's
   answer time; they no longer can.
4. **Filler claimed actions with no tool call behind them** (sanity runs 1 and 2). The first filler
   pool had the assistant say "Done, I've moved it" or summarise Maya's notes; top agents, which can
   see they made no such call, redid the action or retracted the summary. Every top prompt now says
   earlier turns used tools whose calls aren't shown, and the filler was rewritten (pool v2) so the
   assistant in filler never acts or looks anything up.
5. **Stale shared first turns** (sanity runs 2 to 4). Discarding runs moved the run files but not
   the cache of shared first turns, so later runs reused requests written under the old pool and
   prompt. Each cached first turn now carries a fingerprint of the model, depth, top prompt, tools
   and conversation, and the runner refuses a mismatch; the judge's cache is keyed by a hash of
   the reply and the judge prompt's version.
6. **Clarifications were forgotten by the caller** (smoke run 1). A caller answered a callee's
   question in a side turn that was then thrown away, with a note appended to the call's result.
   Its own thread never saw its answer or Maya's reply: in `alone` s01-open the top agent asked
   Maya, got "The Berlin one", and then told her it wasn't sure which flight she meant. Questions
   now arrive in the caller's own history (as the pending call's interim result, or as a message),
   the caller answers in its own thread, and the result arrives afterwards as a message.
7. **"Settled" items were not settled at long lengths** (smoke run 1). See plan revision 1:
   questions now point back to what Maya said.
8. **The judge ran out of tokens** (smoke run 1). gpt-6.1-sol sometimes spent its 20-token budget on
   reasoning and returned nothing; it now has 2,000.
9. **Stale dates in records** (sanity run 6, smoke run 2). Nadia's "party on Sunday", sent on
   Friday 5 June, meant the 7th, before today; Priya's birthday dinner was booked for a Saturday
   already past, which made the open s03 item lean to the other dinner; Hannah's and Dad's
   messages pointed to past dates; and s20's open line leaned to the garage that does the MOT.
   All fixed in `scenarios.py`.
10. **Judge definitions.** The judge labelled a reply that answered "Room 4B" and mentioned the
    other meeting as a different thing as "both". The definitions now say an answer may mention
    the other record as a different thing, and only a reply that leaves the person to choose is
    "both" (judge prompt v2).

11. **A caller's answer to a callee read later as a reply to Maya** (found reading G1 transcripts,
    10 October 00:55 UTC). When the top agent answered a question from below, the answer sat in
    its own thread as an ordinary message, and haiku sometimes wrote it as if to Maya; its final
    reply then treated it as already said ("I need to correct my last reply") and left out the
    answer. In G1 this touched 3 of 168 `alone` runs and 1 of 168 `full` runs, and no other arm. The
    harness now labels such a message in the caller's own history ("[Reply to the task agent]
    ..."), from G2 on; G1 (haiku and luna) ran without the label.

After these fixes: the sanity condition (complete requests, short conversations, depth 3, Maya
there, every arm, both request sources) gives the target answer in 80 of 80 runs, with no question
asked in 79; the smoke run (4 scenarios, long, depth 3, seed 0, every arm, both request sources)
ran without errors, and its transcripts were read in every arm.

## Results (10 October 2026, 03:20 UTC)

3,310 graded runs (every run in `runs/*.jsonl`; the discarded, older-model and probe runs are not
read). Rates have Wilson 95% intervals over runs; differences are `full` minus the other arm, paired
over items with each item's seeds kept together (10,000 bootstrap resamples). "Right" is the primary
measure: the target answer on settled, complete and open-with-Maya-there items, and not committing
to one candidate on open items with Maya away. Cost is dollars per run from each call's reported
cost; latency is the sum of the run's model-call latencies (calls run one after another). Every
number is in `results.json`; the full tables are printed by `uv run python analyze.py`.

### Verdict

**The idea holds only where its premise holds.** When the request handed down leaves out what the
conversation settled (G2, a fixed terse request), passing the conversation down raises the right
outcome over the request alone by 12 points [5, 18] at the long length and 10 [5, 16] at the short
one on haiku workers (10 [0, 19] on luna), and it does so by settling items without asking: 3% of
settled runs asked anyone, against 57 to 69% for the request alone. It ties a rolling summary
overall (+5 [-1, +10] long, +2 [-3, +6] short), beating it only on settled items at the long length
(+10 [+3, +17]). **In the realistic headline (G1), where the caller writes the request from a long
conversation, the premise mostly fails:** the haiku caller named the deciding detail in 68 of 71
settled first requests (96%), and `full` ties every alternative on both worker families (intervals
all span zero) at 1.7 to 2.9 times the cost of the request alone. By the plan's rule a tie goes to
the cheaper arm: the request alone (or the briefed request), $0.008 a run against `full`'s $0.024 on
haiku, though `full` is the fastest of the delegating arms (28 s against 35 s). **As the workers
get stronger, the gap closes:** on the fixed terse requests at the short length (settled and open
items), `full` minus `alone` is +12 [+6, +18] with haiku workers, +3 [-9, +16] with sonnet-5.5 and +0 with opus-5.5 (both 16 of 16),
because stronger workers ask about almost every ambiguous request (97% and 94% of runs) and the
caller answers from the conversation. What `full` still buys at the top tier is fewer questions
(44% of runs against 94%), less latency (93 s against 150 s) and lower cost ($0.39 against $0.48).

### Headline: realistic requests (G1: long conversation, caller-written first request, depth 3)

56 items (24 settled, 24 open, 8 complete) x 3 seeds = 168 runs per arm. The caller is haiku in both
rows; the delegated work is haiku or gpt-6-luna. `alone`, `summary` and `full` share each item's
first request (the caller's first turn is run once and reused), so they are paired on the very same
request.

| arm | right, haiku workers | right, luna workers | wrong without asking (open), haiku / luna | $ per run, haiku / luna | latency s, haiku / luna |
|---|---|---|---|---|---|
| `alone` | 77% [70, 83] | 77% [70, 83] | 8% / 7% | 0.0082 / 0.0073 | 35 / 46 |
| `brief` | 73% [66, 79] | 75% [68, 81] | 6% / 6% | 0.0082 / 0.0072 | 36 / 44 |
| `summary` | 79% [72, 84] | 80% [73, 85] | 6% / 3% | 0.0135 / 0.0103 | 57 / 59 |
| `full` (the idea) | 77% [70, 83] | 79% [72, 85] | 7% / 8% | 0.0235 / 0.0122 | 28 / 30 |
| `single` (no delegation) | 72% [65, 78] | not run | 6% | 0.0035 | 3.5 |

| `full` minus | haiku workers | luna workers |
|---|---|---|
| `alone` | +0 [-5, +5] | +2 [-5, +9] |
| `brief` | +4 [-3, +10] | +4 [-3, +12] |
| `summary` | -2 [-8, +4] | -1 [-7, +5] |
| `single` | +5 [+0, +10] | not run |

By kind (haiku workers): settled items 92 / 94 / 90 / 93 / 88% correct (`alone`, `brief`, `summary`,
`full`, `single`); complete items 100% in every arm; open items with Maya there 33 / 17 / 42 / 25 /
22%; open items with Maya away 75 / 69 / 78 / 81 / 72% right. The plan's open-item condition (`full`'s
wrong-without-asking rate on open items no higher than `alone`'s, upper bound under 10 points) holds
on haiku (-1 [-4, 0]) and narrowly fails on luna (+1 [0, +4]), but nothing separates the arms on accuracy, because the request already
carried the detail: where it named it, every arm was 94 to 99% correct on settled items; where it
didn't (3 runs in each family), `full` was right in 3 of 3 (haiku) and 2 of 3 (luna) and `alone`
in 1 of 3, too few to count. `single` (the caller searching itself, with the conversation) is the
cheapest and fastest arm, 5 points [0, 10] below `full`.

### Where the request leaves the detail out (G2: fixed terse request, depth 3)

The top agent's first request is fixed ("Find the flight number for Maya's upcoming flight."), so it
leaves out the deciding detail on settled and open items and names it on complete ones. Same 56
items x 3 seeds.

| condition | arm | right | settled correct | asked anyone (settled) | open, Maya there: correct | open, Maya away: says so | wrong without asking (open) | complete | $ per run | latency s |
|---|---|---|---|---|---|---|---|---|---|---|
| long, haiku workers | `alone` | 82% [76, 87] | 86% | 69% | 53% | 92% | 4% | 100% | 0.0086 | 39 |
| | `summary` | 89% [84, 93] | 89% | 15% | 78% | 94% | 0% | 100% | 0.0132 | 58 |
| | `full` | 94% [89, 97] | 99% | 3% | 86% | 89% | 3% | 100% | 0.0258 | 29 |
| | `full_naive` | 90% [84, 94] | 92% | 3% | 78% | 92% | 3% | 100% | 0.0265 | 30 |
| long, luna workers | `alone` | 77% [70, 83] | 78% | 38% | 47% | 92% | 1% | 96% | 0.0048 | 29 |
| | `full` | 86% [80, 91] | 92% | 0% | 78% | 78% | 6% | 96% | 0.0117 | 29 |
| short, haiku workers | `alone` | 88% [82, 92] | 92% | 57% | 64% | 94% | 0% | 100% | 0.0054 | 39 |
| | `summary` | 96% [92, 98] | 93% | 26% | 94% | 100% | 0% | 100% | 0.0084 | 56 |
| | `full` | 98% [94, 99] | 97% | 3% | 97% | 97% | 1% | 100% | 0.0053 | 29 |

| `full` minus | right | settled right | open right | asked anyone (settled) | wrong without asking (open) | $ per run | latency s |
|---|---|---|---|---|---|---|---|
| `alone`, long, haiku | +12 [+5, +18] | +12 [+4, +22] | +15 [+4, +28] | -67 [-81, -53] | -1 [-6, +3] | +0.017 | -10 [-13, -7] |
| `summary`, long, haiku | +5 [-1, +10] | +10 [+3, +17] | +1 [-8, +11] | -12 [-22, -4] | +3 [0, +7] | +0.013 | -29 [-34, -25] |
| `full_naive`, long, haiku | +4 [-1, +9] | +7 [+1, +14] | +3 [-7, +12] | 0 | 0 [0, 0] | -0.001 | -1 [-3, +1] |
| `alone`, long, luna | +10 [0, +19] | +14 [-1, +29] | +8 [-7, +24] | -38 [-50, -25] | +4 [-3, +14] | +0.007 | 0 [-6, +7] |
| `alone`, short, haiku | +10 [+5, +16] | +6 [-4, +14] | +18 [+10, +26] | -54 [-68, -39] | +1 [0, +4] | -0.000 | -10 [-13, -6] |
| `summary`, short, haiku | +2 [-3, +6] | +4 [-4, +12] | 0 [-6, +6] | -24 [-38, -11] | +1 [0, +4] | -0.003 | -27 [-32, -23] |

Against the plan's mechanism criteria: on settled items `full` is above `alone` (long) or equal
(short) and asks far less, as required. On open items it asks about as often as `alone` (78% against
64% of runs at the long length, 89% against 65% at the short) and gives no more wrong answers
without asking on haiku; on luna it does (6% against 1%, +4 [-3, +14]; luna with the conversation in
hand committed to one candidate in 7 of 36 open runs with Maya away, 5 of them to the target,
against 3 of 36 for `alone`). Complete items are
equally right, at 3 times `alone`'s cost at the long length and the same cost at the short one. The
framing criterion is not met: `full_naive` ("Resolve your request in light of this") gave no more
wrong answers without asking than the careful framing (3% each), so the framing's protection
against guessing is not shown; the careful framing made settled answers more decisive instead
(+7 [+1, +14] on settled items, where `full_naive` more often listed both). `full`'s gain on open
items with Maya there comes from where the question is asked: its workers see that the request is
ambiguous and ask, and the caller asks Maya mid-task; in `alone` the workers see an unambiguous
request and return both candidates, and the caller then asked Maya in its final reply in 30 of
`alone`'s 49 misses on this subset across the three G2 conditions (`full`: 7 of 14), which the plan
grades as not right (see the caveats).

### Model strength (G2 at the short length, settled and open items)

The caller (haiku) and Maya (a fixed reply) stay the same; only the delegated work moves up the
tiers. The mid and top tiers ran a subset (mid: 8 scenarios, seeds 0 and 1; top: 4 scenarios, seeds
0 and 1; Maya there in half), so each is shown beside haiku on exactly the same items and seeds.

| workers | runs per arm | `alone` | `full` | `full` minus `alone` | runs that asked anyone, `alone` / `full` | $ per run, `alone` / `full` | latency s, `alone` / `full` |
|---|---|---|---|---|---|---|---|
| haiku-5.5 (cheap), all 48 items | 144 | 85% [79, 90] | 97% [93, 99] | +12 [+6, +18] | 61% / 46% | 0.0058 / 0.0055 | 41 / 30 |
| sonnet-5.5 (mid) | 32 | 91% [76, 97] | 94% [80, 98] | +3 [-9, +16] | 97% / 44% | 0.131 / 0.150 | 72 / 57 |
| haiku-5.5 on the mid tier's items | 32 | 91% [76, 97] | 100% [89, 100] | +9 [0, +19] | 66% / 53% | 0.0066 / 0.0057 | 46 / 31 |
| opus-5.5, reasoning high (top) | 16 | 100% [81, 100] | 100% [81, 100] | +0 | 94% / 44% | 0.484 / 0.394 | 150 / 93 |
| haiku-5.5 on the top tier's items | 16 | 94% [72, 99] | 100% [81, 100] | +6 [0, +19] | 75% / 50% | 0.0073 / 0.0060 | 51 / 33 |

With stronger workers the request alone catches up: they ask about nearly every ambiguous request
and the caller answers from the conversation. `full`'s advantage moves from accuracy to effort: at
the top tier `alone`'s agents made 41 model calls a run against `full`'s 24, and in 10 of 16 `alone`
runs a search agent hit the 8-step cap (1 of 16 in `full`), still finding the answer.

### Cost (from the ledger)

Ledger total $80.90: **$75.05 on today's models** and $5.85 on the older models of amendment 1
(excluded from the analysis and, by the lead's ruling, from the cap). By phase: G2-short on opus
workers $14.04, G2 long on haiku $12.45, G2-short on sonnet $8.98, G1 on haiku $8.97, sanity and smoke
runs (all rounds, including discarded ones) $7.60, cost probes $5.95, G1 on luna $5.34, the filler
pools $3.75, G2-short on haiku $3.21, G2 long on luna $2.77, the judge $1.78, and $0.19 on the G2 run
stopped by amendment 1 (`analyze.py` prints this list; `results.json` holds it as `cost_by_phase`).
Over the brief's $50 aim and under the $80 stop.

### The judge, checked by hand

Replies holding both answers or neither (1,942 of 3,310) were labelled by a judge from the other
family (gpt-6.1-sol, since the caller is always haiku), blind to the arm. 44 judged replies were
read by hand (5 per label from G1 and up to 4 per label from G2, stratified): 37 agree, 6 are borderline
and 1 is wrong. The wrong one (G1, s11-open) led with the target time and was labelled as the
distractor. All 6 borderline cases are one pattern: the reply lists both candidates and then leans
to one ("since you mentioned Jake, the tutoring rate is probably the one you want"), which the judge
labels as committing and a reader might call leaving the choice. That pattern decides the grade only
on open items with Maya away.

### Caveats and threats to validity

1. **The realistic condition may be easier for the caller than real life.** Plan revision 1 made
   every question point back to what Maya said ("that conference I mentioned") so that settled
   items are settled at long lengths; the same pointer lets the caller resolve the reference and
   write it into the request (96%). Real requests may leave details out more often; G2 is the
   other extreme, where the request always leaves them out. The truth for a given deployment lies
   between, and the gain scales with how often the caller's request leaves the detail out.
2. **Mid and top tiers are small** (32 and 16 runs per arm, one family, the short length only;
   their intervals are wide). The long length would cost about $1 (mid) and $2 (top) per `full`
   run.
3. **Asking in the final reply counts as not right when Maya is there.** The plan grades an open
   item with Maya there as right only if the reply gives the target, which needs `ask_person`
   mid-task. `alone`'s caller, handed both candidates from below, asked Maya in its final reply
   instead in 30 of its 49 misses there, which in a live chat costs a round trip rather than failing. This inflates
   `full`'s gain on that subset (+33 points at the short length); settled items, where no such
   question is needed, show the gain without it (+12 [+4, +22] at the long length).
4. **Some open items lean.** In s03 only the target record says Priya "booked" the dinner; in s04
   "lending us her place" fits the cabin message and "this week" fits the building code. Every arm
   sees the records, but only the conversation arms see the open line, so they can lean on it.
5. **One caller, fixed across tiers** (haiku): when it guesses in answering a worker's question
   (opus `alone` s04-open answered "the front door code ... fits 'this week'" without asking Maya),
   the worker's strength can't undo it, though here an opus task agent noticed the guess and
   returned both.
6. **G1 ran before bug 11 was fixed** (a caller's answer to a worker later read as said to Maya);
   it touched 3 of 168 `alone` runs and 1 of 168 `full` runs in G1 on haiku.
7. **Opus at "high" effort reasoned little on these steps** (median 19 reasoning tokens per call,
   at most 181; reasoning reaches it through OpenRouter's effort control), and the 8-step cap per
   agent stopped opus searchers in `alone` runs.
8. **A haiku failure mode, rare in every arm:** records agents sometimes refused to look up a Wi-Fi
   password or code without Maya's own authorisation, and the refusal climbed the chain (2 of 168
   `summary` runs in G2 long, 3 runs in G1 on luna).
9. **One synthetic world**: 24 hand-written scenarios, about 360 records, BM25 search, one
   filler pool written by sonnet-5.5. Latency is the sum of model-call latencies, not wall time
   with parallel calls.

### Not run, and what it would cost

At the measured costs per run: G2-short `full_naive` (gated out at $75.05 + $1.00, about $1); G3
at depth 1 (about $3); G2 at xlong with `alone` and `full` (about $7); G1-short (about $4); so about
$15 for the rest of the cheap-tier design. The mid tier's second family (gpt-6.1-sol, same 64 runs,
about $9). The tiers at G2-short's full size (all 56 items, 3 seeds, `alone` and `full`): mid about
$47 in all ($0.14 a run; $38 beyond what ran), top about $147 ($0.44 a run; $133 beyond). G2 at the
long length on the mid and top tiers: about $200 and $450 (plan amendment 2's estimate).

### Found after the handover

No bug in any arm. Transcripts were read in every arm of every condition (G1 on luna: `alone`,
`brief`, `summary`, `full`; G2: all four arms on haiku, both on luna, `alone`, `summary` and `full`
at the short length; the mid and top tiers, both arms). Two changes to the analysis, neither
touching a result: `analyze.py` skips a line caught mid-write by a copy from the cloud machine
(none remained in the final copy), and it reports today's-model spend separately from the older
models' (`spend_today_models`, `spend_older_models`).

## Re-running

Everything runs from this folder. Tests and data first (no model calls):

```bash
uv sync && uv run pytest -q                       # 138 tests, the model stubbed
```

The filler pool is written once by a model and saved; re-running `scripts/make_filler.py` gives a
different pool, so the experiment reads `data/filler_exchanges.json` (pool v2):

```bash
../../bin/with-openrouter-key uv run python scripts/make_filler.py            # batch 1
../../bin/with-openrouter-key uv run python scripts/make_filler.py --fuller   # batch 2, appended
```

Sanity and smoke (laptop or cloud machine), then the grids on the cloud machine, one process at a
time, each followed by the judge (`scripts/run_grids.sh`, which also holds the budget gate):

```bash
K=../../bin/with-openrouter-key
$K uv run python run.py --grid sanity --requests model --length short --depth 3 --kinds complete --seeds 0 --present all
$K uv run python run.py --grid sanity --requests terse --length short --depth 3 --kinds complete --seeds 0 --present all --arms alone,summary,full,full_naive
$K uv run python run.py --grid smoke --requests model --length long --depth 3 --scenarios s01,s07,s16,s18 --seeds 0
$K uv run python run.py --grid smoke --requests terse --length long --depth 3 --scenarios s01,s07,s16,s18 --seeds 0 --arms alone,summary,full,full_naive
scripts/run_grids.sh part1    # G1 on haiku (the rest of part1 was stopped by amendment 1)
scripts/run_grids.sh amend2   # amendment 2: cheap tier (luna G1, G2 long on haiku and luna, G2-short), mid and top tiers
scripts/run_grids.sh amend2-extra   # after the handover: the top tier's seed 1, G2-short summary (full_naive gated out)
$K uv run python judge.py     # labels for replies the patterns can't settle (cached, keyed by reply)
uv run python analyze.py      # results.json and the tables in Results
uv run python scripts/judge_sample.py 5 11   # judged replies to check by hand, 5 per label
uv run python show.py runs/<condition>.jsonl <arm> <item> <seed>   # one whole transcript
```

`run.py` skips runs already recorded, so a stopped grid resumes. To run a grid afresh, move its
`runs/<condition>*.jsonl` files aside, including `__prefixes.jsonl` (the runner refuses cached first
turns that don't match, but a stale prefix file would still be read).

`runs/` is not in git; `runs.tar.zst` is its archived copy. Restore it with
`zstd -dc runs.tar.zst | tar -xf -` and rebuild it after any new run with
`tar -cf - runs | zstd -19 -T0 -o runs.tar.zst`.
