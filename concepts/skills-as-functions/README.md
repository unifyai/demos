# skills-as-functions

An agent that keeps what it learns as callable, tested functions in a library
does better than one that keeps prose notes, as tasks recur, build on each
other and the environment changes.

## Plan (written 2026-10-09 15:15 PDT, before any experiment run)

### The idea

An agent that works for someone over a long period meets the same kinds of
task again and again, with new arguments, and later tasks are built from the
steps of earlier ones. If the agent keeps each procedure it has got right as a
Python function in a library (it sees every function's signature and
docstring, the functions are defined in its namespace so it can call them, a
new task can be written as calls to them, and redefining one replaces it for
every caller) then a recurring task should cost one short call instead of a
fresh derivation, a composite task should be written from parts that already
work, and when the environment changes, one repair to the shared step should
fix every stored task above it. Notes, by contrast, have to be re-read and
re-typed into code on every task, cannot be run or checked on their own, and
go stale silently.

### Conditions it is built for

From the idea's own reasoning: (1) a long stream of tasks against one
environment; (2) kinds of task that recur with new arguments, and some that
recur word for word; (3) tasks that compose the steps of earlier kinds, in
levels; (4) shared low-level steps used by many kinds; (5) procedures with
several steps and rules that are learned by doing (pagination, record shapes,
business rules that surface as errors), so there is something worth keeping;
(6) one or more interface changes mid-stream that break a shared step.

### Environment and data

A simulated customer workspace with an API of about 21 operations (contacts,
companies, users, support tickets, orders and refunds, email), generated per
seed by `src/skills/world.py`: about 80 contacts, 12 companies, 8 users, 180
tickets and 320 orders at the realistic size. List operations are paginated (5
per page). Some rules are documented only by the errors that enforce them
(refunds over 10,000 cents need an approval code; replying to a ticket bumps
its version, so an update with the old version gets a 409). Every task's
outcome is checked exactly: the submitted answer against a reference computed
from the data, and the workspace state afterwards (emails sent, tickets
changed, refunds, tags) against the state the reference leaves. Before each
task the workspace is reset to the reference state at that point in the
stream, so one arm's mistake never changes what a later task expects.

Tasks come from 16 kinds in three levels (`src/skills/tasks.py`):
- level 1, lookups: a contact's email, a contact's company, a company's account
  manager, a contact's open-ticket count, a contact's latest order status;
- level 2, lists, sums and actions: a company's spend on delivered orders since
  a date, an agent's open tickets in priority order, emailing a contact,
  reassigning one agent's open tickets to another, tagging a company's contacts
  in one city, refunding a contact's latest delivered order;
- level 3, compositions of the above: emailing every contact at a company with
  an open ticket, escalating a company's urgent tickets (reply and reassign),
  refunding and then emailing an apology, naming a company's top spender with
  their email, emailing a company's account manager its open ticket IDs.

A composite kind first appears after the kinds it is built from. About one task
in ten repeats an earlier task word for word. Two interface changes break
shared steps: API v2 replaces `search_contacts` and `list_company_contacts`
with one cursor-paginated `find_contacts` (the old ones return 410 Gone), and
API v3 replaces `send_email` with `create_draft` then `send_draft`. The API
reference every arm sees is always the current one.

Sizes (a sweep of stream length, number of kinds and API surface; the kind
schedule of each size is fixed, and seeds vary the data and arguments):

| size | tasks | kinds | operations | changes |
|---|---|---|---|---|
| sanity | 8 | 4 level-1 lookups | 9 | none |
| small | 24 | 5 (contacts, tickets, email) | 12 | v2 before task 13 |
| medium | 48 | 10 (no orders) | 16 | v2 before task 17, v3 before task 33 |
| realistic | 96 | 16 | 21 | v2 before task 33, v3 before task 65 |

### Arms

Every arm uses the same agent loop, model, system prompt, API reference, task
text, limits and outcome signal; they differ only in what is kept between
tasks. The loop: the model calls one tool, `run_python(code)`, whose code runs
in a fresh namespace per task holding the API operations, a few preloaded
modules and `submit(answer)`, which ends the task; up to 10 calls per task.
After each task every arm is told whether the outcome checked correct (as a
user confirming or rejecting a result), and uses that only as below.

1. **none**: nothing kept.
2. **notes**: one notes document, rewritten by the model after every task from
   the old notes, the task's code and outputs and its outcome (at most 1,500
   words; it may include code snippets; it should correct what proved wrong or
   out of date), shown at the start of every task. This is the curated
   playbook a competent engineer would build today (Dynamic Cheatsheet, ACE,
   an AGENTS.md kept by the agent).
3. **retrieval**: the code of every task that checked correct is stored; each
   new task is shown the three most similar stored tasks (BM25 over the task
   text, most recent first among ties) with their code, as worked examples.
4. **fullctx**: one continuous conversation over the whole stream; every
   earlier task's full exchange (code, outputs, outcome) stays in context, with
   prompt caching. The namespace is still fresh per task, so earlier code can
   be read and copied but not called.
5. **library** (the idea): every function defined in a task that checked
   correct is stored in a library; the signatures and docstrings of all
   library functions are shown at the start of every task, and the functions
   are already defined in the namespace, so code can call them and compose
   them. Redefining a library function in a task that checks correct replaces
   it for every caller (a repair). When a library function appears in a
   traceback, its source is shown with the error. Functions that reference
   names outside their arguments, the API, the builtins, the preloaded modules
   and the library are not stored, so a stored function always runs.

The library is never run without a model call: a task that repeats a stored
task word for word still goes to the model (which may answer with one call to
the stored function). An exact-match cache could be added to any arm that
stores code, and is not what is tested.

### Measures

Per task: correct or not (exact check); model calls; prompt, cached and
completion tokens; cost as OpenRouter reports it in the ledger; latency (the
sum of the task's model-call times); for the notes arm the note-writing call
is counted with its task and also reported separately. Then, by arm and size:
accuracy, mean cost, latency, calls and tokens per task; the same by stream
position (blocks of 8 tasks), by level, for first occurrences against
recurrences, and for word-for-word repeats; failed tasks in the 8 tasks after
each interface change; and for the library its size, the share of tasks that
call a stored function, repairs in place against write-arounds after a change,
and stored functions that later failed.

Comparisons are paired by task position (the schedule is the same across
seeds): the difference between the library and each other arm with a 95%
bootstrap interval over positions, seeds kept together
(`demos_common.stats.bootstrap_diff`); proportions with Wilson intervals.

### Models, seeds, budget

Primary: `anthropic/claude-haiku-5.5`, the cheap fast model one would run a
long stream of routine work on, at temperature 0, every size and arm, seeds
1, 2 and 3. Second family: `openai/gpt-6-luna` on the realistic size, every
arm, seeds 1 to 3. If budget remains: seeds 4 and 5 at the realistic size on
the primary model, then one seed of the realistic size on
`anthropic/claude-sonnet-5.5` as an exploratory check of a stronger model.
Estimated spend $20 to $35; stop and report before $80.

### Order of work

Unit tests with the model stubbed (the API in every version, reference
solutions against scripted API solutions over whole streams, the sandbox, each
arm's prompt and memory update, library storage and repair); the sanity
condition on every arm; a smoke run of the small size, one seed, reading whole
transcripts from every arm; then the full runs on the cloud machine.

### What counts as the idea winning

At a given size, the library wins against an arm if it is more accurate (95%
interval of the difference above zero), or at least as accurate (interval
above minus 5 points) and cheaper per task (interval of the cost difference
below zero). The idea holds at that size if it wins against every other arm,
and its claim about changes holds if, pooled over the changes and seeds, it
fails fewer of the 8 tasks after a change than notes, retrieval and fullctx.
The idea's reasoning predicts that the advantage grows with size and with
stream position, and is largest on composite tasks and recurrences. If the
library is less accurate than any other arm at the realistic size (interval
below zero), the idea does not hold there, whatever it saves.

### Plan refinements before the first experiment run (2026-10-09 15:40 PDT)

Made while building, from the sanity condition and two probes of the library
arm on the 8-task sanity stream (`runs/discarded/`, with reasons), before any
experiment run:

1. **The library arm saves functions in a step of its own.** With the
   instructions alone (two wordings, the second telling it to do every task by
   calling a function and never to submit a value copied from output), the
   model explored with print() and submitted the value it saw in all 16 probe
   tasks, and defined no function, so the library stayed empty. An explore-then-
   submit agent does not write functions unless something asks for them at the
   right moment. So, after a task checks correct, the library arm continues the
   task's conversation with one request (up to two attempts): define a function
   for this kind of task, with the task's values as arguments, built from
   library functions where they fit (repairing any that failed during the task),
   and end with `save(fn, <this task's values>)`. The call is re-run in a fresh
   namespace, with the library, on the workspace as it was before the task, and
   the functions that ran are stored only if the re-run gives the checked
   answer and makes the checked changes. The step is skipped when the cell that
   submitted ran a library function, no function was defined and the model's
   own code changed nothing. This is the library's counterpart of the notes
   arm's rewrite after every task, and it makes "tested" literal (as in
   skill induction with verification, Wang et al. 2025). The in-task
   instructions say to call library functions where they fit and to repair one
   that fails by redefining it.
2. **Latency** is reported as the agent loop's model time per task (what the
   person waits for: the answer is in hand at submit). The notes rewrite and the
   library save come after the answer, so their time is reported separately;
   cost, calls and tokens include them.
3. Operation counts as built: 16 at the medium size and 21 at the realistic
   size (planned as 17 and 22).

Three faults found by reading the smoke runs' transcripts (16:00-16:12 PDT),
fixed for every arm before any experiment run (details in
`runs/discarded/README.md`):

4. The notes rewrite was capped at 4,096 output tokens and 20 rewrites hit
   the cap, so the notes were cut off. The cap is now 8,192, and a rewrite that
   still hits it is asked for again, shorter.
5. Every failure in the first smoke runs was the model using its 10 calls
   without submitting. Each tool result now ends with the number of calls left.
6. With a tool call required on every turn and `submit()` only callable inside
   code, a model that thought it was done could not end its turn, and looped,
   printing summaries or re-running its last call. There is now a `submit` tool
   beside `run_python` (`submit()` in code still works).

### Amendment 1 (2026-10-09 17:15 PDT, from the lead, for Dan; superseded at 17:35 by amendment 2)

Reason, as given: smarter models need thinner harnesses, so the fair test of a
design is on the models it was built for. Model strength becomes an axis:

1. **Headline: the models of the design's own time (May 2025).**
   `openai/gpt-4.1` is the primary model and `anthropic/claude-sonnet-4` the
   second family, every arm with the same model.
2. **Today's cheap tier.** The runs made so far on `anthropic/claude-haiku-5.5`
   and `openai/gpt-6-luna` stay, as a labelled "today's cheap tier" condition,
   not the headline. The cheap tier's extra seeds 4 and 5 were stopped part
   way when the plan changed (`runs/discarded/`), so it has seeds 1 to 3; the
   exploratory `claude-sonnet-5.5` run was never started.
3. **Today's frontier, if budget allows after the headline:**
   `anthropic/claude-opus-5.5` with reasoning effort high, or
   `openai/gpt-6.1-sol`, on the headline comparison only, with fewer seeds if
   need be, to show whether the library's advantage shrinks with stronger
   models.
4. **Budget:** a hard stop at $80 for the whole concept, spend so far
   included ($6.11 when the amendment arrived). The era models cost 20 to 30
   times as much per token as the cheap tier. If the realistic size does not
   fit for the headline, the largest size that does is run, and the README
   reports what the realistic size would have cost.

Everything else in the plan stands: the same arms, harness, data, seeds,
measures and criterion.

How it was applied (17:25 PDT, before any era-model experiment run):

- Both era models passed the sanity condition, every arm 8 of 8
  (`runs/sanity/sanity/gpt-4.1/`, `runs/sanity/sanity/claude-sonnet-4/`), and
  their transcripts read as the cheap tier's did.
- **The headline runs at the medium size** (48 tasks, 10 kinds, two
  changes), seeds 1 to 3, every arm. Scaling the cheap tier's per-task cost
  by each model's cost on the sanity condition, three seeds of the realistic
  size would cost about $44 on gpt-4.1 and $70 on claude-sonnet-4 ($114
  together) against about $71 left, and the medium size about $19 and $30.
- **Frontier:** `openai/gpt-6.1-sol` at its default reasoning, on the medium
  size, with as many seeds as the budget leaves. It is preferred to
  claude-opus-5.5 with high reasoning because a probe of that model rejected
  a forced tool call (HTTP 400: tool_choice "any" is not supported for this
  model), which every arm's loop makes on every turn, and because it costs
  less.

### Amendment 2 (2026-10-09 17:35 PDT, from the lead, for Dan; supersedes amendment 1)

Use only today's models that someone would reasonably choose today, on the
cost and quality frontier, and no era models. Model strength stays an axis,
across three tiers:

- **cheap**: `anthropic/claude-haiku-5.5` and `openai/gpt-6-luna`. The full
  design (every size, every arm, seeds 1 to 3) runs on both families.
- **mid**: `anthropic/claude-sonnet-5.5` or `openai/gpt-6.1-sol` (one family is
  enough), and **top**: `anthropic/claude-opus-5.5` with reasoning effort high.
  These run the headline comparison only, with fewer seeds or items if need
  be, so the results show how the library's advantage changes as models get
  stronger.

Every arm in a condition uses the same model. What was run on era models under
amendment 1 (the sanity condition on gpt-4.1 and claude-sonnet-4, 92 tasks of
gpt-4.1 at the medium size, and probes) is kept in
`runs/discarded/era-models-amendment-1/` and left out of the analysis. Budget:
a hard stop at $80 for the whole concept, spend so far included (about $11
when this amendment arrived); if the mid or top tier does not fit at full
size, what fits is run and the README reports what the rest would cost.

How amendment 2 was applied (17:55 PDT, before any mid- or top-tier experiment run):

- **Cheap tier, full design, both families.** claude-haiku-5.5 has every size
  (seeds 1 to 3). gpt-6-luna is run at every size too. Its first realistic run
  was discarded: reading the era models' transcripts showed that the harness
  did not display a cell's final bare expression, as a notebook does, and
  gpt-6-luna had relied on that in 6.4% of its cells (claude-haiku-5.5 in 2 of
  5,092), wasting calls. The harness now shows that value (fault 7 below), and
  every gpt-6-luna size is run with it.
- **Mid tier: `anthropic/claude-sonnet-5.5`; top tier:
  `anthropic/claude-opus-5.5` with reasoning effort high.** One family across
  all three tiers. Both reject a forced tool call (HTTP 400, as probed), which
  the loop makes on every turn, so both run with `tool_choice="auto"`: the
  model may reply in text, and is then told to use a tool or submit (a call
  counted like any other). So that the change of harness is not mistaken for
  the change of model, **claude-haiku-5.5 is also run with `"auto"`** at the
  same size (a cheap control). Opus gets 12,000 extra output tokens per call
  for its reasoning.
- **Size: the medium size (48 tasks, both changes), every arm**: seeds 1 to 3
  on claude-sonnet-5.5 and on the claude-haiku-5.5 control, seeds 1 and 2 on
  claude-opus-5.5. From each model's cost on the sanity condition (every arm
  8 of 8 on both), scaled by the cheap tier's cost per task: about $18 for
  the mid tier and $30 for the top tier, against about $67 left. The
  realistic size would cost about $44 (mid, three seeds) and $114 (top, three
  seeds), so it does not fit; neither do three top-tier seeds at the medium
  size ($45) with a safe margin.
- Update (18:33 PDT): measured on its first 100 tasks, the top tier cost about
  $7 a seed, not $15, so a third seed fits (projected total about $55). Its
  run was restarted with seeds 1 to 3 and more concurrency; the first run's
  unfinished streams were set aside by `run.py` in `runs/discarded/tiers/`.
- Update (19:20 PDT): that measure was taken before the notes arm's rewrites
  grew long. The top tier in fact cost $38.19 for its three seeds (about
  $12.70 a seed, $26.27 of it the notes arm, whose rewrites with high
  reasoning reached about $0.20 a task), close to the first estimate of $15 a
  seed. It finished at 19:18 PDT within the cap: today's models cost $73.77
  in all.

## Results (written 2026-10-09 19:30 PDT, after every planned run)

### Verdict

**The idea does not hold, at any size or model tier tested.** By the plan's
criterion the library has to beat every other arm, and in none of the nine
conditions does it: it is never more accurate than no memory, retrieval of
past solutions or the long context, and never cheaper than retrieval. It wins
against the notes arm everywhere, by cost, and against no memory or the long
context in some conditions, also by cost (no memory at the small and medium
sizes on gpt-6-luna; the long context at the realistic size, at the medium
size on gpt-6-luna and in the haiku-5.5 control). At the realistic size on
claude-haiku-5.5 it is *less* accurate than the long context (-2.1 points
[-4.2, -0.3]), which the plan says rules it out there. The claim about
interface changes fails in every condition: the library did not fail fewer of
the tasks after a change than notes, retrieval and the long context, because
no arm had trouble there, so one repair to a shared function had nothing to
win. The predicted growth with size and stream position does not appear. On
the mid and top tiers every arm got every task right, and the library cost
about 60% more than no memory. The mechanism itself worked as designed (see
below); the strong simple baselines left it nothing to win.

### Headline tables

Accuracy (% of tasks correct; tasks x 3 seeds), cost ($ per 1,000 tasks) and agent latency (s per task), by arm:

| condition | none | notes | retrieval | fullctx | library |
|---|---|---|---|---|---|
| cheap: haiku-5.5, small | 100.0%, $0.46, 4.4 s | 100.0%, $4.34, 4.4 s | 100.0%, $0.45, 3.6 s | 100.0%, $0.50, 3.1 s | 98.6%, $0.55, 3.5 s |
| cheap: haiku-5.5, medium | 98.6%, $0.60, 5.0 s | 98.6%, $6.26, 4.9 s | 97.2%, $0.59, 4.1 s | 97.2%, $0.78, 3.5 s | 99.3%, $0.64, 3.8 s |
| cheap: haiku-5.5, realistic | 97.9%, $0.51, 4.7 s | 98.3%, $8.49, 4.6 s | 99.3%, $0.53, 3.8 s | 100.0%, $1.13, 3.4 s | 97.9%, $0.58, 3.6 s |
| cheap: gpt-6-luna, small | 100.0%, $0.47, 5.3 s | 100.0%, $1.38, 4.7 s | 100.0%, $0.35, 3.7 s | 100.0%, $0.39, 5.0 s | 100.0%, $0.34, 3.5 s |
| cheap: gpt-6-luna, medium | 99.3%, $0.53, 6.8 s | 100.0%, $2.08, 5.8 s | 98.6%, $0.40, 4.0 s | 100.0%, $0.75, 6.8 s | 99.3%, $0.36, 3.6 s |
| cheap: gpt-6-luna, realistic | 100.0%, $0.33, 6.5 s | 98.6%, $2.58, 6.3 s | 99.3%, $0.27, 3.1 s | 100.0%, $0.98, 6.9 s | 99.0%, $0.32, 3.4 s |
| cheap, auto: haiku-5.5, medium | 99.3%, $0.51, 4.9 s | 94.4%, $6.00, 6.4 s | 100.0%, $0.60, 4.2 s | 96.5%, $0.90, 4.1 s | 100.0%, $0.70, 4.7 s |
| mid, auto: sonnet-5.5, medium | 100.0%, $6.27, 5.0 s | 100.0%, $88.72, 5.0 s | 100.0%, $7.77, 4.6 s | 100.0%, $10.69, 4.7 s | 100.0%, $9.90, 6.1 s |
| top, auto: opus-5.5 high, medium | 100.0%, $14.52, 7.3 s | 100.0%, $182.40, 7.3 s | 100.0%, $19.19, 7.0 s | 100.0%, $25.94, 5.9 s | 100.0%, $23.18, 7.7 s |

The library minus each arm, paired by task position with seeds kept together
(95% bootstrap intervals, `demos_common.stats.bootstrap_diff`): accuracy in
points, then cost in $ per 1,000 tasks. **Bold**: the interval excludes zero.

| condition | none | notes | retrieval | fullctx | the plan's criterion |
|---|---|---|---|---|---|
| cheap: haiku-5.5, small | -1.4 [-4.2, +0.0]; +0.09 [-0.05, +0.25] | -1.4 [-4.2, +0.0]; **-3.79 [-4.55, -3.04]** | -1.4 [-4.2, +0.0]; +0.10 [-0.04, +0.26] | -1.4 [-4.2, +0.0]; +0.05 [-0.09, +0.21] | wins against notes; does not hold |
| cheap: haiku-5.5, medium | +0.7 [+0.0, +2.1]; +0.04 [-0.13, +0.19] | +0.7 [+0.0, +2.1]; **-5.63 [-6.28, -4.94]** | +2.1 [+0.0, +4.9]; +0.04 [-0.08, +0.17] | +2.1 [+0.0, +6.2]; -0.14 [-0.31, +0.03] | wins against notes; does not hold |
| cheap: haiku-5.5, realistic | +0.0 [-2.4, +2.8]; +0.07 [-0.03, +0.18] | -0.3 [-2.8, +1.7]; **-7.91 [-8.47, -7.32]** | -1.4 [-3.5, +0.3]; +0.05 [-0.04, +0.14] | **-2.1 [-4.2, -0.3]**; **-0.55 [-0.71, -0.42]** | wins against notes, fullctx; does not hold (less accurate than fullctx) |
| cheap: gpt-6-luna, small | +0.0 [+0.0, +0.0]; **-0.14 [-0.25, -0.01]** | +0.0 [+0.0, +0.0]; **-1.04 [-1.30, -0.77]** | +0.0 [+0.0, +0.0]; -0.01 [-0.12, +0.11] | +0.0 [+0.0, +0.0]; -0.05 [-0.16, +0.07] | wins against none, notes; does not hold |
| cheap: gpt-6-luna, medium | +0.0 [-2.1, +2.1]; **-0.17 [-0.26, -0.08]** | -0.7 [-2.1, +0.0]; **-1.72 [-1.91, -1.51]** | +0.7 [-1.4, +3.5]; -0.03 [-0.11, +0.05] | -0.7 [-2.1, +0.0]; **-0.39 [-0.53, -0.25]** | wins against none, notes, fullctx; does not hold |
| cheap: gpt-6-luna, realistic | -1.0 [-2.4, +0.0]; -0.01 [-0.07, +0.05] | +0.3 [-1.4, +2.1]; **-2.26 [-2.43, -2.07]** | -0.3 [-1.4, +0.7]; +0.05 [-0.01, +0.10] | -1.0 [-2.4, +0.0]; **-0.66 [-0.80, -0.53]** | wins against notes, fullctx; does not hold |
| cheap, auto: haiku-5.5, medium | +0.7 [+0.0, +2.1]; **+0.19 [+0.05, +0.32]** | **+5.6 [+2.1, +9.0]**; **-5.30 [-6.01, -4.58]** | +0.0 [+0.0, +0.0]; +0.10 [-0.01, +0.23] | +3.5 [+0.0, +8.3]; **-0.20 [-0.39, -0.02]** | wins against notes, fullctx; does not hold |
| mid, auto: sonnet-5.5, medium | +0.0 [+0.0, +0.0]; **+3.63 [+2.15, +5.21]** | +0.0 [+0.0, +0.0]; **-78.82 [-88.36, -68.51]** | +0.0 [+0.0, +0.0]; **+2.13 [+0.86, +3.52]** | +0.0 [+0.0, +0.0]; -0.79 [-3.35, +1.22] | wins against notes; does not hold |
| top, auto: opus-5.5 high, medium | +0.0 [+0.0, +0.0]; **+8.65 [+5.37, +12.38]** | +0.0 [+0.0, +0.0]; **-159.22 [-174.80, -142.46]** | +0.0 [+0.0, +0.0]; **+3.98 [+0.61, +7.60]** | +0.0 [+0.0, +0.0]; -2.77 [-9.27, +2.53] | wins against notes; does not hold |

"auto" marks the harness in which the model may answer in text (needed by
sonnet-5.5 and opus-5.5, which reject a forced tool call); the haiku-5.5 row
under it is the control for that change. Every table behind these (by subset,
phase, block of 8 tasks, after each change, and the library's own measures)
is printed by `analyze.py` and every number is in `results.json`, which also
lists each failed task.

### What the runs show

- **Accuracy is at the ceiling for every arm.** Every arm got 94 to 100% of
  tasks right in every condition. The library's difference from each arm is
  mostly within 2 points, and its interval includes zero except against the
  long context at the realistic size on haiku-5.5 (-2.1 [-4.2, -0.3], against
  the library) and against notes in the haiku-5.5 "auto" control (+5.6 [+2.1,
  +9.0], for it, from a failure of the notes arm described below). The
  hardest kind for every arm was the escalation (reply, then reassign, where a
  reply bumps the ticket's version and an update with the old one gets a
  409): 30 of the 60 failures across all conditions.
- **Cost: the notes arm is the expensive one; against the cheap simple arms
  the library breaks even at best.** Notes rewrite up to 2,000 words after
  every task, at 4 to 15 times the library's cost per task. The long context
  grows with the stream and costs 2 to 3 times no memory at the realistic size
  even with caching, so the library is cheaper than it there. Against no
  memory and retrieval the library's saving on recurring tasks (-$0.06 [-0.16,
  +0.05] per 1,000 tasks against no memory at the realistic size on haiku-5.5,
  -$0.09 [-0.15, -0.04] on gpt-6-luna) is cancelled by its save step after each
  new kind (+$0.62 [+0.40, +0.94] and +$0.38 [+0.27, +0.50] on first-of-kind
  tasks) and by the library listing in every prompt. On the mid and top tiers
  the save step and the listing are paid at the stronger model's price and the
  library costs more than no memory (+$3.63 and +$8.65 per 1,000 tasks) and
  than retrieval (+$2.13 and +$3.98).
- **Calls and latency.** The library cuts agent calls and the wait for an
  answer against no memory on the cheap tier (-0.58 calls and -1.1 s a task on
  haiku-5.5, -1.48 calls and -3.1 s on gpt-6-luna, realistic size), but
  retrieval does as well (gpt-6-luna with retrieval wrote one cell that did
  the whole task and submitted it in 65% of tasks), and on haiku-5.5
  the long context needs fewer calls still. On sonnet-5.5 and opus-5.5 the
  library makes more agent calls than every arm and answers 0.4 to 1.8 s
  later: it often reads a stored function's source before calling it.
- **The interface changes cost no arm anything.** In the 8 tasks after each
  change every arm was 92 to 100% correct. The library's difference from
  notes, retrieval and the long context there runs from -4.2 to +2.1 points on
  the cheap tier (+8.3 against notes in the haiku-5.5 control, from the notes
  failure below), with no interval above zero, and it is +0.0 on the mid and
  top tiers: the change claim fails in all nine conditions. Every arm
  is shown the current API reference, and a retired operation fails at once
  with "410 Gone: send_email() was removed in API v3. See the API reference.",
  so code written fresh adapts at once: over the realistic streams the
  no-memory arm never called a retired operation, while the arms that reuse
  old code did (notes 8 times, retrieval 14, long context 15, library 61),
  each then fixing it. The library did what the idea says, repairing the
  shared step in place in most cases (in 24 of the 27 tasks in which a stored
  function failed on haiku-5.5 at the realistic size, 25 of 30 on
  gpt-6-luna). But its stored composites brought a failure
  of their own on the first tasks to use a broken step: 16 of 18 correct
  against 18 of 18 for every other arm on haiku-5.5, 17 of 18 on gpt-6-luna.
- **No growth with size or stream position.** In the second half of the
  realistic stream the library against no memory is +0.0 points [-4.9, +4.9]
  and -$0.09 [-0.23, +0.06] per 1,000 tasks on haiku-5.5, and -2.1 [-4.9, +0.0]
  and -$0.12 [-0.18, -0.04] on gpt-6-luna; against retrieval -2.8 and -1.4
  points with costs level.
- **Model strength.** The mid and top tiers got all 144 tasks right in every
  arm, so the library's machinery was pure cost there. The haiku-5.5 control
  under the same "auto" harness kept the cheap tier's picture (library 100%,
  retrieval 100%, no memory 99.3%), so the change of harness is not what
  levelled the stronger models.

### The library at work, and its own failures

The mechanism ran as designed in every condition: at the realistic size 84 to
92 of 96 tasks called stored functions; libraries ended with 24 to 36
functions, about half called by other stored functions, in chains up to 5
deep; nearly all were reused in two or more tasks. Its failures, read in the
transcripts, were of two kinds that only stored code has:

1. **A latent bug that passed the one-case verification** (haiku-5.5,
   realistic, seed 2). A stored pagination helper dropped the page argument
   when given extra arguments, so it re-read page 1; it passed its save-time
   re-run because that contact had one page of orders. At task 43 (top
   spender) it double-counted one contact's orders and the answer was wrong;
   the model noticed it in a comment ("check fetch_all_pages bug ... drops
   page") and still submitted. At task 92, a word-for-word repeat, one call to
   the stored function gave the same wrong answer.
2. **A side effect before a broken step** (task 75, refund then apologise, on
   haiku-5.5 seeds 2 and 3 and gpt-6-luna seed 3; task 93 on haiku-5.5 seed 2).
   The stored composite refunded the order, then raised 410 at the retired
   `send_email`. The model, not seeing that the refund had gone through,
   either ran the refund again and refunded a second order, or found no
   delivered order left, lost track and never sent the email. A task that
   fails saves nothing, so in haiku-5.5 seed 2 the broken composite stayed in
   the library and failed the same way at task 93.

### The other arms' failures, from the transcripts

- **Notes: an invented rule.** On haiku-5.5 under the "auto" harness the notes
  writer recorded "Action tasks submit nothing. In the Oakmoor Schools tagging
  task, one run submitted `None` after a correct action. That was wrong.",
  though every task it cited had been reported correct, and the system prompt
  says to submit None. With the rule in its notes the model finished six
  action tasks in seed 3 correctly but declined to submit them, through seven
  to nine reminders each, so they count as not done. Under the forced tool
  call of the main runs the same model could not end a task without a tool
  call. Notes going wrong
  silently is what the idea predicts of notes, but here it was a made-up rule,
  not a stale one, and it cost accuracy only in the condition where the model
  could act on it.
- **Long context and notes: declining a duplicate.** Task 32 at the medium
  size repeats an email task word for word. Haiku-5.5 with the whole
  conversation (all three seeds, in both harnesses) or with notes that
  recorded the first send declined to send it again "to avoid duplicate
  emails". The task asks for the send, so these count as failures; a reader may
  judge the choice sensible.

### Caveats and threats to validity

1. **Ceiling.** With the current API reference in view, today's models solve
   this environment (21 operations, rules taught only by errors, two breaking
   changes) almost every time, from scratch. That is the result, but it also
   means the environment cannot show what memory adds when fresh derivation
   fails often. A harder setting (no reference after a change, silent changes
   in behaviour, much longer procedures) was not in the plan and was not added
   after the fact.
2. **Changes announce themselves.** A retired operation fails at once with a
   410 that names the change. Changes that silently alter results would favour
   whichever memory notices them; none was tested.
3. **The mid and top tiers are one family, one size, three seeds,** under the
   "auto" harness, while the cheap tier's main runs force a tool call. The
   haiku-5.5 control shows the harness matters for the notes arm only.
4. **Fault 9's re-scoring** (13 sonnet-5.5 tasks) corrects their verdicts but
   not what the wrong "not correct" did later in their streams, which could
   only have hurt the library, notes and retrieval arms.
5. **haiku-5.5's main runs predate fault 7** (bare expressions shown); it used
   them in 2 of 5,092 cells.
6. **Small differences.** Three seeds and 24 to 96 tasks give intervals of
   several points on accuracy; differences of 1 to 2 points are noise here.
   Temperature 0 is not deterministic, and opus-5.5 with reasoning takes no
   temperature.
7. **The library's verification re-runs one case,** the task's own. The
   latent-bug failure shows how weak a test that is; a library with stronger
   tests (several cases, property checks) was not the design tested.

### Cost (from the ledgers)

Today's models: **$73.77** (laptop $2.59, cloud machine $71.18), under the $80
stop; older models briefly run under amendment 1: $2.57, outside the analysis.
`uv run python -m demos_common.ledger runs/ledger.jsonl runs/ledger-vm.jsonl`
reports $76.33 for both together.

| condition (kept streams only) | cost |
|---|---|
| cheap, haiku-5.5: small, medium, realistic (3 seeds each) | $0.45, $1.28, $3.24 |
| cheap, gpt-6-luna: small, medium, realistic | $0.21, $0.59, $1.29 |
| cheap, haiku-5.5 "auto" control, medium | $1.25 |
| mid, sonnet-5.5, medium | $17.76 |
| top, opus-5.5 high, medium | $38.19 |
| sanity, smoke, probes and discarded runs | $9.51 |

The ledgers by run label, size and model are in `results.json` (`spend`).
**Not run, for budget:** the realistic size on the mid tier (about $45 for
three seeds) and the top tier (about $97), scaled from their medium-size cost
by the cheap tier's ratio of realistic to medium cost (2.5).

## Build

- `src/skills/world.py`: the workspace's data from a seed (users, companies,
  contacts, tickets, orders), and the observable state every check compares.
- `src/skills/api.py`: the API in three versions (v2 and v3 retire operations
  with 410 Gone), with the rules that only errors teach (refund approvals over
  10,000 cents, ticket versions that a reply bumps).
- `src/skills/tasks.py`: the 16 kinds, each with a sampler, a task text and a
  reference solution on the data; the fixed schedule per size; the stream
  generator; and the exact check (answer, and every change to the workspace).
- `src/skills/sandbox.py`: one namespace per task, with limits (3 million
  traced lines per cell, 600 API calls per task), restricted imports, captured
  output, tracebacks that show only the model's code and the library, and the
  bookkeeping the library needs (which functions ran, where API calls came
  from, what library functions returned).
- `src/skills/agent.py`: the loop every arm shares (a `run_python` tool and a
  `submit` tool, `submit(answer)` also callable in code, 10 tool calls per
  task, the calls left shown after each one),
  the shared system prompt and prompt caching (the shared prompt, the memory
  section and the task message are cache breakpoints, and in the long-context
  arm the start of the previous task too).
- `src/skills/arms.py`: the five arms (see the plan), including the library's
  save step and its verification.
- `run.py` runs streams; `run_all.sh` holds every batch behind the results;
  `analyze.py` writes `results.json` and the tables.
- `tests/`: 102 tests with the model stubbed. They cover the API in every
  version, a correct program written only against the API passing every task
  of every stream (four sizes, five seeds), ignoring pagination failing some,
  the sandbox's limits, each arm's prompt and memory update, the library's
  storage rules, save, verification, repair and skip rule, retrieval finding a
  task of the same kind first in over 90% of cases, and a whole small stream
  through every arm.

## Checks before the runs

- **Sanity condition** (`runs/sanity/`): 8 one-step lookups, no change, seed
  1. Every arm got 8 of 8, before and after each fix below.
- **Smoke runs** (`runs/smoke/`): the small size (24 tasks) and the first 40
  tasks of the realistic size, seed 1, every arm, read in full. With the
  final harness: small 24/24 in every arm; realistic 40/40 for retrieval,
  fullctx and the library, 39/40 for none and notes (both on the first
  escalation, which needs a rule that only an error teaches).
- Earlier sanity and smoke runs, discarded for the faults they found, are in
  `runs/discarded/` with a README giving each reason.

## Bugs found and fixed

Faults 1 to 5 were found before any experiment run, 6, 7 and 9 while reading the runs' transcripts, and 8 in the analysis.

1. Library arm: with instructions alone, the model never defined a function,
   so the library stayed empty (16 of 16 probe tasks). Fixed by the save step
   (plan refinement 1).
2. Library arm: the rule that skips the save when the library already did the
   task missed "print a library function's result, then submit the printed
   value", which caused redundant saves and near-duplicate functions under new
   names. Fixed by also skipping when the submitted answer is a value a library
   function returned, and by telling the save step to reuse or replace, not
   add a second version.
3. Notes arm: rewrites were cut at the 4,096-token output cap (20 times in the
   smoke runs), silently losing the end of the notes. Fixed: cap 8,192, and a
   cut rewrite is asked for again, shorter.
4. All arms: the model ran out of calls without submitting. Fixed: each tool
   result shows the calls left.
5. All arms: with a tool call forced on every turn, a model that was done could
   not end its turn and looped, printing summaries or re-running its last call
   (once re-sending emails eight times). Fixed: a `submit` tool.
6. Notes arm (found in the primary runs): claude-haiku-5.5 ignored the
   1,500-word limit in about half its rewrites, and the 2,000-word safety cap
   then cut the end off its notes. Fixed: a rewrite over 2,000 words is asked
   for again, shorter. The notes arm was re-run on claude-haiku-5.5 at every
   size; gpt-6-luna's notes never reached the cap.
7. All arms (found in the era models' transcripts): the harness showed only
   printed output, but some models end a cell with a bare expression and
   expect its value, as in a notebook. gpt-6-luna did so in 6.4% of its cells
   and claude-haiku-5.5 in 2 of 5,092, wasting calls. Fixed: a final bare
   expression's value is shown. gpt-6-luna was re-run at every size;
   claude-haiku-5.5's runs were kept, since two cells in 5,092 cannot move its
   numbers.
8. Analysis: an early measure counted a library function as "left broken"
   whenever it appeared in a failing traceback, though the fault was often in
   its caller (a generic pagination helper handed a retired operation). The
   measure now counts a stored function as left broken only if it raised again
   in a later task with no replacement in between.
9. Check (found in the mid tier's failures): claude-sonnet-5.5 sometimes sent
   a list answer as text through the `submit` tool, whose answer field has no
   type (`"['t_5010', 't_5049', ...]"`), and the check counted the right list
   wrong. All 13 of its failures of this form had the right list; no other
   model did it. Fixed: the check parses a list sent as text, and `rescore.py`
   re-checks recorded runs from their transcripts, keeping the original verdict
   in `passed_original`. What those 13 tasks' wrong "not correct" outcome did
   later in their streams (notes written from it, solutions not stored, a
   conversation told it failed) could not be undone, and could only have hurt
   the library, notes and retrieval arms, where they all fell.

## Re-running

```bash
cd concepts/skills-as-functions
uv sync && uv run pytest -q                                   # 102 tests, no model calls
# the numbers from the recorded runs: restore runs/, re-check, analyse
zstd -dc runs.tar.zst | tar -xf -
uv run python rescore.py                                      # fault 9: re-check answers sent as text (idempotent)
uv run python analyze.py --labels main,tiers                  # writes results.json, prints every table
uv run python -m demos_common.ledger runs/ledger.jsonl runs/ledger-vm.jsonl

# the runs themselves, from scratch (about $75 at 9 October 2026 prices)
K=../../bin/with-openrouter-key
$K uv run python run.py --label sanity --size sanity --seeds 1
$K uv run python run.py --label sanity --size sanity --seeds 1 --model anthropic/claude-sonnet-5.5 --tool-choice auto
$K uv run python run.py --label sanity --size sanity --seeds 1 --model anthropic/claude-opus-5.5 --tool-choice auto --reasoning high --extra-max-tokens 12000
$K uv run python run.py --label smoke --size small --seeds 1
$K uv run python run.py --label smoke --size realistic --seeds 1 --limit 40
# on the cloud machine (see the repo README), one batch at a time; concurrency within each is 12 or less
./run_all.sh haiku; ./run_all.sh luna; ./run_all.sh tier-mid; ./run_all.sh tier-top; ./run_all.sh tier-cheap-auto
# (./run_all.sh haiku-notes re-ran the notes arm after fault 6; on a fresh run it skips the complete streams)
uv run python rescore.py && uv run python analyze.py --labels main,tiers
```

A stream is resumable only from its start: `run.py` skips complete streams
and moves an incomplete one to `runs/discarded/` before running it again.
Model replies at temperature 0 still vary a little between runs, so a re-run
reproduces the numbers within their intervals, not exactly.

## Handover (2026-10-09 18:45-19:30 PDT)

The agent that built this concept was stopped by a usage limit at 18:45 PDT
while the last two batches ran on the cloud machine (the top tier, restarted
at 18:33, and the haiku-5.5 "auto" control, started at 18:34). A second agent
took over: it found both running and healthy, left them to finish (19:14 and
19:18 PDT) behind a spend guard, restarted nothing, copied `runs/` back,
re-applied `rescore.py`, read transcripts from every arm of the two new
conditions (and the library's failures in the cheap tier), added the spend
summary and failure lists to `analyze.py` and `results.json`, and wrote the
results above. It also recorded two discarded runs that `runs/discarded/README.md`
had missed, and made `run_all.sh tier-top` match the top tier as run.
