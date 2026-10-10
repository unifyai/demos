# separate-loops

## Stopped on 9 October 2026 at Dan's request

Dan stopped the experiments at 20:57 PDT on 9 October (03:57 UTC on 10
October). Every process of this concept on the cloud machine was stopped by
PID, and everything it held is in `runs/` here. The results were not
analysed: there is no Results section and no `results.json` (`analyze.py`
can produce it from `runs/`; it also scores the Plan's three criteria and
sums both ledgers).

- **Ran in full:** sanity on all four models (12 runs each: `sanity-haiku`,
  `sanity-luna`, `sanity-sonnet`, `sanity-opus`); `small-haiku` 36,
  `medium-haiku` 36, `realistic-haiku` 48, `dual-haiku` 36,
  `realistic-sonnet` 48, `realistic-opus` 48.
- **Stopped partway:** `realistic-luna` 40 of 48 (missing C-s2 background;
  B-s3 blocking, background and handle; all four arms of C-s3) and
  `medium-luna` 34 of 36 (missing C-s2 blocking and background). Runs in
  flight when stopped left no record.
- **Never ran:** `dual-luna`, `small-luna` and `dual-sonnet` (36 runs each).
- **Judge** (`google/gemini-3.8-flash`): read the six full counted batches
  above (`judge.jsonl` in each), not the GPT-6 luna batches or sanity.
- **Spend, from both ledgers (neither is a copy of the other):** today's
  models $43.24 (`runs/ledger-vm.jsonl`, the cloud machine, $35.26;
  `runs/ledger.jsonl`, the laptop, $7.98, of which the judge $4.14); the older
  models of Amendment 1 $3.38 (laptop ledger), kept apart and not counted
  against the cap; $46.63 in all.

Talking and acting in separate loops, joined by a steerable handle, keeps a
conversation responsive and lets corrections reach work in flight.

From the post "Talking and acting in separate loops"
(`landing-page/content/blog/talking-and-acting-in-separate-loops.mdx`), read for
the idea only. The earlier experiment on the original code
(`landing-page/experiments/talking-and-acting-in-separate-loops/README.md`)
found that two loops answered "how's it going?" in about 7 s against about 48 s
for one loop, and that corrections reached running work within seconds, but
that code lost the finished work's result, crashed when it resumed work that
was running, and lost questions from the work. This folder rebuilds the idea
from scratch, without those faults, and tests it against what a competent
engineer would build instead.

## Plan (9 October 2026, 22:20 UTC, before any experiment run)

### The idea

An agent that works for a person for minutes at a time should run the
conversation and the work as two loops. The conversation loop answers the
person and never waits on the work. Starting work returns a handle the
conversation can steer while the work runs: pass it a correction or an
addition, pause it, resume it, stop it, ask how it is going, and answer a
question the work asks back. So the person is answered at once while the work
runs, a correction reaches work in flight before more is done with the old
value, and the work can ask for a fact instead of guessing.

### Conditions it is built for (from its own reasoning)

- **Long work that is itself an agent:** many tool steps, each a page load or
  a form submission of seconds to tens of seconds, so a task runs for minutes.
- **A person who keeps talking:** six to ten messages while the work runs:
  questions about progress, a detail they forgot, a correction of a detail
  already used, "hold on" and "carry on", a change of mind about pending
  items, an unrelated question, and the answer to a question from the work.
- **A fact only the person has,** which shows only once the work is under way.
- **Several pieces of work at once** (a second request while the first runs,
  then cancelling one of them), which the post names as untested.

Where it should not matter: short steps, few messages. The sweep of sizes is
there to show where the idea pulls ahead, or that it does not.

### Arms

The same model, the same tools, the same data and the same shared
instructions in every arm. They differ only in how the person reaches the
work. Two factors: is the conversation separate from the work, and can the
running work be steered?

| arm | separate? | steerable? | what it is |
|---|---|---|---|
| `flat` | no | yes (it is the work) | One loop and one context, the strongest single loop: it does the work with the site tools, the person's messages are added between its steps, a message that arrives while the model is deciding its next step discards that decision so the model decides again with the message in view, it replies in text in the same response as its next tool calls (since fault 7: with a `reply_to_person` tool), and its turn ends when it calls no tools. This is also "everything in one long context". |
| `blocking` | no | no | A conversation loop whose one tool, `run_task`, runs the work agent to completion and returns its report. The person's messages wait until it returns. The common "sub-agent as a tool". |
| `background` | yes | no | The conversation starts work in the background (`start_task` returns an id at once), reads its progress (`task_status`), stops it (`stop_task`), and is told when it finishes. A running task cannot be given new instructions and cannot ask anything: to change it, stop it and start a new one, or follow up afterwards; a task that needs a fact finishes with a question. Fire and forget, with polling and cancel. |
| `handle` | yes | yes | The idea. As `background`, plus `send_to_task` (a correction, an addition, or the person's answer, delivered to the work before its next step, discarding a decision in flight as `flat` does), `pause_task`, `resume_task`, and `ask_person` in the work, whose question is put to the conversation as a new notice and whose answer comes back as a message. |

The work agent in `blocking`, `background` and `handle` is one loop (the same
code); `handle` adds only the inbox, the pause and stop gates and
`ask_person`. Faults of the old code designed out, each with a test: a
finished task's report is a new notice at the end of the conversation (not
written into an earlier message); pause, resume and stop check the task's
state and answer with a message, never an exception; every question from the
work is its own notice, and any message sent to a task while it has questions
pending answers all of them.

Instructions: a line giving today's date (added for fault 5, below), one block
on how to talk to the person and one on how to do the
work (apply corrections at once and fix what they affect, ask rather than
guess, pause and stop when asked, report the outcome) are shared verbatim.
Each arm adds only a paragraph on its own mechanism.

### Environment and data

Three simulated sites, each a set of tools with state and a log of every
action (when it was queued, started and finished, by which worker, with which
arguments): a messaging site (groups, profiles, contacts, messages), a booking
site (search, availability, menus, holds, sharing) and a shop (search, items,
cart, checkout). Every action takes a time drawn deterministically from the
instance seed, the tool and its arguments, so the same action takes the same
time in every arm; each worker has one session per site that does one action
at a time, as a browser would. Every arm is offered every site's tools.
Outcomes are checked from the sites' state and logs, never from the agent's
prose.

Three task families, generated by `run.py gen` from seeds:

- **A, invitations:** invite a group and "my neighbour Sam" to an event,
  checking each profile and skipping members who are away. The address book
  has two Sams (the fact only the person has).
- **B, booking:** check every restaurant in a search for a table, hold the
  cheapest with room, share it with a friend. Holding needs a phone number
  the request leaves out.
- **C, ordering:** order a list of office supplies, the cheapest in-stock
  option of each, and check out. Checkout needs a purchase-order number the
  request leaves out.

Sizes (the sweep):

| size | items per task | action times | person's messages while the work runs |
|---|---|---|---|
| `sanity` | 4, nothing missing from the request | 0.2-0.8 s | 1: a status question |
| `small` | 5 | 0.5-3 s | 3: status, correction, unrelated question; plus the answer to the work's question |
| `medium` | 10 (7 for C) | 2-10 s | 6: status, correction, addition, pause, resume, unrelated; plus the answer |
| `realistic` | 16 (10 for C) | 3-30 s | 8: status, correction, addition, pause, resume, unrelated, status, change of mind; plus the answer |
| `dual` | realistic main task, plus a second task (6 items, another family) requested partway | as realistic | 7: status, correction, the second request, status on both, unrelated, cancel the second, addition; plus the answer |

The person is scripted. Each message fires when the work reaches a point
(for example when the fourth distinct invitee's message starts), at least 6
to 8 s after the previous one, and at most 45 to 150 s after it if the work
has stalled; "carry on" comes 20 s after "hold on". Progress triggers make the
correction arrive after the same amount of work in every arm. The person
answers the agent's questions through a responder model given a fixed sheet
of facts (the right Sam, the phone number, the PO number, "yes" to
confirmations, "use your judgement" otherwise, "not yet" while paused), 2 s
after the question. The run ends when the script is done, the person has
nothing to answer and nothing has happened for 10 s, or at a cap (6, 15, 30
and 35 minutes).

### Measures

Per message from the person, while the work runs:
- `reply_s`: seconds to the first message to the person from a model call
  that had the message in view.
- `answer_s` (questions about progress and unrelated questions): seconds to
  the first message that answers it accurately, as judged with the sites' log
  at that time; unanswered questions reported separately.
- For corrections, additions and changes of mind: `reach_s`, seconds until
  the loop doing the work first decides a step with it in view; and
  `stale_steps`, actions started after the person's message that use the old
  value, leave out the addition, or act on the dropped item.
- Pause: actions started between "hold on" and "carry on". Cancel: actions
  on the cancelled task started after the request.
- The work's question: whether the person was asked and gave the fact, and
  whether a guess reached a site.

Per run: `correct` (every final-state check passes: the right people with the
right time and the dish, nobody away, the right Sam; the right restaurant,
party size, phone and share; the right items and quantities, address and PO,
and for `dual` nothing done on the cancelled task after it was cancelled),
`checks` (share passed), `reported` (the agent's last messages state the
outcome accurately, judged), wrong and duplicate side effects, stalls (the
work stopped unfinished with nothing pending), crashes, caps, total time to
the outcome message, model calls, tokens and cost per run from the ledger,
and per call the time spent queued in the client beyond the model's own
latency (reported, since runs share a client).

Statistics: proportions with Wilson intervals; differences between arms by
bootstrap over items, paired (`demos_common.stats`); medians and 90th
percentiles of times.

### Models

- Primary, every arm and loop: `anthropic/claude-haiku-5.5` (a fast model, as
  an agent that talks would use; no reasoning by default), provider defaults.
- Second family on the headline sizes (`realistic`, `dual`):
  `openai/gpt-6-luna`.
- The person's responder and the judge: `google/gemini-3.8-flash`
  (reasoning effort low for the responder, default for the judge), a third
  family for both agent models. The judge is checked by hand on a sample.

### Items, seeds and size of the run

One run per item, arm and model; items are generated instances (seeds).
`sanity`: 3 families x 1 seed; `small`, `medium`, `dual`: 3 x 3 seeds;
`realistic`: 3 x 4 seeds. Primary model: 12 + 36 + 36 + 48 + 36 = 168 runs
(4 arms each). Second family: sanity 12, realistic 48, dual 36. About 3 to 4
hours on the cloud machine at six runs at a time per model, with the client's
`max_concurrency` at 6 per process (12 in all). Estimated spend $25 to $35.

Procedure: unit tests with the model stubbed; the sanity condition (every
arm should get every task right); a smoke run at `medium`, one item per
family, every arm, with every transcript read; then the batches, copying
`runs/` back after each.

### What counts as the idea winning (fixed now)

At the `realistic` size and in `dual`, on the primary model, against the
strongest baseline on each measure:

1. **Responsive:** the median `answer_s` for questions asked while work runs
   is lower for `handle` than for `flat`, with the 95% interval of the
   difference excluding zero.
2. **Steers work in flight:** `handle`'s stale steps per run are no more than
   `flat`'s (interval includes zero or favours `handle`) and fewer than
   `background`'s and `blocking`'s.
3. **No worse at the work:** `handle`'s share of runs `correct` is within 10
   points of the best baseline's (point estimate, intervals shown), and the
   outcome is reported in at least 90% of its runs.

All three: the idea holds. 1 without 2 or 3: a trade-off, reported as such.
Not 1: it does not pull ahead. The same direction on the second family is
needed to call it more than one model's behaviour. Expected, from the
mechanism: no responsiveness advantage at `small` (steps are short), growing
with step length and the number of messages; cost higher for `handle` than
`flat` (the conversation's calls), lower context per call.

## Amendment 1 (10 October 2026, 00:25 UTC): model strength as an axis (replaced by Amendment 2)

From Dan, through the lead. Reason: smarter models need thinner harnesses, so
the fair test of a design is on the models it was built for.

- **Headline: the models of the design's own time, June 2025.**
  `openai/o4-mini` is the primary model (the original code's default for
  both loops), and `anthropic/claude-sonnet-4` the second family. Every loop
  of every arm in a condition uses the same model, at the provider's default
  settings (o4-mini reasons at its default effort).
- **Today's cheap tier.** The runs on `anthropic/claude-haiku-5.5` and
  `openai/gpt-6-luna` stay as a labelled condition, not the headline.
- **Today's frontier, if the budget allows after the headline:**
  `anthropic/claude-opus-5.5` with reasoning effort high, or
  `openai/gpt-6.1-sol`, on the headline comparison only, with fewer seeds if
  need be, to show whether the advantage shrinks with stronger models.
- The person's responder and the judge stay `google/gemini-3.8-flash`, a
  different family from every model under test.
- **Budget:** a hard stop at $80 for the whole concept, including the $3.31
  spent before this amendment. If the realistic size does not fit on the era
  models, the largest size that does is run, and the cost of the full size
  is reported.
- The criteria for "the idea wins" are unchanged; they now apply to o4-mini at
  `realistic` and `dual`, with Sonnet 4 for the second family's direction.

Order of work and estimates. Measured per `realistic` run on the cheap tier:
prompt tokens 79k (blocking) to 261k (background) with Anthropic's
tokenizer, 34k to 152k with OpenAI's; 13 to 42 model calls. At list prices
(o4-mini $1.10 and $4.40 per million tokens in and out, cached input $0.275;
Sonnet 4 $3 and $15) and allowing for o4-mini's hidden reasoning, a
`realistic` run costs about $0.15 to $0.30 on o4-mini and $0.85 on Sonnet 4,
plus about $0.03 for the person and the judge.

1. Sanity on o4-mini and Sonnet 4 (every arm should get every task right).
2. o4-mini: `realistic` (48 runs) and `dual` (36): about $20 to $30.
3. Sonnet 4: `realistic`, seeds 0 and 1 (24 runs, about $21); all 48 runs
   would cost about $41, and `dual` about $35 more, which do not fit.
4. Today's cheap tier as planned: `realistic` and `dual` on both models, and
   the size sweep (`small`, `medium`) on Haiku: about $12 with the judge.
5. o4-mini `small` (the size sweep on the headline model): about $5.
6. The frontier, with what remains: GPT-6.1 sol `realistic`, seed 0 (12
   runs, about $4), or Opus 5.5 (about $19 for the same 12).

The spend is checked against the ledgers before each step, and nothing
starts that could take the total past $80.

Prompt caching (10 October, 00:45 UTC). OpenAI's models cache a repeated
prompt prefix on their own (82% of GPT-6 luna's prompt tokens and 61% of
o4-mini's were read from the cache); Anthropic's cache only where a request
marks it, and none had been. Every Anthropic call now marks three cache
points, as a competent engineer would: the system prompt, the end of the
previous call's prompt and the end of this one
(`core.anthropic_cache`; a probe showed the second call reading 3,391 of
3,396 tokens at a tenth of the price). It changes what a run costs, not what
the model writes. Runs on Anthropic models before it (Haiku's and Sonnet 4's
sanity runs) were uncached.

## Amendment 2 (10 October 2026, 00:50 UTC): today's models in three tiers

From Dan, through the lead; it replaces Amendment 1. Use only today's models
that someone would reasonably choose today, on the cost and quality frontier,
and no models from the design's own time. Model strength stays an axis, so
the results show how the idea's advantage changes as models get stronger:

| tier | models | what runs |
|---|---|---|
| cheap | `anthropic/claude-haiku-5.5`, `openai/gpt-6-luna` | the full design on both families: sanity, `small`, `medium`, `realistic`, `dual`, every arm |
| mid | `anthropic/claude-sonnet-5.5` | the headline comparison: `realistic`, every arm; `dual` if the budget allows |
| top | `anthropic/claude-opus-5.5`, reasoning effort high | the headline comparison: `realistic`, every arm, as many seeds as fit |

- Every loop of every arm in a condition uses that condition's model: in
  `blocking`, `background` and `handle` the conversation loop and the work
  loops run the same model, as `flat`'s one loop does.
- One family is enough above the cheap tier. Sonnet 5.5 is the mid tier so
  the three tiers are one family's ladder (Haiku, Sonnet and Opus 5.5) and a
  change with strength is not a change of family; GPT-6 luna gives the cheap
  tier's second family.
- Everything run on the June 2025 models under Amendment 1 (sanity on
  o4-mini and Sonnet 4, a medium smoke on Sonnet 4, a realistic smoke on
  o4-mini stopped partway; $3.38) is kept in `runs/era-models/` with the
  reason, and left out of every analysis.
- The person's responder and the judge stay `google/gemini-3.8-flash`, a
  different family from every model under test.
- Budget: a hard stop at $80 for the whole concept, including the $7.14 spent
  when this amendment was recorded. Order: sanity on all four models; the
  cheap tier, both families, every size (about $18 with the judge); Sonnet
  5.5 `realistic` (about $15 to $20); Opus 5.5 `realistic`, seeds as fit
  (about $1.50 a run); then Sonnet 5.5 `dual` if what remains allows. What
  does not fit is reported with its estimated cost.
- The criteria for "the idea wins" are unchanged. They are judged on each
  tier where the comparison runs: on the cheap tier at `realistic` and
  `dual`, with GPT-6 luna for the second family's direction, and on the mid
  and top tiers at `realistic`.

## Faults found before the counted runs, and changes to the plan

**Fault 1 (9 October, 22:45 UTC, found in the smoke run).** The work model
(Haiku 5.5) puts 8 to 10 tool calls in one response, and a worker's session
does one action at a time, so one response's actions queue for 40 to 80 s at
`medium`. Two consequences, both wrong: in `handle` and `background`, actions
still queued when the conversation paused or stopped the task went on to
start (`pause_task` at 88.7 s, the next queued message sent at 90.2 s; a
stopped task kept adding to the cart for 18 s), so "no new step starts" was
false; and in `flat`, a message could not be read until the whole queue had
run, which no competent single loop would accept, so the baseline was
handicapped by an artefact of the simulation. The fix, the same in every
arm: an action whose turn comes after a new message from the person (in
`flat`), or a message passed on, a pause or a stop (in a work loop), is not
started, and its result says why, so the model decides again with the
message in view. An action already under way still finishes. The status view
now tells queued actions from those under way. Tests:
`test_pause_mid_batch_holds_the_actions_not_yet_started`,
`test_message_mid_batch_skips_the_rest_in_the_work_and_in_flat`,
`test_background_stop_mid_batch_starts_nothing_more`. The plan's rule "a
message that arrives while the model is deciding its next step discards that
decision" now reads "while the model is deciding its next step, or before an
action it decided has started". The result of an action not started first
read "... Decide again with that in view."; in the third smoke run `flat`
answered an unrelated question that had interrupted six queued messages and
ended its turn without sending them, so the text now reads "... Call it again
if it is still needed." (the same text in every arm; the runs before are in
`runs/discarded/`). The sanity and smoke runs made before the fix
are in `runs/discarded/` with the reason.

**Fault 2 (9 October, 23:05 UTC, found in the second smoke run).** The
generator gave every restaurant a street in Shoreditch whatever area the
person searched, and names such as "Trattoria Verde" whatever the cuisine. In
one `background` run the work noticed that the Soho results had Shoreditch
addresses and asked; the person said to skip them, and nothing was booked.
Restaurants now get streets in the searched area and names that fit any
cuisine. The sanity and smoke runs made before the fix are in
`runs/discarded/`.

**Fault 3 (9 October, 23:45 UTC, found in the realistic smoke run).** The
scripted person's responder could send a reply it had thought out before a
scripted message went out: in one `background` run the person said "carry
on" and, half a second later, the responder's "Not yet, I will tell you when"
(written while the person was still paused), and the person then held to it,
so the work never resumed. Now a reply is dropped, and the responder thinks
again, if the person said anything else while it was thinking or waiting.
Test: `test_responder_drops_a_reply_thought_out_before_the_person_said_something_else`.
The realistic smoke run is in `runs/discarded/`.

**Fault 4 (9 October, 23:50 UTC, found in the sanity run on the second
family).** The friend a booking is shared with was not in the address book,
and a filler contact could share their first name. GPT-6 luna's `background`
work looked the friend up before sharing, found only "Mei Murphy" for "Mei
Okafor", and rightly did not share. The friend is now in the address book,
and nobody else there has their first name. Both sanity runs and the medium
smoke run were re-run; the earlier ones are in `runs/discarded/`.

**Fault 5 (10 October, 00:00 UTC, found in the first counted batches, which
were stopped after 15 runs).** In `handle`, a resumed task was never told it
had been resumed: its last results said "not started: the work was paused",
and GPT-6 luna's work replied "Work is paused, so I haven't sent anything
further" and ended the task. The work now gets a note when it carries on
after a pause ("the work was paused and has now been resumed"). In the same
runs a work loop asked which year "Sunday 25 October" was, since no prompt
gave today's date; every system prompt, in every arm, now starts "Today is
Friday 9 October 2026." (the scenarios' dates run from 12 October to 27
November 2026). The 15 runs are in `runs/discarded/`; both sanity runs were
made again, and every counted batch was started afresh.

**Fault 6 (10 October, 00:20 UTC, found when checking settings for the era
models).** The cap on each agent reply, 2000 tokens with reasoning included,
truncated 78 of 3,175 agent calls, all from Haiku 5.5 and nearly all at the
`realistic` size, in every arm: a reply with 13 to 20 tool calls ran past the
cap and its last call was cut off ("one call was malformed", as the model put
it). The cap is now 16,000 tokens, far above any reply, for every arm and
model, and every call's finish reason is recorded. The `realistic` Haiku
batch (26 truncated calls) was stopped and set aside. GPT-6 luna's replies
never reached the cap, so its `realistic` batch stands (checked again on the
finished batch).

**Fault 7 (10 October, 00:30 UTC, found in the sanity run on o4-mini).**
`flat` could only talk while working by writing text in the same response as
its next tool calls. o4-mini never does that: asked how it was going, it
answered in text alone, which ends a single loop's turn, and the work stopped
there (2 of 3 sanity tasks left unfinished, where every arm should succeed).
A competent single loop on such a model gives it a tool to message the
person without ending its turn, so `flat` now has `reply_to_person` (the
message reaches the person at once), and its prompt says to use it while
working. This strengthens the baseline. Every `flat` run made before it (the
sanity runs on all three models so far and GPT-6 luna's `realistic` runs) was
moved to `runs/discarded/` and run again.

**Fault 8 (10 October, 00:40 UTC, found in a Sonnet 4 smoke run, now set
aside).** A group member's absence could end before the event ("away until 5
November" for a dinner on 14 November): 14 of the 34 members marked away were
back by their event. The request says to skip anyone who is away, and the
checks counted any message to them as wrong, but a model that reads the dates
can fairly invite them (Sonnet 4 did, in three of four arms). Every absence
now runs past the event. The generator also now draws each part of an
instance (the main task, the second task, each filler site, the friend, the
script's wording) from its own seeded stream, so a fix to one part cannot
shift another. All data changed; GPT-6 luna's `realistic` batch (26 runs, also
made before fault 7's fix) was stopped and set aside, and the sanity runs
were made again on the new data.

**A measure made precise before any `dual` run.** "Nothing done on the
cancelled task after it was cancelled" counts forward side effects only
(sending a message, holding or sharing a table, adding to the cart, checking
out); undoing (removing from the cart, cancelling a hold) after a cancel is
not held against an arm. Reads after the cancel are reported as
`cancel_steps`.

## How the counted batches ran

All on the cloud machine, two batches at a time (6 runs at once each, client
`max_concurrency` 6 each, so 12 in all), each batch under a spending guard
(`--stop-at`, which starts no new run once the ledgers reach it):

- From 00:51 UTC: `realistic-haiku` and `realistic-sonnet` (48 runs each),
  both complete by 01:58.
- Lane A (`runs/lane_a2.sh`): `dual-haiku`, `medium-haiku`, `small-haiku`.
- Lane B (`runs/lane_b2.sh`): `realistic-opus` (guard $52), then GPT-6 luna.
- The first agent was cut off by a usage limit at 01:45 UTC; a second agent
  took over at 02:00 with both lanes running and nothing lost. It re-queued
  what was still waiting, so the two lanes finish together, without touching
  a running batch: `runs/lane_b3.sh` runs `realistic-luna` and `dual-luna`
  after the Opus batch, and `runs/lane_a4.sh` runs `medium-luna` and
  `small-luna` after lane A, then `dual-sonnet` (the last step of Amendment 2)
  under a lower guard ($66, counting the $3.38 on the older models) so that it
  can never take spend from the cheap tier or the judge. Lane A waits up to 30
  minutes between its parts while the judge reads the finished batches on the
  laptop, so the concept never has more than 12 calls in flight.
  (`runs/lane_a3.sh.replaced` was its first version, stopped before it started
  anything.)
- Everything stopped at 03:57 UTC at Dan's request (the note at the top),
  during `realistic-luna` and `medium-luna`.

## How it is built

| file | what it is |
|---|---|
| `src/separate_loops/sites.py` | the three simulated sites: tools, state, the log, action times, one session per worker |
| `src/separate_loops/scenarios.py` | the generator: families A, B, C, the sizes, the person's scripts and facts, what the checks expect |
| `src/separate_loops/core.py` | the run's record, the one wrapper every model call goes through (`demos_common.LLM`), mailboxes |
| `src/separate_loops/arms.py` | the shared instructions, the work loop (`Task`: inbox, pause, resume, stop, status, `ask_person`), and the four arms |
| `src/separate_loops/person.py` | the scripted person and its responder |
| `src/separate_loops/runner.py` | one run: the arm, the person, the end (idle, cap or crash), the record |
| `src/separate_loops/checks.py` | the measures from a record: final-state checks, stale steps, reply and reach times, pause and cancel steps, the missing fact |
| `src/separate_loops/judge.py` | the judge: first accurate answer to each progress question, and the outcome report |
| `run.py` | `gen`, `run`, `show` (a run's whole timeline), `judge`, `eval` |
| `analyze.py` | `results.json` and the tables below |
| `tests/` | the mechanism and every arm with the model stubbed (the old code's three faults among them), the sites, the person, the generator and the checks |

The handle in `arms.py` is about 150 lines: `Task.interject` (a message is
queued for the work and answers any question it has pending), `pause`,
`resume` and `stop` (each checks the task's state and answers with a
sentence), `status` (what the work has done, what is under way or queued, the
question it waits on, its latest note), and the loop, which reads queued
messages before each model call, discards a decision made while a message,
pause or stop arrived, holds actions not yet started, waits for an answer when
it has asked and has nothing else to do, and on finishing puts its report to
the conversation as a new notice.

## Re-running

```sh
cd concepts/separate-loops
uv sync && uv run pytest -q                       # 39 tests, the model stubbed
uv run python run.py gen                           # data/*.json from the seeds
K=../../bin/with-openrouter-key
R="uv run python run.py run --parallel 6 --max-concurrency 6 --stop-at 78"   # on the cloud machine add --ledger runs/ledger-vm.jsonl
# sanity on every model: every arm should get every task right
for m in anthropic/claude-haiku-5.5:haiku openai/gpt-6-luna:luna anthropic/claude-sonnet-5.5:sonnet; do
  $K $R --condition sanity --batch sanity-${m#*:} --model ${m%:*}
done
$K $R --condition sanity --batch sanity-opus --model anthropic/claude-opus-5.5 --reasoning high
# the cheap tier: both families, every size
for c in realistic dual medium small; do
  $K $R --condition $c --batch $c-haiku --model anthropic/claude-haiku-5.5
  $K $R --condition $c --batch $c-luna --model openai/gpt-6-luna
done
# the mid and top tiers: the headline comparison
$K $R --condition realistic --batch realistic-sonnet --model anthropic/claude-sonnet-5.5
$K $R --condition realistic --batch realistic-opus --model anthropic/claude-opus-5.5 --reasoning high
$K $R --condition dual --batch dual-sonnet --model anthropic/claude-sonnet-5.5        # as the budget allowed
# the judge (google/gemini-3.8-flash), then the analysis
for b in sanity-haiku sanity-luna sanity-sonnet sanity-opus {small,medium,realistic,dual}-{haiku,luna} realistic-sonnet realistic-opus dual-sonnet; do
  $K uv run python run.py judge --batch $b
done
uv run python analyze.py                           # results.json and the tables
uv run python run.py show --batch realistic-haiku --run 'realistic-A-s0|handle'   # one run's whole timeline
uv run python run.py judgecheck --batch realistic-haiku --n 20                     # the judge's verdicts, to check by hand
uv run python -m demos_common.ledger runs/ledger.jsonl runs/ledger-vm.jsonl        # cost
```
