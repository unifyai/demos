# fast-slow-voice

A spoken agent splits its side of the conversation between a fast model that
starts speaking within about a second and never guesses, and a slow model with
the whole context and every tool. Two designs of the split are tested against a
single fast model, a single slow model and a fast model with everything in its
context, in simulated spoken conversations with an explicit timing model.

## Plan (written 2026-10-09 22:30 UTC, before any experiment run)

### The idea

Speech sets a deadline: people take turns with gaps of about 200 ms, and a
reply that starts a second late sounds hesitant, while a capable model with the
whole context and every tool takes several seconds to decide anything and
longer to answer once a lookup returns. A small model is fast but lacks the
context and tools, and when asked for a fact it tends to produce a plausible
one. So split the agent's side: a fast model holds the conversation and never
states a fact it was not given, and a slow model with everything does the work.
Two designs of the split were built. **(A) relay**: the slow model passes the
fast one data only (a fact, a question to ask, a notification), and the fast
model chooses every word the person hears. **(B) slow speaks**: the slow model
writes every substantive line, spoken word for word, and the fast model only
buys time: it answers pure small talk itself (and the slow model's run for that
turn is cancelled), says a brief filler that gives no information, or replays
the slow model's interrupted or last line word for word.

### The conditions it is built for (from its own reasoning)

- Multi-turn spoken conversations where the first word should come within
  about a second, and silence of several seconds sounds like a dropped line.
- Many turns need a lookup or an action that needs tools and the person's
  records; some need several steps (a meeting's attendee, then their number), a
  choice between near-duplicates (two Priyas), or picking the latest of
  several records (a code that changed).
- The conversation moves on while the slow model works: small talk while
  waiting, a second question asked before the first is answered, a question
  withdrawn, the person cutting in while the agent speaks, follow-ups that refer
  back ("and their email?") or ask to hear something again.
- A capable model that is much slower to its first word than a small one
  (reasoning is mandatory for Claude Sonnet 5.5 and GPT-6.1 Sol; the small
  models can run without it).

### Arms

Every arm has the same persona, speech-style rules, the same fact rule ("never
state a fact that is not in the conversation, a tool result or a note"), the
same tools (or the same records, for `fast_context`), the same person scripts,
world, timing model and models; they differ only in who speaks and who
thinks.

| Arm | Who speaks | Who uses tools |
|---|---|---|
| `relay` (design A) | fast model, every word; defers briefly on anything it does not have; works the slow model's notes into what it says | slow model, which sends the fast one notes that carry data only (it is told never to write lines to say and never to answer small talk) |
| `slow_speaks` (design B) | slow model's lines, verbatim; fast model answers pure small talk (cancelling the slow run for that turn), says a filler, or replays the slow model's last or interrupted line word for word | slow model (told never to add a filler of its own) |
| `slow_only` | slow model, with tools; says a brief holding line in the same response as a tool call | slow model |
| `fast_only` | fast model, with tools; same prompt as `slow_only` | fast model |
| `fast_context` | fast model with all of the person's records in its context (prompt-cached); only the action tool (`send_text`) | fast model |
| `slow_filler` (secondary) | `slow_only` plus a canned filler ("One moment.") played when nothing has been said 1.2 s after the person stops | slow model |

`slow_filler` is the simplest way to mask a slow model's latency without a
second model; it isolates what B's fast model adds beyond a canned filler. It
runs at the realistic size only, after the primary arms, budget permitting.
"Everything in one long context" for the slow model is not run (cost); for the
fast model it is `fast_context`.

**Loop policy, the same for every arm.** A *voice loop* (the model whose output
is spoken: `slow_only`, `fast_only`, `fast_context`, `slow_filler`, the fast
model in A and B) runs one call at a time; a new event (the person finishes a
turn, a tool result, a note) that arrives before the in-flight call's first
token cancels it and starts a fresh call with everything; one that arrives
later waits for the call to finish. Queued agent speech that has not started
when the person starts speaking is dropped (it answered an older state), except
B's verbatim lines, which wait for the floor, as in design B. The slow model in
A and B is a *worker*: it never cancels an in-flight call; events that arrive
meanwhile are batched into its next call (rapid speech must not stop it from
finishing). Tools run asynchronously in all arms: the conversation carries on
while they run, and their results arrive as events.

The two designs as published used a separate stale-guidance check (A) and two
fast-model calls per turn (B); the minimal versions here do without both: in A
the fast model, which hears every word, decides whether a note is still wanted;
in B the fast model's one call returns its label and its line together.

### Simulation and timing model

Conversations are simulated in text in virtual time, by a discrete-event
simulator. Model calls are real calls; their timing comes from an explicit
model, so arms differ in timing only through their architecture and the tokens
their calls use:

- A model call started at `t0` has its first visible token at
  `t0 + a_m + b_m * prompt_ktokens + reasoning_tokens / r_m` and finishes at
  `t0 + a_m + b_m * prompt_ktokens + completion_tokens / r_m` (completion
  includes reasoning). Speech from a call can start at its first visible token
  plus 0.25 s of text-to-speech latency; its tool calls are dispatched when it
  finishes. `a_m`, `b_m` and `r_m` per model come from a calibration run
  (`calibrate.py`, step 0 below) of streamed calls at several prompt sizes, with
  the prompt prefix cached as it is in the runs; they are written to
  `data/timing.json` and are not changed afterwards.
- The agent speaks at 0.40 s per word (digits in groups count as half a word
  per digit), the person at 0.35 s per word; the agent hears a turn 0.5 s after
  the person stops (end-of-turn detection); when the person starts speaking
  over the agent, the agent stops 0.2 s later and only the words spoken by then
  were heard.
- Tools take a base latency (contacts 0.7 s, calendar 0.9 s, invoices 1.2 s,
  trips 1.0 s, message search 2.2 s, read message 0.6 s, send text 1.5 s)
  times a log-normal factor (sigma 0.35) drawn from a hash of the script,
  run seed, tool and arguments, so the same call takes the same time in every
  arm.

The person is scripted. Each turn has a text and a trigger: after any agent
speech (plus a gap, with a patience limit), after a substantive agent line
(not a holding line), after the answer to named questions has been heard, or
cutting into the agent's answer a set time after it starts. The scripts never
change with the arm; only their timing does.

### Data

`gen.py` generates, from a seed, a world of Sam Rivera's records (contacts,
calendar, messages, invoices, trips, with invented names and numbers) and
conversation scripts. Every script has a greeting, five episodes and a goodbye
(about 13 to 16 person turns and 6 questions). Episode kinds: a simple lookup;
a lookup with small talk while it runs; a second question asked while the
first runs; a lookup withdrawn while it runs; the person cutting into the
answer (to hear it again, or to pause and say go on); a request to hear a fact
again; a follow-up that refers back ("and their email?"); a multi-step lookup;
and sending a text. Questions are of eleven kinds, easy (a contact's mobile, a
meeting time, a flight number, an unpaid invoice, a booking reference, a
meeting place) and hard (a mobile between two contacts with the same first
name, the latest of two codes, a meeting's attendee then their mobile, the
unpaid invoice of a contact's company, the email of the person just
discussed). The facts each question needs are planted in the world, and a
reference solver checks every answer is reachable through the tools and
unambiguous.

Sizes (the same scripts and planted facts; only the surrounding records
change): `small` (20 contacts, 15 events, 25 messages, 8 invoices, 3 trips,
about 4k tokens as text), `medium` (100, 70, 200, 40, 6; about 20k) and
`realistic` (300, 200, 700, 100, 12; about 70k, below the 100k-token price
step). A `sanity` condition (a tiny world, two simple lookups, no mid-lookup
turns, generous patience) must be passed by every arm first.

### Measures

1. **Time to the first word**: for each person turn that expects a reply, from
   the end of the person's speech to the start of the next agent speech;
   median, p90, and the share of such turns with more than 2 s or no reply
   before the person speaks again.
2. **Silent gaps**: per conversation, the longest stretch with nobody
   speaking (before the goodbye), and the silence while each answered question
   was outstanding.
3. **Correct answers delivered**: share of still-wanted questions whose answer
   (a normalised key: digits of a phone number or amount, a code, a time, an
   email, a street) was heard in full by the end of the call; time to answer.
4. **Wrong answers**: share of still-wanted questions where the agent stated
   another value for the fact asked (judge, given the gold answer).
5. **Invented facts**: values the agent said (numbers, times, amounts, codes,
   emails, flight numbers) that appear nowhere in the world or in what Sam
   said (programmatic), plus facts or completed actions the judge finds
   unsupported by the records and tool results; share of conversations with
   any.
6. **Stale answers**: share of withdrawn questions whose answer was said after
   the withdrawal.
7. **Double replies and contradictions**: Sam turns that got two separate
   substantive replies, and agent statements that contradict an earlier one
   (judge), plus a programmatic count of redundant holding lines (more than
   one per question).
8. **Naturalness**: the judge's 1-5 score of the call as heard, with its
   timing, without knowing the arm. The judge is checked by hand on a blinded
   sample of at least 24 calls across arms (my labels against its labels).
9. **Cost** per conversation from the ledger (every role), calls and tokens;
   real API latency per role from the ledger, for reference.

Intervals: Wilson for proportions; differences between arms with the paired
bootstrap over scripts (`demos_common.stats.bootstrap_diff`, each script's
seeds and questions kept together).

### Models

- Primary: fast `anthropic/claude-haiku-5.5` with reasoning disabled; slow
  `anthropic/claude-sonnet-5.5` at reasoning effort `low` (its minimum;
  reasoning is mandatory).
- Second family, on the headline comparison (realistic size): fast
  `openai/gpt-6-luna` (effort `none`), slow `openai/gpt-6.1-sol` (effort
  `low`, its minimum).
- Judge: `google/gemini-3.8-flash`, a third family, so it never judges its own
  family.
- Sensitivity, the "slow brain" premise of the original post: the slow model
  at Sonnet 5.5's default effort (`high`) in the three slow arms, realistic
  size.

No temperature is set (the Anthropic and OpenAI models here do not take it);
the run seeds are repeats.

### Items, seeds, order of runs and budget

0. Calibration of the timing model (`calibrate.py`), then the `sanity`
   condition (all arms), then a smoke run (2 scripts x 1 seed x all arms at
   `small` and `realistic`), reading every transcript of every arm. Bugs are
   fixed and what they touched is re-run; discarded runs go to
   `runs/discarded/` with the reason.
1. Primary: 16 scripts x 3 seeds x 5 arms at `small` and at `realistic`.
2. Second family: the 5 arms at `realistic`, 16 scripts x 2 seeds.
3. `slow_filler` at `realistic`, 16 scripts x 3 seeds.
4. Sensitivity (slow effort `high`): the three slow arms at `realistic`, 16
   scripts x 1 seed.
5. Sweep point `medium`: all 5 arms, 16 scripts x 1 seed.
6. Judge every conversation; hand-check the judge.

Budget: aim under $50, stop before $80. After the smoke run the measured cost
per conversation sets the scale: if 1-6 would exceed $50, steps 5, 4, 3 and 2
are cut back in that order (fewer scripts or seeds), and only then the primary
(to 12 scripts). This rule uses cost only, never results.

### What counts as the idea winning

At the realistic size with the primary models, a split design "holds" if, against
`slow_only`, (a) its median time to the first word is at least 0.5 s lower and
its share of turns waiting more than 2 s is lower (interval excluding 0);
(b) its correct-answer rate is within 5 points of `slow_only`'s and its invented
facts are no more frequent; and (c) its judged naturalness is not lower
(interval including 0 or above it) and its double replies and contradictions
are not more frequent by more than 0.5 per call. It "pulls ahead" of the single
fast model where, in addition, its correct-answer rate is higher than
`fast_only`'s and `fast_context`'s, or its invented facts fewer (intervals
excluding 0). The idea does not hold where `fast_only` or `fast_context` match
`slow_only`'s correctness at the fast model's latency, or where the split's own
failures (double replies, lost or stale answers, contradictions) leave it worse
than `slow_only` on correctness or naturalness. Between the designs, one beats
the other if it has fewer double replies, contradictions and lost answers
(interval excluding 0) without more stale answers or lower naturalness.

### Plan amendment 1 (2026-10-09 22:47 UTC, after calibration, before any experiment run)

**The fault.** The plan assumed that the slow models' mandatory reasoning
would make them much slower to their first word than the small ones, so that
the measured timing would contain the idea's central condition. The
calibration (`calibrate.py`, `runs/calibration.jsonl`, `data/timing.json`)
says otherwise. Through OpenRouter, with the prompt prefix cached, every
model's first visible token comes at about the same time:

| Model (setting) | first token, median | fitted a | b per 1k prompt tokens | r (tokens/s) | reasoning tokens on short tasks |
|---|---|---|---|---|---|
| Claude Haiku 5.5 (reasoning off) | 1.06-1.30 s | 1.10 s | 0.003 s | 128 | 0 |
| Claude Sonnet 5.5 (effort low) | 1.00-1.48 s | 1.18 s | 0.002 s | 89 | 0 |
| Claude Sonnet 5.5 (effort high) | 1.05-1.31 s | (same model) | | | 0 (but ~560 and 7.8 s on a 150-word task) |
| GPT-6 Luna (effort none) | 1.07-1.27 s | 1.16 s | 0 | 104 | 0 |
| GPT-6.1 Sol (effort low) | 1.14-1.36 s | 1.29 s | 0 | 56 | 0 (38 on a 150-word task) |

"Mandatory" reasoning is adaptive: on short turns the slow models think for 0
tokens. A probe (`probe.py`) also showed that neither Sonnet 5.5 nor the
OpenAI models say a holding line in the same response as a tool call,
although the prompt asks for one; Haiku 5.5 does. Prefill with a cached
prefix costs almost nothing (0.2 s at 75k tokens), so `fast_context` pays
little for its long context.

**The change.** The measured timing stays the primary condition ("today,
through OpenRouter"). Because it lacks the idea's own premise (the post
measured 8.9 s for the slow model to decide), a **slow-brain delay sweep** is
added: a fixed delay D added to the slow model's time to first token on every
slow-model call in every arm that uses it (`Timing.delay`), standing for a
capable model that thinks, reads a long uncached context, or runs on a slower
provider. D = 0 (measured), 4 s and 8 s (8 s brings a decision to about the
post's 8.9 s). The fast-only arms are unaffected by D, so they are run once.
The `slow_filler` arm is run at every D (a canned filler is the simplest
answer to a slow model). The run order becomes:

1. Primary: 5 arms (+ `slow_filler`) at `realistic`, D = 0, 16 scripts x 3 seeds;
   and at `small`, D = 0, 16 x 3.
2. Delay sweep at `realistic`: the slow arms (`relay`, `slow_speaks`,
   `slow_only`, `slow_filler`) at D = 4 and D = 8, 16 scripts x 3 seeds.
3. Second family at `realistic` (D = 0 and D = 8), 16 scripts x 2 seeds.
4. Sonnet at effort `high` (D = 0), the slow arms, 16 x 1.
5. `medium` size, all arms, D = 0, 16 x 1.

The criteria for the idea holding are unchanged, and are applied at each D:
"holds at D" if (a)-(c) hold there. Budget rule as before (cut 5, 4, 3 in that
order, then seeds of 2).

### Plan amendment 2 (2026-10-09 22:55 UTC, after the first smoke run, before any main run)

Two secondary arms strengthen the single-fast baselines: `fast_only_think` and
`fast_context_think` are `fast_only` and `fast_context` with the fast model
allowed to reason (Haiku 5.5 at effort `low`; GPT-6 Luna at effort `low` for
the second family). Reason: with reasoning off, Haiku doing the lookups itself
spoke its reasoning aloud, and a competent engineer handing the small model
the tool work would let it think. The fast model inside `relay` and
`slow_speaks` keeps reasoning off, as the idea specifies (it only talks). The
timing model counts their reasoning tokens like any others. They run wherever
`fast_only` runs.

### Plan amendment 3 (2026-10-10 00:20 UTC, from the lead, before any of these runs; superseded by amendment 4)

**Reason (Dan, via the lead).** Smarter models need thinner harnesses, so the
fair test of a design is on the models it was built for, and the two designs
were built for different models. The model pair becomes an axis, separate from
the design, so that a difference between designs is not a difference between
models.

**The change.**

1. **The February pair** (what design A, the relay, shipped with): fast
   `openai/gpt-5-mini` at reasoning effort `minimal`, slow
   `anthropic/claude-opus-4.6` at effort `low`. Read in the original code at
   the February pin `95999dee5`: `medium_scripts/call.py` builds the fast
   brain as `UnifyLLM(model="gpt-5-mini@openai", reasoning_effort="minimal")`,
   and the slow brain is `new_llm_client(SETTINGS.UNIFY_MODEL)` with
   `UNIFY_MODEL = "claude-4.6-opus@anthropic"` and a client default of
   `reasoning_effort="low"`.
2. **The summer pair** (design B, the slow model speaks): slow
   `openai/gpt-5.6-terra` at effort `high`, fast `openai/gpt-5.4-mini` at
   effort `low`. Read in the original code at `27e50e28f` (11 July 2026, "Route
   ConversationManager slow brain through gpt-5.6-terra@openai"):
   `SLOW_BRAIN_MODEL = "gpt-5.6-terra@openai"` with
   `SLOW_BRAIN_REASONING_EFFORT = "high"`, and `FAST_BRAIN_MODEL =
   "gpt-5.4-mini@openai"` called with `reasoning_effort="low"`. The lead said
   to use GPT-5.6 Luna as the era's fast model unless the code used another;
   the code used GPT-5.4 mini, so it is used.
3. On **both pairs**: both designs (`relay`, `slow_speaks`) and the
   single-model baselines (`slow_only`, `fast_only`, `fast_context`; and
   `slow_filler` if the budget allows), at measured timing (D = 0), with each
   model's timing calibrated as in step 0. Size: `realistic` if it fits the
   budget, otherwise the largest that does, with the cost of the full size
   reported. Scripts and seeds: as many as fit (16 scripts x 2 seeds if
   possible; never fewer than 16 x 1), decided from the smoke run's cost per
   call, never from results.
4. **The runs so far stay** as a labelled condition, "today's tier" (fast
   `anthropic/claude-haiku-5.5`, slow `anthropic/claude-sonnet-5.5`): `small`
   and `realistic` at D = 0 and `realistic` at D = 8. Not run, for the budget:
   D = 4, today's tier on the second family, Sonnet at effort `high`, and the
   `medium` size. The queue on the cloud machine was stopped at 00:15 UTC as
   D = 4 began (nothing of it was recorded; its calls are in the ledger).
5. **If the budget allows, today's frontier pair** (fast
   `anthropic/claude-haiku-5.5`, slow `anthropic/claude-opus-5.5`) on the
   headline comparison only: `relay`, `slow_speaks` and `slow_only` at
   `realistic`, D = 0 (`fast_only` and `fast_context` with Haiku 5.5 are the
   today's-tier runs, the same model and prompts).
6. The judge stays `google/gemini-3.8-flash`, a family different from every
   model under test.
7. **Budget**: a hard stop at $80 for the whole concept, including the $26.35
   spent by 00:15 UTC. Order: calibrate, probe and sanity-check the era models;
   a smoke run per pair, read; the era pairs' main runs; judging every
   condition; then the frontier pair if what is left allows.

The criteria for the idea holding are applied per pair, and the design
comparison is read within each pair.

**Carrying it out (from 00:20 UTC).**

- *Timing for models that reason.* The era models reason on many turns, and
  the calibration showed that one output rate cannot stand for both reasoning
  and visible tokens: Opus 5.5 delivers its visible text in one burst after
  reasoning (a fitted rate of 2,030 tokens/s), Opus 4.6 reasons at about 32
  tokens/s but streams text at about 70. The timing model gained a separate
  reasoning rate `rr` (first visible token at `a + b * prompt_ktokens +
  reasoning / rr`), fitted by least squares on calls that make each model
  reason (`calibrate.py --reasoning-only`, a counting question over the
  records). The today's-tier fits are kept exactly as their runs used them
  (`runs/discarded/timing.v1-single-rate.json` is the file before the change);
  their slow model reasoned on 1.8% of calls, so the change does not touch
  them, except the `_think` arms: Haiku 5.5 reasons at 145 tokens/s where 128
  was used, which made their reasoning calls about 0.2 s too slow (reported as
  a caveat, not re-run).

  | Model (setting) | a | b per 1k prompt tokens | r (visible) | rr (reasoning) |
  |---|---|---|---|---|
  | GPT-5 mini (minimal) | 0.70 s | 0.007 s | 145 | (does not reason) |
  | Claude Opus 4.6 (low) | 1.34 s | 0.121 s | 70 | 32 |
  | GPT-5.4 mini (low) | 0.94 s | 0.003 s | 150 | 149 |
  | GPT-5.6 Terra (high) | 0.89 s | 0 | 114 | 108 |
  | Claude Opus 5.5 (low) | 3.72 s | 0.003 s | burst | 99 |

- *Sanity* (both pairs, every arm including `slow_filler`): 2 of 2, nothing
  invented, in every arm. *Smoke* (2 scripts, `realistic`, five arms each),
  every transcript read: no new harness fault. GPT-5 mini alone with the tools
  fails hard lookups the way the idea expects (it searched invoices under a
  person's name and said "We don't have any unpaid invoices for Amara Marsh's
  company"; it said "one sec, I'll pull their number" without calling the
  tool); Opus 4.6, unlike the newer slow models, says a holding line with its
  tool calls.
- *Budget decision, from the smoke run's cost only.* A call costs about $0.31
  in each February slow arm (Opus 4.6) and $0.15 in each summer slow arm
  (Terra at high effort); five arms cost $0.98 per script for the February
  pair and $0.57 for the summer pair. With $48.52 left at 00:25 UTC and about
  $8 needed to judge every condition, both pairs run at `realistic`, 16
  scripts x 1 seed (about $25). 16 x 2 would cost about $50 and does not fit;
  `slow_filler` is not run on the era pairs. The frontier pair runs only if at
  least $9 remains after judging.

### Plan amendment 4 (2026-10-10 00:38 UTC, from the lead; supersedes amendment 3)

**Reason (Dan, via the lead).** Only today's models that someone would
reasonably choose today, on the cost and quality frontier, and no era models
at all. Model strength stays an axis.

**The change.**

1. **Fast model**: today's fast tier, `anthropic/claude-haiku-5.5` (reasoning
   disabled) and `openai/gpt-6-luna` (effort `none`), the least reasoning each
   API allows.
2. **Slow model, three tiers**: *cheap*, the same two models at the same
   settings (so in this tier `slow_only` and `fast_only` are one arm); *mid*,
   `anthropic/claude-sonnet-5.5` and `openai/gpt-6.1-sol` (effort `low`, their
   minimum); *top*, `anthropic/claude-opus-5.5` at effort `high`.
3. **In full on one pairing per family, the cheap tier**: both designs and the
   single-model baselines, `realistic`, 16 scripts x 3 seeds: Haiku 5.5 with
   Haiku 5.5, and GPT-6 Luna with GPT-6 Luna. The Anthropic family's
   single-model baselines are the Haiku runs already made (`fast_only`,
   `fast_context` and their `_think` forms, the same model and prompts), so
   only `relay` and `slow_speaks` are new there; the OpenAI family runs
   `relay`, `slow_speaks`, `fast_only` and `fast_context`.
4. **The headline comparison as the slow model gets stronger, one family**
   (Anthropic): the *mid* tier is the runs already made with Sonnet 5.5 (in
   full: both sizes, D = 0 and D = 8); the *top* tier runs `relay`,
   `slow_speaks` and `slow_only` with Opus 5.5 at effort `high`, `realistic`,
   D = 0, with as many scripts as fit (1 seed), its single-fast baselines
   being the Haiku runs.
5. **Era runs set aside.** Everything run on era models after amendment 3
   (calibration rows, probes, sanity, smoke and the main runs of the February
   and summer pairs, $31.26 of the spend) is kept in `runs/excluded-era/` and
   left out of the analysis.
6. **Budget**: the hard stop at $80 stands; $57.98 was spent by 00:37 UTC,
   leaving $22.02. Priority: judging the mid tier, the cheap tiers in full
   with their judging, then the top tier with what is left; whatever does
   not fit is reported with what it would cost.

## How it is built

Everything is in `src/fast_slow_voice/` (about 2,500 lines) with 35 tests in
`tests/` (stubbed model calls, no network).

**World and questions (`world.py`, `script.py`).** A seeded generator builds
Sam Rivera's records (contacts with 555-01xx numbers, calendar, emails and
texts, invoices, trips) and plants the facts each scripted question needs:
eleven question kinds, easy (a contact's mobile, a meeting time or place, a
flight number, a vendor's unpaid invoice, a booking reference) and hard (the
right one of two contacts with the same first name, the newer of two gate
codes, a meeting's attendee and then their mobile, the unpaid invoice of a
contact's company, the email of the person just discussed). Seven tools read
the records (`find_contacts`, `calendar`, `search_messages`, `read_message`,
`invoices`, `trips`) or act (`send_text`). A reference solver answers every
planted question through the tools and checks the answer is unambiguous at
every size (all pass, for three base seeds). The 16 scripts (`data/scripts-1.json`,
SHA-256 in `data/scripts-1.sha256`) each have a greeting, five episodes and a
goodbye: 14 to 17 turns of Sam's and 5 to 8 questions; 98 questions in all
(88 lookups, of which 9 are withdrawn while they run, and 10 requests to hear
an answer again; 47 easy and 41 hard lookups) and 9 texts to send. The same
scripts and planted facts are used at every size; only the surrounding records
change (`small`: 20 contacts and about 4k tokens of records; `realistic`: 300
contacts, 200 events, 700 messages, about 78k tokens, so `fast_context`'s
prompt is 79.5k tokens).

**Simulation (`sim.py`, `timing.py`).** A discrete-event simulator runs each
call in virtual time. It owns the floor: Sam's turns, the queue of agent
speech, barge-in (the agent stops 0.2 s after Sam starts over it, and only the
words spoken by then count as heard), end-of-turn detection (0.5 s), tools
(their results arrive after a latency drawn from a hash of the call, so a call
takes the same time in every arm) and the model calls. Every model call is a
real call through `demos_common.LLM`; its first visible token and its end are
placed in virtual time by the timing model, from the tokens the call actually
used (`data/timing.json`). Before processing any event, the simulator awaits
every call started at or before it, so no call's output can be overtaken by a
later event. Sam is scripted: each turn's trigger says when Sam speaks (after
any agent speech, after a substantive line, after named answers have been
heard, or cutting into an answer 1.2 s after it starts); the words never
depend on the arm.

**Arms (`arms.py`, `prompts.py`).** All arms share the persona, the spoken
style, the fact rule, the tools and one silence marker (`[silent]`). Each model
call sees the call so far as a timestamped transcript, one line per event, as
that role could know it (tools and notes only where the role has them), sent
as one content block per line so the provider's prompt cache reuses earlier
calls; Anthropic calls mark the cache explicitly. A voice loop (any model whose
words are spoken) runs one call at a time and cancels a call that a new event
overtakes before its first token; the slow model behind a fast voice is a
worker that never cancels and batches what arrives meanwhile. Queued speech
that has not started when Sam takes the floor is dropped, except the slow
model's verbatim lines in `slow_speaks`, which wait for the floor.

- `relay`: the fast model hears every turn and replies; asked for anything it
  was not given, it says one brief line that it will check. The slow model hears
  the same turns, uses the tools, and its reply (unless `[silent]`) becomes a
  note the fast model sees and works into what it says.
- `slow_speaks`: the fast model labels each of Sam's turns in one call
  (`SMALL_TALK` with a full reply, `HOLD` with a filler, `REPLAY`, or `NONE`);
  on `SMALL_TALK`, `REPLAY` and `NONE` the slow run for that turn is cancelled.
  `REPLAY` replays the slow model's last line word for word, from the start
  of the sentence Sam cut into ("Sorry, as I was saying. ..."). The slow
  model's text is queued verbatim and spoken when the floor is free.
- `slow_only`, `fast_only`, `fast_only_think`: one model with the tools, told
  to say one brief holding line in the same response as a tool call.
- `fast_context`, `fast_context_think`: the fast model with all the records in
  its (cached) system prompt and only `send_text`.
- `slow_filler`: `slow_only` plus a canned filler ("One moment.") played when
  nothing has been said 1.2 s after Sam stops and the model is still working.

**Measures (`metrics.py`, `keys.py`, `judge.py`).** Computed from what Sam
heard. An answer counts when its normalised key (digits of a phone number or
amount, a code, a time, an email, a flight number with the airline's code or
name, an address's number and street) is heard in full in one agent line after
the question; a withdrawn question's answer heard after the withdrawal is
stale. Invented values are numbers, amounts, times, codes, flight numbers,
references and emails the agent said that appear nowhere in the records or in
what Sam said. The judge (Gemini 3.8 Flash, a third family, effort medium)
reads the call as Sam heard it, with timings and silences and no arm label,
plus the questions' correct answers, and returns naturalness (1-5, with a
rubric), double replies, contradictions, wrong or invented facts and stale
answers (`judge_run.py`). It was checked by hand (`validate_judge.py`).

## Checks before trusting

- **Tests.** `uv run pytest` (35 tests, no network): planted answers reachable
  and unambiguous at every size; tools; answer keys; the timing model; barge-in
  and dropped versus waiting speech; calls resolved before later events; the
  voice loop's cancellation and the worker's batching; the relay's notes; the
  slow-speaks labels, the cancelled slow run and the replay (including a replay
  decided while the slow model's own line waits, and one after the slow model
  has spoken again); the canned filler;
  silence markers and cleaned speech; made-up tools ignored; Sam's triggers;
  the measures on hand-built calls; the judge's input carrying no arm. And every
  arm, driven by an oracle that answers perfectly, delivers every answer with
  nothing invented, on three scripts.
- **Sanity condition** (a tiny world, two simple lookups, nothing happening
  while they run): every arm 2 of 2 with nothing invented, with the primary
  models (five times, after each fix) and the second family.
- **Smoke runs** (two scripts, both sizes, every arm), and every transcript of
  every arm read, with the judge's labels beside them. They found the bugs
  below; `runs/discarded/README.md` keeps each discarded run with its reason.

**Bugs found and fixed before the main runs** (each fix applies to every arm):

1. Asked to "reply with no text", Haiku 5.5 spoke its reasoning instead ("Sam
   said 'Okay, thanks,' which doesn't need a reply. I'll stay quiet."); the
   relay's slow model returned `<br>`, which became a note, and the fast
   model then spoke a second time. Now one silence marker, `[silent]`, in every
   prompt, and empty-looking replies count as silence.
2. The relay's slow model called a made-up `noop` tool, which was run and
   answered with an error. Tools an arm does not have are now ignored.
3. "Ava:" labels, and commentary before them, reached the speech. A leading
   speaker label (and anything before it) is now dropped from spoken text; the
   per-call instruction asks for "exactly the words Ava says now, or [silent]".
4. "United 2375" did not count as the flight UA 2375, so a correct answer was
   scored as missed and the simulated Sam, waiting for it, sat through 9.5 s of
   silence. Flight numbers said with the airline's name now count.
5. A "board dinner" was planted at 9:20 am; it is now a "board meeting".
6. In `slow_speaks` the slow model wrote the same line twice while its first
   copy waited, queued, for Sam to stop talking. Queued speech is now marked
   "already queued ... never say them again" in every transcript, and words
   identical to words already waiting are not queued again.
7. The simulated Sam's "Got it, thanks" after an aside waited for the agent to
   speak again even when the answer had already been given, so an agent that
   rightly stayed quiet got 25 s of silence. Only the acknowledgement after Sam
   cuts in now needs a fresh reply.
8. The judge blamed Ava for Sam's scripted "you cut out there"; it is now told
   Sam's lines are scripted and only Ava is judged.
9. GPT-6 Luna repeated its label line in `slow_speaks` ("HOLD: Sure, one
   sec.\n\nHOLD: Sure, one sec.") and the whole text was spoken. The parser now
   takes the line after the first label, on that line only (found in the
   second family's sanity run, fixed before its main run; Haiku never produced
   a multi-line label).

**Bugs found after the main runs**, each fixed for every arm, with exactly the
calls it could change re-run and re-judged (`runs/discarded/README.md` keeps
every replaced call and judgement with its reason):

10. Narration before the silence marker was spoken ("Sam has said goodbye, so
    there's nothing more to say. [silent]"): only a trailing marker was
    dropped. A reply with the marker anywhere is now silence. 36 calls re-run
    (`fast_only` 28, `fast_context` 5, `relay` 3).
11. In `slow_speaks`, a REPLAY replayed the slow model's newest line, just
    spoken anew, instead of the line Sam cut into. The replay now targets the
    line Sam cut into and is skipped when the slow model has spoken again or
    has a line waiting. 7 calls re-run.
12. The fix in 11 lost answers when the slow model is as quick as the fast one
    (found by the second agent, 10 October 02:05 UTC, reading the cheap
    tiers): the slow model's line for Sam's "say that again?" was already
    waiting when the REPLAY label arrived, so the replay was skipped, and then
    the label cancelled that run and dropped the line, so Sam heard nothing (26
    of 30 requests to repeat went unanswered in the OpenAI cheap tier). A
    waiting line of the run the label cancels no longer stops the replay. 58
    calls re-run (cheap tiers only; no Sonnet or Opus call had it).
13. GPT-6 Luna sometimes garbled the silence marker (`[ѕilent]` with a Cyrillic
    s, `[ლsilent]`), which was then spoken or sent as a note. Any bracketed
    marker ending in "ilent" is now silence. 11 calls re-run (OpenAI cheap tier
    only; 5 of them also in 12).
14. The judge crashed on a call that ended while Ava was speaking; such lines
    now end "[the call ended here]" in what the judge reads. No run changed.
15. In the analysis, the median and p90 of the time to the first word left out
    the turns Ava did not answer before Sam spoke again, which are the longest
    waits (all at least 3 s), so `slow_only`'s median read below 2 s while most
    of its turns waited longer. They now count as the wait until Ava's next
    words, as the plan defines the measure; the Results show what the first
    version would have concluded.

Also seen and kept, as the models' own behaviour under each design: Sonnet
5.5, Opus 5.5, GPT-6.1 Sol and GPT-6 Luna rarely say a holding line in the same
response as a tool call (Haiku 5.5 almost always does; Caveat 3 gives the
shares), so `slow_only` is silent until an answer is ready; Haiku without reasoning, doing the tool work itself, sometimes speaks
its reasoning aloud; the relay's slow model sometimes sends a fact the fast
model has just said, and the fast model says it again; in `slow_speaks`, a
line written just before Sam withdraws a question is spoken after the
withdrawal, since verbatim lines wait for the floor.

## Handover (10 October 2026, 02:00 UTC)

The first agent was cut off by a usage limit at 01:45 UTC while writing up.
The second agent found nothing of this concept running on the cloud machine,
every run of Plan amendment 4 recorded and judged, and every record and both
ledgers on the laptop (checked file by file against the machine). It then read
transcripts of every arm in the cheap and top tiers, found bugs 12 and 13 and
the analysis fault 15 above, fixed them, re-ran and re-judged the 66 affected
calls ($1.23), and wrote the Results. Budget at handover, from the lead: the
era-model runs of Plan amendment 3 are outside the analysis and outside the
cap, which applies to today's models (stop before $80). No run beyond Plan
amendment 4 was added; "Not run" below prices what was left out.

<!-- results:start (generated by report_tables.py) -->
## Results

Every number below is generated from `results.json` by `report_tables.py`; `runs/analysis.txt` has the full tables for every condition, with every paired difference. Intervals are 95%: Wilson for proportions, a paired bootstrap over the 16 scripts for means, medians and differences. "Correct answers" counts still-wanted questions whose answer Sam heard in full; "calls with a wrong or invented fact" is the share of calls where the judge found a wrong or unsupported fact or the programmatic check found a value that is nowhere in the records; "first word" is from the end of Sam's turn to the start of Ava's next words, counting a turn Ava left unanswered until Sam spoke again as the wait until her next words.

### Verdict

**By the plan's own criteria the idea holds in part, and not against the strongest simple alternative.** Design B (`slow_speaks`) holds against a single slow model at the headline condition and pulls ahead of the two single-fast arms the plan named (`fast_only`, `fast_context`); design A (`relay`) holds only with Opus 5.5 as the slow model. But with today's models the slow model is not slow to its first word (about 1.2 s, as the fast model; Plan amendment 1), so the split's advantage over a single slow model comes from saying something while the tools run, which a canned filler gives as well; and the fast model with all the records in its prompt and reasoning on, which a competent engineer would try first at this size, is as correct, answers sooner, is judged more natural and costs less.

A single slow model with the tools (`slow_only`) is slow only because Sonnet 5.5 and Opus 5.5 almost never say anything while a lookup runs (a holding line with 6% and 8% of their tool calls; on the turns it does answer at once, Sonnet starts in 1.94 s): its median first word is 4.47 s against 1.85 s for `relay` and 1.88 s for `slow_speaks`. The same slow model with a canned "One moment." (`slow_filler`) starts at 1.20 s, with the same correct answers (99%), no withdrawn answer said, and a lower cost per call ($0.034 against $0.040 and $0.048); what B's fast model adds over it is a better-judged call (+0.52 [+0.33, +0.71] naturalness) at the price of withdrawn answers said (+59.3 [+29.6, +85.2] points). The fast model with all of Sam's records in its prompt and reasoning on (`fast_context_think`, 79.5k tokens at the realistic size) answered 100% of questions, with a median answer time of 2.9 s, against 99% in 6.5 s for `slow_speaks` and 99% in 8.8 s for `relay`; the judge found it more natural (3.98 against 3.52 and 2.85), and it cost $0.023 a call. What it gives up is the quick first word: it reads and thinks before it says anything (median 2.82 s, p90 7.4 s) and says nothing meanwhile. Without reasoning (`fast_context`) it starts at 2.06 s but answered 95% and stated a wrong or invented fact in 46% of calls.

Against `slow_only`, the plan's criteria (a)-(c) hold for `slow_speaks` in the mid tier, `realistic`, D = 0; `slow_speaks` in the mid tier, `small`, D = 0; `slow_speaks` in the mid tier, `realistic`, D = 8 s; `relay` in the top tier, Anthropic. Design A (`relay`) fails criterion (c) wherever Sonnet 5.5 or Haiku 5.5 is the slow model: its slow model's notes repeat what the voice has already said and the voice says them again (1.25 double replies per call against 0.04 for `slow_only` at the headline condition). Design B (`slow_speaks`) avoids that (0.04 per call) and is judged the most natural of the slow arms there, but because its lines wait for the floor and are spoken word for word, it said 59% of withdrawn answers anyway (`relay` 4%, `slow_only` 0%). By the plan's rule neither design beats the other at the headline condition (B has fewer double replies but more withdrawn answers said). With Opus 5.5 as the slow model the designs swap places: Haiku labelled most of Sam's "what was that again?" turns HOLD, and Opus, told that the voice replays its lines, stayed silent, so `slow_speaks` answered 4 of 10 requests to repeat (92% correct in all against 99% for `slow_only`), while `relay` held (98% correct, naturalness 3.12 against 2.44). With the same small model on both sides (the cheap tier) neither design holds: Haiku 5.5 alone already says a holding line with its tool calls (first word 1.86 s), so there is nothing to gain, and both designs are judged less natural than it (2.54 and 1.81 against 3.02). GPT-6 Luna alone does not say a holding line (first word 4.31 s), so both designs answer sooner, and correctly a little more often (+4.9 [+1.4, +9.0] and +4.5 [+0.0, +9.3] points), but each said one value found nowhere in the records, which criterion (b) does not allow, and `slow_speaks` said 81% of withdrawn answers; Luna with the records in its prompt (`fast_context`) beat every arm of its family (98% correct, naturalness 4.44, first word 1.91 s).

Where the slow model really is slow (D = 8 s added to every slow-model call, standing for the post's 8.9 s decision), `slow_only` leaves Sam waiting (21.89 s median on requests, 48% of turns unanswered before he goes on) while both designs start in under 2 s; `slow_speaks` holds and is judged more natural than the canned filler (2.50 against 1.92). The fast model with the records does not use the slow model, so its D = 0 results stand there too (100% correct, naturalness 3.98). Where the split pulls ahead, it is of the fast model doing the lookups itself (`fast_only`: +7.5 [+3.5, +11.6] points correct for `relay` and +7.9 [+3.8, +11.9] for `slow_speaks` at the headline condition; `fast_only_think`: +2.6 [+0.4, +5.4] for `slow_speaks`) and of the fast model with the records but no reasoning (`fast_context`: +3.4 [+0.4, +6.6] and +3.8 [+1.1, +6.8] points, and far fewer calls with a wrong or invented fact). Against `fast_context_think` neither design is ahead on anything but the first word.

**Sensitivity of the verdict to one measurement choice.** The first analysis left the turns Ava did not answer before Sam spoke again out of the first-word median (they are in the share of turns waiting more than 2 s either way). Those are mostly `slow_only`'s lookups, so its median fell to 1.94 s while 58% of its turns waited more than 2 s. Counted as the plan defines the measure (the wait until Ava's next words), criterion (a) passes; counted the first way, it fails for `slow_speaks` in the mid tier, `realistic`, D = 0; `slow_speaks` in the mid tier, `small`, D = 0, and the split would hold only where the last column of the criteria table says yes.

### The headline: mid tier (Haiku 5.5 with Sonnet 5.5 at effort low), `realistic` size, measured timing

16 scripts x 3 seeds per arm (48 calls; 267 still-wanted questions and 27 withdrawn ones per arm).

| Arm | Correct answers | Calls with a wrong or invented fact | Withdrawn answers said anyway | Naturalness (judge, 1-5) | Double replies per call | First word, median (p90) | Turns waiting > 2 s | Answer time, median | Cost per call |
|---|---|---|---|---|---|---|---|---|---|
| `relay` | 99% [96, 99] (263/267) | 8% [3, 20] | 4% [1, 18] (1/27) | 2.85 [2.52, 3.23] | 1.25 [0.92, 1.60] | 1.85 s (1.9) | 3% [2, 5] | 8.8 s [8.1, 9.7] | $0.048 |
| `slow_speaks` | 99% [97, 100] (264/267) | 4% [1, 14] | 59% [41, 75] (16/27) | 3.52 [3.33, 3.71] | 0.04 [0.00, 0.12] | 1.88 s (1.9) | 2% [1, 4] | 6.5 s [5.2, 7.3] | $0.040 |
| `slow_only` | 99% [97, 100] (264/267) | 6% [2, 17] | 0% [0, 12] (0/27) | 3.29 [3.10, 3.48] | 0.04 [0.00, 0.12] | 4.47 s (12.6) | 58% [54, 63] | 5.0 s [4.9, 5.4] | $0.034 |
| `slow_filler` | 99% [97, 100] (264/267) | 2% [0, 11] | 0% [0, 12] (0/27) | 3.00 [3.00, 3.00] | 0.00 [0.00, 0.00] | 1.20 s (1.2) | 0% [0, 1] | 5.0 s [4.9, 5.4] | $0.034 |
| `fast_only` | 91% [87, 94] (243/267) | 15% [7, 27] | 0% [0, 12] (0/27) | 3.02 [2.79, 3.25] | 0.15 [0.06, 0.23] | 1.86 s (1.9) | 10% [7, 13] | 5.5 s [5.5, 5.9] | $0.002 |
| `fast_only_think` | 96% [93, 98] (257/267) | 8% [3, 20] | 0% [0, 12] (0/27) | 3.10 [2.94, 3.29] | 0.00 [0.00, 0.00] | 4.98 s (15.1) | 81% [77, 84] | 7.9 s [6.0, 8.4] | $0.003 |
| `fast_context` | 95% [92, 97] (254/267) | 46% [33, 60] | 0% [0, 12] (0/27) | 3.75 [3.35, 4.12] | 0.10 [0.02, 0.21] | 2.06 s (2.1) | 100% [99, 100] | 2.1 s [2.1, 2.1] | $0.022 |
| `fast_context_think` | 100% [99, 100] (267/267) | 2% [0, 11] | 0% [0, 12] (0/27) | 3.98 [3.73, 4.21] | 0.00 [0.00, 0.00] | 2.82 s (7.4) | 100% [99, 100] | 2.9 s [2.7, 3.2] | $0.023 |

| Difference (first arm minus second) | Correct (points) | Calls with a wrong or invented fact (points) | Withdrawn answers said (points) | Naturalness | Double replies per call | First word, median (s) | Turns waiting > 2 s (points) | Answer time (s) | Cost per call ($) |
|---|---|---|---|---|---|---|---|---|---|
| relay minus slow_only | -0.4 [-1.2, +0.0] | +2.1 [-4.2, +8.3] | +3.7 [+0.0, +11.1] | -0.44 [-0.71, -0.15] | +1.21 [+0.88, +1.58] | -2.62 [-2.79, -2.46] | -54.9 [-59.5, -50.6] | +3.7 [+3.1, +4.6] | +0.014 [+0.012, +0.017] |
| slow_speaks minus slow_only | +0.0 [+0.0, +0.0] | -2.1 [-6.2, +0.0] | +59.3 [+29.6, +85.2] | +0.23 [-0.04, +0.50] | +0.00 [+0.00, +0.00] | -2.59 [-2.77, -2.43] | -55.8 [-61.0, -50.6] | +1.5 [+0.2, +2.0] | +0.006 [+0.004, +0.009] |
| slow_speaks minus relay | +0.4 [+0.0, +1.2] | -4.2 [-10.4, +0.0] | +55.6 [+25.9, +85.2] | +0.67 [+0.31, +1.04] | -1.21 [-1.58, -0.88] | +0.03 [+0.03, +0.03] | -0.9 [-4.0, +2.2] | -2.3 [-3.1, -2.1] | -0.008 [-0.012, -0.005] |
| slow_speaks minus slow_filler | +0.0 [+0.0, +0.0] | +2.1 [+0.0, +6.2] | +59.3 [+29.6, +85.2] | +0.52 [+0.33, +0.71] | +0.04 [+0.00, +0.12] | +0.68 [+0.68, +0.68] | +2.4 [+0.6, +4.4] | +1.5 [+0.2, +2.0] | +0.006 [+0.003, +0.008] |
| relay minus fast_context_think | -1.5 [-3.9, +0.0] | +6.2 [-2.1, +16.7] | +3.7 [+0.0, +11.1] | -1.12 [-1.50, -0.73] | +1.25 [+0.92, +1.60] | -0.97 [-1.21, -0.82] | -96.8 [-98.7, -94.5] | +5.9 [+5.2, +6.8] | +0.025 [+0.022, +0.029] |
| slow_speaks minus fast_context_think | -1.1 [-3.5, +0.0] | +2.1 [-6.2, +12.5] | +59.3 [+29.6, +85.2] | -0.46 [-0.81, -0.10] | +0.04 [+0.00, +0.12] | -0.94 [-1.18, -0.79] | -97.7 [-99.4, -95.6] | +3.6 [+2.2, +4.4] | +0.017 [+0.013, +0.021] |
| relay minus fast_context | +3.4 [+0.4, +6.6] | -37.5 [-54.2, -20.8] | +3.7 [+0.0, +11.1] | -0.90 [-1.40, -0.40] | +1.15 [+0.77, +1.54] | -0.21 [-0.21, -0.21] | -96.8 [-98.7, -94.5] | +6.7 [+6.1, +7.6] | +0.026 [+0.022, +0.030] |
| slow_speaks minus fast_context | +3.8 [+1.1, +6.8] | -41.7 [-58.3, -25.0] | +59.3 [+29.6, +85.2] | -0.23 [-0.67, +0.23] | -0.06 [-0.17, +0.02] | -0.18 [-0.18, -0.18] | -97.7 [-99.4, -95.6] | +4.4 [+3.1, +5.2] | +0.018 [+0.014, +0.022] |
| relay minus fast_only | +7.5 [+3.5, +11.6] | -6.2 [-25.0, +10.4] | +3.7 [+0.0, +11.1] | -0.17 [-0.60, +0.29] | +1.10 [+0.75, +1.50] | -0.01 [-0.01, -0.01] | -6.4 [-9.6, -3.4] | +3.3 [+2.6, +4.2] | +0.045 [+0.042, +0.050] |
| slow_speaks minus fast_only | +7.9 [+3.8, +11.9] | -10.4 [-29.2, +6.2] | +59.3 [+29.6, +85.2] | +0.50 [+0.19, +0.81] | -0.10 [-0.21, +0.00] | +0.03 [+0.02, +0.03] | -7.3 [-9.7, -4.6] | +1.0 [-0.3, +1.7] | +0.037 [+0.034, +0.041] |

### As the slow model gets stronger (Anthropic family, `realistic`, measured timing)

The fast model is Haiku 5.5 throughout. The single-fast arms are the same Haiku runs in every tier. The top tier is 16 scripts x 1 seed (Plan amendment 4); the others 16 x 3.

| Slow model (fast: Haiku 5.5) | Arm | Calls | Correct answers | Withdrawn answers said anyway | Naturalness | Double replies per call | First word on requests, median | Answer time, median | Cost per call |
|---|---|---|---|---|---|---|---|---|---|
| cheap: Haiku 5.5 | `relay` | 48 | 94% [91, 97] (252/267) | 4% [1, 18] | 2.54 [2.31, 2.81] | 1.27 [0.88, 1.65] | 1.85 s | 9.0 s | $0.004 |
| cheap: Haiku 5.5 | `slow_speaks` | 48 | 88% [84, 91] (235/267) | 0% [0, 12] | 1.81 [1.65, 1.98] | 0.25 [0.12, 0.38] | 1.86 s | 11.5 s | $0.003 |
| cheap: Haiku 5.5 | `slow_only` (= `fast_only`) | 48 | 91% [87, 94] (243/267) | 0% [0, 12] | 3.02 [2.79, 3.25] | 0.15 [0.06, 0.23] | 1.86 s | 5.5 s | $0.002 |
| mid: Sonnet 5.5 (low) | `relay` | 48 | 99% [96, 99] (263/267) | 4% [1, 18] | 2.85 [2.52, 3.23] | 1.25 [0.92, 1.60] | 1.85 s | 8.8 s | $0.048 |
| mid: Sonnet 5.5 (low) | `slow_speaks` | 48 | 99% [97, 100] (264/267) | 59% [41, 75] | 3.52 [3.33, 3.71] | 0.04 [0.00, 0.12] | 1.88 s | 6.5 s | $0.040 |
| mid: Sonnet 5.5 (low) | `slow_only` | 48 | 99% [97, 100] (264/267) | 0% [0, 12] | 3.29 [3.10, 3.48] | 0.04 [0.00, 0.12] | 5.02 s | 5.0 s | $0.034 |
| top: Opus 5.5 (high) | `relay` | 16 | 98% [92, 99] (87/89) | 11% [2, 44] | 3.12 [3.00, 3.31] | 0.12 [0.00, 0.31] | 1.85 s | 11.6 s | $0.100 |
| top: Opus 5.5 (high) | `slow_speaks` | 16 | 92% [85, 96] (82/89) | 56% [27, 81] | 2.75 [2.50, 3.00] | 0.00 [0.00, 0.00] | 1.88 s | 11.8 s | $0.101 |
| top: Opus 5.5 (high) | `slow_only` | 16 | 99% [94, 100] (88/89) | 0% [0, 30] | 2.44 [2.19, 2.69] | 0.00 [0.00, 0.00] | 10.62 s | 11.1 s | $0.093 |
| (no slow model) | `fast_only` | 48 | 91% [87, 94] (243/267) | 0% [0, 12] | 3.02 [2.79, 3.25] | 0.15 [0.06, 0.23] | 1.86 s | 5.5 s | $0.002 |
| (no slow model) | `fast_only_think` | 48 | 96% [93, 98] (257/267) | 0% [0, 12] | 3.10 [2.94, 3.29] | 0.00 [0.00, 0.00] | 6.44 s | 7.9 s | $0.003 |
| (no slow model) | `fast_context` | 48 | 95% [92, 97] (254/267) | 0% [0, 12] | 3.75 [3.35, 4.12] | 0.10 [0.02, 0.21] | 2.06 s | 2.1 s | $0.022 |
| (no slow model) | `fast_context_think` | 48 | 100% [99, 100] (267/267) | 0% [0, 12] | 3.98 [3.73, 4.21] | 0.00 [0.00, 0.00] | 2.84 s | 2.9 s | $0.023 |

| Difference (first arm minus second) | Correct (points) | Calls with a wrong or invented fact (points) | Withdrawn answers said (points) | Naturalness | Double replies per call | First word, median (s) | Turns waiting > 2 s (points) | Answer time (s) | Cost per call ($) |
|---|---|---|---|---|---|---|---|---|---|
| relay minus slow_only | -1.1 [-3.5, +0.0] | +0.0 [+0.0, +0.0] | +11.1 [+0.0, +33.3] | +0.69 [+0.44, +1.00] | +0.12 [+0.00, +0.31] | -7.76 [-8.27, -7.34] | -93.6 [-97.4, -89.4] | +0.5 [-1.7, +3.0] | +0.007 [-0.002, +0.017] |
| slow_speaks minus slow_only | -6.7 [-11.2, -2.3] | +6.2 [+0.0, +18.8] | +55.6 [+22.2, +88.9] | +0.31 [-0.06, +0.75] | +0.00 [+0.00, +0.00] | -7.73 [-8.24, -7.31] | -95.5 [-98.0, -93.0] | +0.7 [-1.9, +2.3] | +0.007 [-0.001, +0.017] |
| slow_speaks minus relay | -5.6 [-10.8, +0.0] | +6.2 [+0.0, +18.8] | +44.4 [+11.1, +77.8] | -0.38 [-0.69, -0.06] | -0.12 [-0.31, +0.00] | +0.03 [+0.03, +0.03] | -1.9 [-4.6, +0.6] | +0.2 [-2.2, +1.4] | +0.000 [-0.011, +0.011] |
| relay minus fast_context_think | -2.2 [-5.5, +0.0] | -2.1 [-6.2, +0.0] | +11.1 [+0.0, +33.3] | -0.85 [-1.08, -0.62] | +0.12 [+0.00, +0.31] | -0.97 [-1.21, -0.82] | -93.6 [-97.4, -89.4] | +8.7 [+8.4, +11.3] | +0.077 [+0.060, +0.102] |
| relay minus fast_context | +2.6 [+0.0, +5.6] | -45.8 [-62.5, -29.2] | +11.1 [+0.0, +33.3] | -0.62 [-1.00, -0.27] | +0.02 [-0.17, +0.23] | -0.21 [-0.21, -0.21] | -93.6 [-97.4, -89.4] | +9.5 [+9.3, +12.2] | +0.078 [+0.061, +0.102] |
| relay minus fast_only | +6.7 [+1.5, +11.5] | -14.6 [-31.2, -2.1] | +11.1 [+0.0, +33.3] | +0.10 [-0.17, +0.38] | -0.02 [-0.21, +0.21] | -0.01 [-0.01, -0.01] | -3.2 [-7.5, +1.1] | +6.1 [+5.7, +8.7] | +0.098 [+0.081, +0.122] |

### Second family: the cheap tier on OpenAI (GPT-6 Luna with GPT-6 Luna, effort `none`)

With the same model on both sides, `slow_only` and `fast_only` are one arm.

| Arm | Correct answers | Calls with a wrong or invented fact | Withdrawn answers said anyway | Naturalness (judge, 1-5) | Double replies per call | First word, median (p90) | Turns waiting > 2 s | Answer time, median | Cost per call |
|---|---|---|---|---|---|---|---|---|---|
| `relay` | 89% [84, 92] (237/267) | 42% [29, 56] | 0% [0, 12] (0/27) | 3.19 [2.98, 3.40] | 0.42 [0.23, 0.60] | 1.91 s (1.9) | 6% [4, 9] | 6.0 s [5.9, 7.9] | $0.013 |
| `slow_speaks` | 88% [84, 92] (236/267) | 40% [27, 54] | 81% [63, 92] (22/27) | 3.21 [2.94, 3.46] | 0.17 [0.00, 0.38] | 1.95 s (1.9) | 1% [1, 3] | 5.9 s [4.6, 6.7] | $0.012 |
| `fast_only` (= `slow_only`) | 84% [79, 88] (224/267) | 48% [34, 62] | 0% [0, 12] (0/27) | 3.08 [2.85, 3.33] | 0.02 [0.00, 0.06] | 4.31 s (13.9) | 68% [64, 72] | 4.5 s [4.2, 4.7] | $0.007 |
| `fast_context` | 98% [96, 99] (262/267) | 17% [9, 30] | 0% [0, 12] (0/27) | 4.44 [4.23, 4.65] | 0.00 [0.00, 0.00] | 1.91 s (4.6) | 14% [11, 18] | 1.9 s [1.9, 1.9] | $0.011 |

| Difference (first arm minus second) | Correct (points) | Calls with a wrong or invented fact (points) | Withdrawn answers said (points) | Naturalness | Double replies per call | First word, median (s) | Turns waiting > 2 s (points) | Answer time (s) | Cost per call ($) |
|---|---|---|---|---|---|---|---|---|---|
| relay minus fast_only | +4.9 [+1.4, +9.0] | -6.2 [-20.8, +6.2] | +0.0 [+0.0, +0.0] | +0.10 [-0.21, +0.40] | +0.40 [+0.21, +0.60] | -2.40 [-2.67, -2.24] | -62.2 [-65.0, -59.7] | +1.5 [+1.4, +3.3] | +0.005 [+0.004, +0.008] |
| slow_speaks minus fast_only | +4.5 [+0.0, +9.3] | -8.3 [-22.9, +2.1] | +81.5 [+59.3, +100.0] | +0.12 [-0.23, +0.44] | +0.15 [-0.02, +0.38] | -2.36 [-2.63, -2.20] | -66.7 [-70.1, -63.4] | +1.3 [+0.1, +2.1] | +0.005 [+0.003, +0.009] |
| slow_speaks minus relay | -0.4 [-2.7, +1.8] | -2.1 [-10.4, +6.2] | +81.5 [+59.3, +100.0] | +0.02 [-0.35, +0.38] | -0.25 [-0.54, +0.08] | +0.04 [+0.04, +0.04] | -4.5 [-6.5, -2.4] | -0.2 [-1.7, +0.2] | -0.000 [-0.002, +0.002] |
| relay minus fast_context | -9.4 [-13.6, -4.7] | +25.0 [-2.1, +54.2] | +0.0 [+0.0, +0.0] | -1.25 [-1.54, -0.96] | +0.42 [+0.23, +0.60] | +0.00 [+0.00, +0.00] | -8.3 [-11.6, -5.4] | +4.1 [+4.0, +6.0] | +0.001 [-0.001, +0.004] |
| slow_speaks minus fast_context | -9.7 [-14.6, -4.7] | +22.9 [-4.2, +52.1] | +81.5 [+59.3, +100.0] | -1.23 [-1.56, -0.90] | +0.17 [+0.00, +0.38] | +0.04 [+0.04, +0.04] | -12.8 [-16.2, -9.7] | +3.9 [+2.7, +4.8] | +0.001 [-0.002, +0.005] |

### A slow model that is slow: D = 8 s added to every slow-model call

| Arm | Slow-model delay D | Correct answers | Withdrawn answers said anyway | Naturalness | Double replies per call | First word on requests, median | Turns with no reply before Sam goes on | Answer time, median |
|---|---|---|---|---|---|---|---|---|
| `relay` | 0 s | 99% [96, 99] (263/267) | 4% [1, 18] | 2.85 [2.52, 3.23] | 1.25 [0.92, 1.60] | 1.85 s | 3% [2, 5] | 8.8 s |
| `relay` | 8 s | 96% [93, 98] (257/267) | 19% [8, 37] | 2.58 [2.35, 2.77] | 0.44 [0.21, 0.71] | 1.85 s | 2% [1, 4] | 31.6 s |
| `slow_speaks` | 0 s | 99% [97, 100] (264/267) | 59% [41, 75] | 3.52 [3.33, 3.71] | 0.04 [0.00, 0.12] | 1.88 s | 2% [1, 4] | 6.5 s |
| `slow_speaks` | 8 s | 95% [92, 97] (254/267) | 41% [25, 59] | 2.50 [2.31, 2.71] | 0.02 [0.00, 0.06] | 1.88 s | 4% [3, 7] | 23.9 s |
| `slow_only` | 0 s | 99% [97, 100] (264/267) | 0% [0, 12] | 3.29 [3.10, 3.48] | 0.04 [0.00, 0.12] | 5.02 s | 21% [17, 25] | 5.0 s |
| `slow_only` | 8 s | 93% [89, 95] (247/267) | 0% [0, 12] | 1.46 [1.27, 1.65] | 0.00 [0.00, 0.00] | 21.89 s | 48% [43, 52] | 21.4 s |
| `slow_filler` | 0 s | 99% [97, 100] (264/267) | 0% [0, 12] | 3.00 [3.00, 3.00] | 0.00 [0.00, 0.00] | 1.20 s | 0% [0, 1] | 5.0 s |
| `slow_filler` | 8 s | 95% [91, 97] (253/267) | 0% [0, 12] | 1.92 [1.81, 2.00] | 0.00 [0.00, 0.00] | 1.20 s | 0% [0, 1] | 21.7 s |

| Difference (first arm minus second) | Correct (points) | Calls with a wrong or invented fact (points) | Withdrawn answers said (points) | Naturalness | Double replies per call | First word, median (s) | Turns waiting > 2 s (points) | Answer time (s) | Cost per call ($) |
|---|---|---|---|---|---|---|---|---|---|
| relay minus slow_only | +3.8 [-0.8, +9.3] | +6.2 [-2.1, +14.6] | +18.5 [+0.0, +37.0] | +1.12 [+0.90, +1.33] | +0.44 [+0.21, +0.71] | -19.84 [-21.32, -19.32] | -94.7 [-97.3, -91.5] | +10.2 [+8.6, +11.6] | +0.012 [+0.009, +0.015] |
| slow_speaks minus slow_only | +2.6 [+0.4, +5.5] | -2.1 [-12.5, +6.2] | +40.7 [+11.1, +70.4] | +1.04 [+0.83, +1.25] | +0.02 [+0.00, +0.06] | -19.81 [-21.29, -19.29] | -93.8 [-96.2, -91.7] | +2.6 [+0.3, +6.0] | +0.004 [+0.001, +0.007] |
| slow_speaks minus relay | -1.1 [-5.3, +1.6] | -8.3 [-14.6, -2.1] | +22.2 [-7.4, +51.8] | -0.08 [-0.29, +0.12] | -0.42 [-0.69, -0.19] | +0.03 [+0.03, +0.03] | +0.9 [-2.1, +3.7] | -7.7 [-9.8, -4.2] | -0.008 [-0.010, -0.006] |
| slow_speaks minus slow_filler | +0.4 [-0.8, +1.6] | -2.1 [-12.5, +6.2] | +40.7 [+11.1, +70.4] | +0.58 [+0.38, +0.79] | +0.02 [+0.00, +0.06] | +0.68 [+0.68, +0.68] | +6.2 [+3.9, +8.3] | +2.2 [+0.0, +5.8] | +0.003 [-0.001, +0.006] |

### The small size (about 4k tokens of records)

16 scripts x 3 seeds per arm; the judge's measures (wrong or invented facts, naturalness, double replies) are on seed 0 only (16 calls per arm), as planned.

| Arm | Correct answers | Calls with a wrong or invented fact | Withdrawn answers said anyway | Naturalness (judge, 1-5) | Double replies per call | First word, median (p90) | Turns waiting > 2 s | Answer time, median | Cost per call |
|---|---|---|---|---|---|---|---|---|---|
| `relay` | 99% [97, 100] (265/267) | 0% [0, 19] | 4% [1, 18] (1/27) | 2.62 [2.38, 2.88] | 1.44 [1.12, 1.75] | 1.85 s (1.9) | 3% [2, 5] | 8.6 s [7.5, 9.5] | $0.034 |
| `slow_speaks` | 100% [99, 100] (267/267) | 0% [0, 19] | 70% [52, 84] (19/27) | 3.56 [3.31, 3.81] | 0.06 [0.00, 0.19] | 1.88 s (1.9) | 2% [1, 3] | 6.5 s [5.3, 7.2] | $0.029 |
| `slow_only` | 100% [99, 100] (267/267) | 0% [0, 19] | 11% [4, 28] (3/27) | 3.31 [3.12, 3.56] | 0.06 [0.00, 0.19] | 4.38 s (12.6) | 57% [52, 61] | 5.1 s [4.9, 5.8] | $0.025 |
| `slow_filler` | 100% [98, 100] (266/267) | 0% [0, 19] | 11% [4, 28] (3/27) | 2.94 [2.81, 3.00] | 0.00 [0.00, 0.00] | 1.20 s (1.2) | 0% [0, 1] | 5.1 s [4.9, 5.4] | $0.026 |
| `fast_only` | 96% [93, 98] (256/267) | 19% [7, 43] | 0% [0, 12] (0/27) | 3.25 [2.88, 3.62] | 0.12 [0.00, 0.31] | 1.85 s (1.9) | 9% [7, 12] | 5.5 s [5.5, 6.3] | $0.002 |
| `fast_only_think` | 97% [94, 98] (258/267) | 0% [0, 19] | 4% [1, 18] (1/27) | 2.94 [2.69, 3.19] | 0.00 [0.00, 0.00] | 4.87 s (15.0) | 81% [77, 84] | 7.2 s [5.8, 7.9] | $0.002 |
| `fast_context` | 100% [99, 100] (267/267) | 6% [1, 28] | 0% [0, 12] (0/27) | 4.62 [4.25, 4.94] | 0.06 [0.00, 0.19] | 1.86 s (1.9) | 10% [7, 13] | 1.9 s [1.9, 1.9] | $0.002 |
| `fast_context_think` | 100% [99, 100] (267/267) | 0% [0, 19] | 0% [0, 12] (0/27) | 4.50 [4.25, 4.75] | 0.00 [0.00, 0.00] | 1.86 s (5.2) | 56% [51, 60] | 2.2 s [1.9, 2.3] | $0.002 |

### The plan's criteria, applied mechanically (`analyze.py`, `verdict`)

(a) median first word at least 0.5 s sooner and fewer turns waiting more than 2 s (interval excluding 0); (b) correct answers within 5 points and no more calls with an invented value (the programmatic check); (c) naturalness not lower (interval including 0 or above it) and double replies plus contradictions not more by over 0.5 per call. "Ahead of" lists the single-fast arms the design beats on correct answers or invented values with an interval excluding 0.

| Condition | Design | (a) answers sooner | (b) as correct, no more invented | (c) as natural, not more doubles | **Holds against `slow_only`** | Ahead of (correct or fewer invented, interval excluding 0) | Holds if the first-word median counts only answered turns |
|---|---|---|---|---|---|---|---|
| mid tier, `realistic`, D = 0 (the headline) | `relay` | yes | yes | no | **no** | `fast_only`, `fast_context` | no |
| mid tier, `realistic`, D = 0 (the headline) | `slow_speaks` | yes | yes | yes | **yes** | `fast_only`, `fast_only_think`, `fast_context` | no |
| mid tier, `small`, D = 0 | `relay` | yes | yes | no | **no** | none | no |
| mid tier, `small`, D = 0 | `slow_speaks` | yes | yes | yes | **yes** | `fast_only`, `fast_only_think` | no |
| mid tier, `realistic`, D = 8 s | `relay` | yes | yes | no | **no** | (no single-fast arm run) | no |
| mid tier, `realistic`, D = 8 s | `slow_speaks` | yes | yes | yes | **yes** | (no single-fast arm run) | yes |
| cheap tier, Anthropic (Haiku 5.5 with Haiku 5.5) | `relay` | no | yes | no | **no** | none | no |
| cheap tier, Anthropic (Haiku 5.5 with Haiku 5.5) | `slow_speaks` | no | yes | no | **no** | none | no |
| cheap tier, OpenAI (GPT-6 Luna with GPT-6 Luna) | `relay` | yes | no | yes | **no** | `fast_only` | no |
| cheap tier, OpenAI (GPT-6 Luna with GPT-6 Luna) | `slow_speaks` | yes | no | yes | **no** | none | no |
| top tier, Anthropic (Haiku 5.5 with Opus 5.5 at effort high), 16 scripts x 1 seed | `relay` | yes | yes | yes | **yes** | `fast_only` | yes |
| top tier, Anthropic (Haiku 5.5 with Opus 5.5 at effort high), 16 scripts x 1 seed | `slow_speaks` | yes | no | yes | **no** | none | no |

### The judge, checked by hand

Gemini 3.8 Flash judged every call. Before any judge output existed, 32 blinded calls of the headline condition (4 per arm) were labelled by hand (`runs/judge_validation/`). Naturalness: the judge's score was within 1 point of the hand score on 97% of calls and equal on 62% (Spearman 0.57); it was +0.22 points more generous on average, most of all to the single-fast arms. Its event labels:

| Label | Calls the hand labels flag | Calls the judge flags | Both | Agreement | Cohen's kappa |
|---|---|---|---|---|---|
| Double replies | 4 | 6 | 4 | 94% | 0.76 |
| Contradictions | 3 | 1 | 1 | 94% | 0.48 |
| Withdrawn answers said | 2 | 2 | 2 | 100% | 1.00 |
| Wrong or invented facts | 7 | 5 | 5 | 94% | 0.80 |

Contradictions are rare and the judge misses some, so they carry little weight here. The hand check covers the mid tier only; for the cheap and top tiers, the judge's labels were read beside eleven transcripts while checking them, without a formal score, and matched what the transcripts show.

### Cost

From both ledgers (`runs/ledger.jsonl`, the laptop's; `runs/ledger-vm.jsonl`, the cloud machine's; neither is a copy of the other), every role and the judge included. The era-model runs of Plan amendment 3 are counted apart: the lead set them outside the analysis and outside today's budget.

| Spend | Cost |
|---|---|
| Today's models: mid tier | $23.43 |
| Today's models: cheap tier | $3.02 |
| Today's models: top tier | $4.70 |
| Today's models: judge | $11.31 |
| Today's models: set-up (calibration, probes, sanity and smoke runs, D = 4 calls in flight when stopped) | $3.82 |
| **Today's models, in all** | **$46.27** |
| Era models of Plan amendment 3 (set aside, not analysed; outside today's budget) | $31.26 |
| Both ledgers, in all | $77.53 |

### Not run

Everything Plan amendment 4 lists was run. These were planned earlier (Plan amendments 1 and 3) and left out for the budget before any result was known, or allowed by amendment 4 only at one seed. Estimates use the measured cost per call of the nearest condition and the judge's measured cost per call.

| Not run | Estimated cost | Estimated from |
|---|---|---|
| Top tier, seeds 1 and 2 (3 arms x 16 scripts x 2) | $10.28 | from the top tier |
| Delay D = 4 s, mid tier (4 slow arms x 16 x 3) | $8.31 | from D = 8 |
| `medium` size, mid tier (8 arms x 16 x 1) | $3.80 | midway between `small` and `realistic` |
| Mid tier on OpenAI, GPT-6.1 Sol as the slow model (3 slow arms x 16 x 3) | $7.13 | Sonnet 5.5's cost per call (the same list prices); a guess |
| Sonnet 5.5 at effort high (3 slow arms x 16 x 1) | $2.38 | Sonnet 5.5 at effort low; a lower bound |
| **All of the above** | **$31.90** | |

### Caveats

1. **Simulated speech, real model calls.** Every word comes from a real call, but when it is heard comes from a fitted timing model (`data/timing.json`): first token, reading speed and output speed per model, measured through OpenRouter on 9 and 10 October 2026, plus fixed text-to-speech (0.25 s), end-of-turn (0.5 s) and speaking-rate constants. A real voice stack adds its own latency, the same for every arm.
2. **The slow model was not slow.** Through OpenRouter every model here reached its first visible token in about 1.1-1.3 s (Sonnet 5.5: a = 1.18 s; Haiku 5.5: a = 1.10 s), except Opus 5.5 (a = 3.72 s). The idea's own premise, a slow model several seconds from its first word, is present only in the D = 8 s condition and, partly, the top tier. D = 4 s was not run, so nothing is known between 0 and 8 s.
3. **One model behaviour drives `slow_only`'s latency.** The prompt asks a single model to say a brief holding line in the same response as a tool call. Haiku 5.5 did so in 99% of its tool-calling replies, Sonnet 5.5 in 6%, Opus 5.5 in 8% and GPT-6 Luna in 1%, so `slow_only` is silent while a lookup runs (58% of its turns at the headline condition waited more than 2 s). A canned filler (`slow_filler`) or a model that does say the line removes most of the split's advantage.
4. **The strongest baseline depends on the records fitting in a cached prompt.** At the realistic size they are 79.5k tokens, below the 100k price step; `fast_context_think` read them in about 0.2 s thanks to caching. Much larger records, or records that cannot be cached, would change that comparison; no size above `realistic` was run. It was also the slowest to its first word of the strong arms (p90 7.4 s), since it says nothing while it thinks.
5. **The top tier is small.** 16 scripts x 1 seed (89 still-wanted and 9 withdrawn questions per arm), so its intervals are wide. Opus 5.5's timing was fitted at effort low and used at effort high (its reasoning tokens timed at its measured reasoning rate).
6. **The `_think` arms are timed slightly slow.** Haiku 5.5 reasons at 145 tokens/s and they were timed at 128, which makes their reasoning calls about 0.2 s too slow (Plan amendment 3); not re-run.
7. **The judge.** Naturalness agreement with the hand labels is moderate (Spearman 0.57 on 32 calls) and the judge was more generous to the single-fast arms than the hand labels; its contradictions labels miss some. It was hand-checked on the mid tier only.
8. **The scripted Sam** waits 3 s for a reply before going on and never changes his words; a real caller would react to silence, fillers and repetition in his own way.
9. **Criterion (b) is a point comparison.** One invented value in 48 calls fails it, as happened to both designs in the OpenAI cheap tier; the intervals of those differences include 0.
10. **Model quirks are counted as behaviour.** Haiku 5.5 without reasoning sometimes speaks its reasoning aloud (in `fast_only`, `fast_context` and as the slow model of `slow_speaks` in the cheap tier); GPT-6 Luna at effort `none` sometimes appends stray characters, or once an invented sentence, to a filler. Neither is a harness fault.
11. **The second family covers the cheap tier only.** The mid tier on OpenAI (GPT-6.1 Sol) was not run under Plan amendment 4, so the headline comparison is one family.
12. **The small size's judged measures** are on seed 0 only (16 calls per arm), as planned.

<!-- results:end -->

## Re-running

```bash
cd concepts/fast-slow-voice
uv sync && uv run pytest                        # 35 tests, no network
zstd -dc runs.tar.zst | tar -xf -               # restore runs/ (every record, judgement and both ledgers)
uv run python analyze.py > runs/analysis.txt    # results.json and every table, from runs/
uv run python report_tables.py                  # the Results section above, from results.json
uv run python validate_judge.py compare         # the judge against the hand labels
uv run python show.py runs/anthropic-realistic-d0.jsonl --arm relay --script s1-03 --seed 0 --judge --calls   # read a call
```

Everything from scratch (about $46 of model spend; every step resumes where it
stopped, skipping calls already recorded):

```bash
../../bin/with-openrouter-key uv run python probe.py         # optional: the model settings are accepted
../../bin/with-openrouter-key uv run python calibrate.py     # step 0 of the plan: data/timing.json (the fits used are kept there)
../../bin/with-openrouter-key bash run_all.sh                # sanity, mid tier (both sizes, D = 0 and 8), cheap and top tiers, judging, analysis
```

On the cloud machine, as run here: copy the folder over with
`rsync -az --exclude .venv --exclude __pycache__ --exclude runs/ concepts/fast-slow-voice/ ubuntu@34.67.145.107:demos/concepts/fast-slow-voice/`,
then run `LEDGER=runs/ledger-vm.jsonl STEPS="0 1 2 9 10 11" setsid nohup ../../bin/with-openrouter-key bash run_all.sh > runs/main.log 2>&1 < /dev/null &`
in the folder there (with `~/.local/bin` on the `PATH`), and copy `runs/` back
with `rsync -az ubuntu@34.67.145.107:demos/concepts/fast-slow-voice/runs/ runs/`.
