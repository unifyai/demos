# Demonstration by reference

A demonstration is an ordinary request whose screen frames travel by reference:
the request carries, for every frame kept from a screen share, an id, the time
and what the person said while it was on screen, and the model looks at a frame
(a tool call that returns its image, labelled with its id) only when it needs
to. The claim (from the post "Demonstration without a mode") is that this keeps
a long session cheap, since each step's prompt grows by a line per frame rather
than an image per frame, and accurate, since the words point at the few frames
that matter.

## Plan (written 2026-10-09 22:56 UTC, before any scored run)

### The idea, and the conditions it is built for

Teaching by showing needs no special mode: the frames of a shared screen go with
an ordinary request, each as a reference (id, time, what was said), and the loop
that receives them looks at a frame only when the words say it matters. Its own
reasoning says where it should pay off: **long sessions**, hundreds of frames
over several minutes, in which the task depends on **a few key frames picked out
by what the person said** ("first up is this one", "then this"), while the rest
is ordinary activity, including talk that says "this" without picking anything.
Short sessions are the low end, where sending every frame costs little.

### Task and data

A person shares a browser with five tabs (a task tracker with list and detail
views, mail, a calendar, team chat, a planning doc) and, among other things,
picks 3 to 5 tasks for next week's queue, in order, by opening them or pointing
at their row while speaking. Their request at the end is the same in every
session: "Please set next week's queue to the tasks I picked out for next week
while I was sharing my screen, in the order I gave them. Replace whatever is in
the queue now." The agent has the 40-task list as text (IDs and titles; titles
come in families of three near-duplicates, such as "Draft Q3 on-call rota",
"Review Q3 on-call rota", "Draft Q4 on-call rota") and a `set_queue(task_ids)`
tool. **Correct** means `set_queue` was called with exactly the right IDs in the
right order. The speech never names a task, so the answer needs the pixels: the
words say which moments matter, and only the frame at that moment (or the next
one) shows which task was meant.

`make_data.py` generates every session from seeds (`src/demoref/world.py`,
`session.py`) and renders each frame with PIL (`render.py`, 1280x800, DejaVu
fonts from matplotlib, lossless WebP, about 25 KB each, deterministic; sha256 in
`data/frames_manifest.json`). Each item fixes a world and a **key script** that
is identical at every length:

- an intro ("ok so I'll show you what I want done next week, in order");
- K picks (K = 3, 4, 5 in turn across items), each shown as its detail page with
  the words on that frame, as the pointer on its row in the list with the words
  on that frame, or with the words starting a frame before the page opens
  ("and then…" on the list, "…this one" as it opens);
- before a pick, with probability 0.2, a near-duplicate opened by mistake ("hmm,
  this isn't the one I'm looking for");
- one task set aside out loud: already done (any task; even items) or for the
  week after (a near-duplicate of a pick; odd items);
- with probability 0.5, a revisit of a pick ("back to this one for a sec, see,
  it's due Oct 14");
- in a third of the items (2, 5 and 8), a late correction ("actually, scratch the
  second one, do this one instead"), which changes the answer;
- an outro; and at least one neutral frame after and two before every beat.

Ordinary activity fills the rest of the session: browsing and scrolling the
list (often with the pointer on rows), opening other tasks, searching, mail,
calendar, chat and the doc, with filler or remarks that pick nothing ("let me
just check something on this one", "I did this one first thing last week",
"then I'll show you the rest"). Frames are 0.8 to 2.2 s apart.

- **Sizes:** 64, 128, 256 and 500 frames (about 1.5, 3, 6 and 12 minutes). 500
  is close to the most one request can carry: OpenRouter refused 1,000 images
  and accepted 600 on both models (it estimates 1,600 tokens an image against the
  1M context), and 500 real frames are about 670k prompt tokens on Haiku.
- **Sanity condition:** each item's three-pick version with every pick on its
  own page, the words on the same frame, no negatives and nothing between beats
  but the pauses (20 frames). Every arm should get these right.
- **Items and seeds:** 10 items (world and script seeds 0 to 9) at each size,
  each run with 3 independent seeds per arm. Items 100 to 107 are development
  items used for the pilots below and never scored.

Amendment, 2026-10-09 22:58 UTC, before any scored run: the set-aside kind and
the correction were first drawn at random (p = 0.5 and 0.35), which gave 9 of the
10 items "already done" and only one correction. They are now assigned by item,
as above, so each kind is balanced. Nothing else changed.

### Arms

Same model, same system prompt apart from one sentence on how frames arrive, the
same task list, the same frame lines (id, time, what was said), the same request,
the same `set_queue` tool and loop (`src/demoref/arms.py`):

| arm | what the model gets |
|---|---|
| `inline_all` | every frame's image after its line; one request (the million-token contexts make this a real option) |
| `inline_low` | every frame at half width and height (a quarter of the pixels, about a quarter of the image tokens) |
| `inline_sampled` | all lines, and the images of 32 evenly spaced frames, the usual video-model budget |
| `captions` | a cheap model describes each frame as it arrives (it sees the frame and what was said); the request is text only |
| `pipeline` | a dedicated demonstration pipeline: notes on each chunk of 24 frames as it completes, a written procedure once the share ends, and the request answered from the procedure |
| `by_reference` | the idea: the lines only, and `look(frame_ids)`, which returns those frames' images, each after a label with its id |

The by-reference loop is offered `look` from its first step, and the pixels a
look returns are in the very next request (the old code's first bug: frames never
reached its forced first search step). Every look result is a tool message keyed
by its own call, and every image follows its own frame id, whatever order the
fetches finish in (the old code's second bug: answers keyed by call id were
applied in arrival order). Both are tested with the model stubbed
(`tests/test_arms.py`), as is what every other arm sends.

### Models and settings

- **Primary:** `anthropic/claude-haiku-5.5` for every call in every arm (the
  request loop and the procedure; it is also the cheap model that writes the
  captions and chunk notes).
- **Second family, headline comparison only:** `openai/gpt-6-luna` on
  `inline_all` against `by_reference` at every size, same items and seeds.
- Reasoning effort medium on request and procedure calls, none on captions and
  chunk notes; `max_tokens` 16,000; first-party provider first with fallbacks
  allowed (the provider is recorded per call); a per-run nonce opens the system
  prompt so no run can read another's prompt cache. At most 12 calls in flight.
- Why not Sonnet: in the pilots `claude-sonnet-5.5` got 7 of 8 at 50 frames in
  both arms (Haiku 6 to 7 of 10), but inlining 500 frames costs $1.37 a request
  on it, so this design would cost about $130 on Sonnet.

### Measures

- **Accuracy:** correct (exact order); also the right set in any order, and the
  share of positions right.
- **Cost per request:** the billed cost of every call the request needed,
  captions and chunk notes included (from each reply's usage; the ledger holds
  the same numbers).
- **Latency:** request latency is the summed latency of the calls made after the
  share ends, which run one after another (the loop's steps; for the pipeline,
  the procedure and the request). Captions and chunk notes run as frames arrive,
  so they are reported separately as preprocessing compute, not counted in
  request latency.
- **Tokens:** prompt tokens per step of the request loop, and per request.
- **The idea's own:** frames looked at, their share of the session, whether every
  frame showing a pick (or the correction) was looked at, and steps.
- Intervals: Wilson for proportions; paired bootstrap over items (each item's
  seeds kept together) for differences in accuracy, cost and latency
  (`demos_common.stats`).

### What counts as the idea winning

The primary comparison is `by_reference` against `inline_all` at the realistic
sizes, 256 and 500 frames pooled (60 runs per arm), with each size reported on
its own as well.

- **Holds** if `by_reference` is non-inferior in accuracy (the lower end of the
  95% interval of the paired difference, by_reference minus inline_all, above
  -10 points) **and** costs less per request **and** answers faster (request
  latency), each with its interval.
- **Pulls ahead** if, in addition, its accuracy is higher with the interval
  above 0.
- **Wins outright** at a size if it is also non-inferior (same margin) to every
  other arm and cheaper than every arm that is non-inferior to it.
- **Loses** if any arm is more accurate with the interval of the difference
  below 0, or if an arm is non-inferior in accuracy and cheaper and faster.
- At 64 frames no advantage is predicted; parity in accuracy is expected, and
  inlining may well be cheaper.
- The same comparison on `gpt-6-luna` says whether the result is one model's
  quirk.

### Amendment, 2026-10-10 00:25 UTC: model strength becomes an axis (Dan, through the lead). Superseded at 00:37 UTC, see below

Reason, as given: smarter models need thinner harnesses, so the fair test of a
design is on the models it was built for. The idea was built in November and
December 2025, for that period's models and their context windows.

- **Headline (the design's own time):** `openai/gpt-5` at reasoning effort high
  (the original code's model), and a second family from the same months,
  `anthropic/claude-opus-4.5` or `anthropic/claude-sonnet-4.5` (both take images).
  Every arm in a condition uses that condition's model. GPT-5 takes at most
  272k prompt tokens (400k context) and Opus 4.5 200k, far below today's million;
  that is part of the conditions the idea was built for, so where the inline arm
  stops fitting is recorded rather than worked around. If the realistic size does
  not fit, the inline arm runs at the largest size that does, and what the full
  size would cost is reported.
- **Today's cheap tier:** the Haiku 5.5 runs made so far (items 0 and 1 complete,
  every arm and size, three seeds; 38 runs of items 2 to 9 before the batch was
  stopped for this amendment) stay as a labelled condition, not the headline.
  They were stopped at $27.77 of total spend so that the headline fits the budget;
  the Luna second-family runs planned for this tier are dropped.
- **Today's frontier, if budget allows after the headline:** `anthropic/claude-opus-5.5`
  at reasoning effort high or `openai/gpt-6.1-sol`, on the headline comparison
  only, with fewer seeds if need be, to show whether any advantage shrinks with
  stronger models.
- **Budget:** a hard stop at $80 for the whole concept, including the $27.77
  already spent.
- For the record (from the lead): the old code at 31 December 2025 got 15 of 20
  by reference, and every failure there was answers applied in arrival order
  rather than matched to their frames.

The exact sizes, items and arms on the era models follow from what fits in
their context and in the remaining budget; they are set in the note below
before any era run, from probes that measure each model's tokens per frame,
image limit and cost per request.

### Era design, set 2026-10-10 00:35 UTC before any era run. Superseded at 00:37 UTC, see below

Probes (`runs/probes.jsonl`) and a pilot on development items
(9 runs, every one right where it fitted; now `runs/discarded/era-pilot8-records.jsonl`) measured
what fits:

- **Claude Sonnet 4.5 and Opus 4.5 refuse more than 100 images in a request**
  ("Too much media: 101 images > 100"), and Opus 4.5 also has a 200k window. So
  the inline arm fits 64 frames and not 128 on either.
- **GPT-5 accepted 80 inlined frames (73k prompt tokens) and refused 90** as over
  its context window, although a frame bills at 910 tokens and the window takes
  272k prompt tokens: the provider's check counts each large image at several
  times what it bills. At half size, 128 frames fit and 256 do not. 500 frames
  would be 455k prompt tokens at full size, beyond the window at any price.
- In the pilot, by reference cost about $0.05 a request on both models at 64
  frames, against $0.10 (GPT-5) and $0.27 (Sonnet 4.5) for inlining.

The era condition, items 0 to 7 with three seeds each (the remaining budget
does not stretch to ten items on two families):

| model | arm | sizes |
|---|---|---|
| `openai/gpt-5`, effort high | `by_reference` | 64, 128, 256, 500 |
| | `inline_all` | 64, 128, 256, 500 (above 64 the request is refused: recorded as a failed request, which costs nothing) |
| | `inline_low` | 64, 128, 256, 500 (refused above 128) |
| | `inline_sampled` | 500 (the workaround that always fits) |
| `anthropic/claude-sonnet-4.5`, effort high | `by_reference` | 64, 128, 256, 500 |
| | `inline_all` | 64, 128, 256, 500 (refused above 64) |

Sonnet 4.5 rather than Opus 4.5 for the second family: both have the same
100-image cap, and Sonnet costs $3 a million input tokens against $5.
`captions` and `pipeline` are not run on the era models: with the condition's
own model writing the captions, a 500-frame session would cost about $3 (GPT-5)
to $5 (Sonnet 4.5) to caption, more than the whole remaining budget allows for
a design with seeds. Every arm uses the condition's model and the same prompts,
harness and data as the cheap tier; reasoning effort is high on every call.

What counts as the idea winning is unchanged, applied to the era models: by
reference against inlining at 64 frames, where both fit, and at 128 to 500,
where inlining does not fit and its refused requests count as failures (the
point of the condition), with `inline_low` and `inline_sampled` as the
workarounds an engineer of the time would reach for. Expected cost: about $34,
leaving about $16 of the $80 for today's frontier on the headline comparison.

### Amendment, 2026-10-10 00:37 UTC: today's models in three tiers (Dan, through the lead; supersedes the two sections above)

Dan replaced the era amendment: use only today's models that someone would
reasonably choose today, on the cost and quality frontier, and no era models.
Model strength stays an axis, in three tiers:

- **cheap:** `anthropic/claude-haiku-5.5` and `openai/gpt-6-luna`, the full design
  (every arm, every size, three seeds), both families;
- **mid:** `anthropic/claude-sonnet-5.5` (or `openai/gpt-6.1-sol`), the headline
  comparison (`by_reference` against `inline_all`), one family, fewer seeds or
  items if need be;
- **top:** `anthropic/claude-opus-5.5` at reasoning effort high, the headline
  comparison, fewer seeds or items if need be;

so the results show how the idea's advantage changes as models get stronger.
With today's million-token contexts the inline arm can fit everything (every
tier's model took 520 images in one request; 500 frames are 600k to 680k prompt
tokens), so its accuracy, cost and latency against by reference are the point at
every tier. Every arm in a condition uses that condition's model. The hard stop
stays $80 for the whole concept.

What was run on older models is kept and left out of the analysis: 39 GPT-5 runs
of the era condition (`runs/discarded/era-gpt5-records.jsonl`, $1.68, stopped
when this amendment arrived) and the 9-run era pilot on GPT-5 and Sonnet 4.5
(`runs/discarded/era-pilot8-records.jsonl`, $0.99), with their probes.

**Design under the remaining budget.** Spent so far: $30.82, so $49.18 is left.
Measured costs: the cheap tier on Haiku costs $4.56 an item (every arm, size and
seed; inlining 500 frames is $0.34 a request and captioning a 500-frame session
$0.30); Luna is estimated at $2.80 an item from its prices and the pilots. On the
mid and top tiers, inlining 500 frames costs about $1.36 (Sonnet 5.5) and $2.71
(Opus 5.5) a request, and by reference about $0.08 and $0.40.

| tier | model, reasoning effort | arms | sizes | items × seeds | estimate |
|---|---|---|---|---|---|
| cheap | Haiku 5.5, medium | all six | 64, 128, 256, 500 | items 0 to 3 × 3 (items 0 and 1 done; the 10 runs of items 2 and 3 already made are kept) | $8.5 more |
| cheap | Luna, medium | all six | 64, 128, 256, 500 | items 0 to 3 × 3 | $11 |
| mid | Sonnet 5.5, medium | `by_reference`, `inline_all` | 64, 128, 256, 500 | items 0 to 3 × 1 | $12 |
| top | Opus 5.5, high | `by_reference`, `inline_all` | 64, 500 | items 0 to 2 × 1 | $12 |

About $44 in all, leaving about $5 under the stop. The stages run in that order,
and the ledger is checked before each: a stage that would cross $78 is cut to the
items that fit, and what the rest would cost is reported. The 28 runs of Haiku
on items 4 to 9 made before the batch was stopped stay in the records file but
out of the tables, which use items 0 to 3. With 12 runs a cell on the cheap tier
and 3 or 4 on the others, the cheap tier carries the comparisons with intervals,
and the mid and top tiers show direction and cost, not significance.

### Order of work

Unit tests (model stubbed) pass; then the sanity condition, one seed, every
arm; then a smoke run of item 0 at every size, one seed, every arm, reading whole
transcripts from every arm; then the full design on the cloud machine, copying
`runs/` back after every batch.

### Budget

About $35 for the primary design (inlining 500 frames on Haiku is $0.34 a
request, as prompts above 100k tokens cost five times as much), about $7 for
the second family and about $1.5 for the sanity and smoke runs: about $44, or
about $55 with the $11.12 spent on probes and pilots. That is over the $50 aim
and under the $80 stop; if the primary design runs over its estimate, the
second family is cut to 64 and 500 frames.

### Probes and pilots before this plan ($11.12, development items only)

`probe.py` measured the constraints (`runs/probes.jsonl`): both cheap models
accept 600 images in one request and refuse 1,000; a 1280x800 frame is 1,337
prompt tokens on Haiku (348 at 640x400) and 1,201 on Luna (313), and 1,092 on
Gemini 3.8 Flash at either size (so halving the resolution saves nothing there).
Both cheap models think by default; `{"effort": "none"}` turns it off.

Pilots on development items (`runs/pilot*-records.jsonl`, not counted) changed
the design before this plan, and the record of what and why:

1. The plumbing works end to end on the real API, including images sent after
   tool results, and every arm got every sanity session right (`pilot1`).
2. The request first said "the tasks I showed you"; every model then included
   every task it saw on screen. It now says "the tasks I picked out for next
   week" (`pilot2`).
3. Haiku by reference looked at all 200 frames of a session, so the look tool's
   sentence now says "call look with the ids of the frames you need to see"
   (`pilot2`, `pilot3`). This is part of the idea ("looks only when it needs
   to"), not tuning against a result.
4. The generator put beats back to back, so speech such as "then this one" / "no,
   sorry, not this" or "yeah" over a task opened just before "next up is…" was
   ambiguous even to a careful reader, and Sonnet as well as the cheap models
   misread it (`pilot3` to `pilot6`). Every beat now has pauses around it, the
   asides never sit next to a beat, set-aside phrases say what they are about,
   and filler never affirms. Sizes moved from 24/80/200/500 to 64/128/256/500 to
   fit the pauses.
5. With those fixes Sonnet got 7 of 8 at 50 frames in both arms, its one miss
   being the "yeah" case fixed in 4; Haiku got 6 of 8 in both arms at 64 frames
   (`pilot7`), so the task is clean and the cheap model is the limit.

## Results (written 2026-10-10 02:30 UTC)

### Verdict

**Partly.** Sending frames by reference keeps long sessions cheap on every tier:
at 256 and 500 frames a request costs 2 to 16% of what inlining every frame costs.
On the cheap models its accuracy stays flat as sessions grow while inlining's
falls, and on the mid and top tiers it answers 3 to 5 times faster at 500 frames.
But on the pre-registered primary test (256 and 500 frames pooled) it does not
clear the bar on any tier. On the cheap tier it is non-inferior to inlining and far
cheaper, but not reliably faster. On the mid tier, four items are too few for the
accuracy interval. Nowhere is it more accurate with an interval above zero. At 64
and 128 frames it **loses** on both cheap models: captions or half-size inlining
are more accurate, or half-size inlining is as accurate and cheaper and faster.
With the strongest model (Opus 5.5) both arms got every request right at 64 and 500 frames,
so only cost and latency are left to win, and by reference wins those (8% of the
cost and 13.5 s against 70 s at 500 frames).

Everything in the amended plan ran: no stage was cut.

| tier | model, reasoning | arms | sizes | items × seeds | scored runs | spend on them |
|---|---|---|---|---|---|---|
| cheap | Haiku 5.5, medium | all six | 64, 128, 256, 500 | 0–3 × 3 | 288 | $18.41 |
| cheap | GPT-6 Luna, medium | all six | 64, 128, 256, 500 | 0–3 × 3 | 288 | $7.30 |
| mid | Sonnet 5.5, medium | by_reference, inline_all | 64, 128, 256, 500 | 0–3 × 1 | 32 | $13.73 |
| top | Opus 5.5, high | by_reference, inline_all | 64, 500 | 0–2 × 1 | 12 | $10.35 |

Every number below is in `results.json` (`uv run python analyze.py --items 0-3`,
which also prints every table, with error kinds, for each model).

### Headline: by reference against inlining every frame, at every tier

Accuracy is exact order. Intervals are Wilson for proportions, and a bootstrap over
items (each item's seeds and sizes kept together) for differences and ratios. The
difference is by reference minus inline, and the cost ratio is by reference over
inline. Cost is per request, including every call the request needed. Latency is
the summed latency of the calls made after the share ends.

| tier, model | frames | by reference right (95% CI) | inline right (95% CI) | accuracy diff, points [95% CI] | cost by ref / inline, $ | cost ratio [95% CI] | latency by ref / inline, s (median) | latency diff, s [95% CI] |
|---|---|---|---|---|---|---|---|---|
| cheap, claude-haiku-5.5 | 64 | 5/12 (19–68) | 6/12 (25–75) | -8 [-25, +0] | 0.0058 / 0.0098 | 0.59 [0.28, 0.89] | 12.6 / 12.9 | -0.3 [-3.2, +2.6] |
| cheap, claude-haiku-5.5 | 128 | 5/12 (19–68) | 9/12 (47–91) | -33 [-83, +17] | 0.0048 / 0.0918 | 0.05 [0.02, 0.08] | 9.6 / 14.7 | -1.9 [-8.4, +4.5] |
| cheap, claude-haiku-5.5 | 256 | 5/12 (19–68) | 3/12 (9–53) | +17 [-17, +50] | 0.0095 / 0.1794 | 0.05 [0.03, 0.08] | 18.5 / 19.6 | -1.3 [-5.9, +6.8] |
| cheap, claude-haiku-5.5 | 500 | 5/12 (19–68) | 1/12 (1–35) | +33 [+0, +67] | 0.0069 / 0.3431 | 0.02 [0.01, 0.03] | 13.8 / 20.5 | -9.2 [-18.1, -1.3] |
| **cheap, claude-haiku-5.5** | **256+500** | 10/24 (24–61) | 4/24 (7–36) | **+25 [-4, +54]** | 0.0082 / 0.2613 | **0.03 [0.02, 0.04]** | 16.7 / 20.5 | **-5.2 [-12.0, +2.8]** |
| cheap, gpt-6-luna | 64 | 5/12 (19–68) | 3/12 (9–53) | +17 [+0, +50] | 0.0098 / 0.0105 | 0.93 [0.71, 1.18] | 29.1 / 15.4 | +17.2 [+10.3, +28.7] |
| cheap, gpt-6-luna | 128 | 5/12 (19–68) | 8/12 (39–86) | -25 [-67, +17] | 0.0138 / 0.0203 | 0.68 [0.54, 0.81] | 43.8 / 19.6 | +22.3 [+12.3, +29.7] |
| cheap, gpt-6-luna | 256 | 9/12 (47–91) | 8/12 (39–86) | +8 [-17, +33] | 0.0131 / 0.0798 | 0.16 [0.13, 0.22] | 41.6 / 32.8 | +6.9 [-1.5, +12.9] |
| cheap, gpt-6-luna | 500 | 8/12 (39–86) | 4/12 (14–61) | +33 [+0, +75] | 0.0141 / 0.1540 | 0.09 [0.07, 0.11] | 35.0 / 40.1 | -6.7 [-13.0, -0.5] |
| **cheap, gpt-6-luna** | **256+500** | 17/24 (51–85) | 12/24 (31–69) | **+21 [+0, +50]** | 0.0136 / 0.1169 | **0.12 [0.09, 0.14]** | 35.4 / 37.4 | **+0.1 [-3.0, +3.2]** |
| mid, claude-sonnet-5.5 | 64 | 1/4 (5–70) | 2/4 (15–85) | -25 [-75, +0] | 0.0692 / 0.1845 | 0.38 [0.23, 0.52] | 9.7 / 18.3 | -6.3 [-9.1, -3.0] |
| mid, claude-sonnet-5.5 | 128 | 1/4 (5–70) | 3/4 (30–95) | -50 [-100, +0] | 0.2504 / 0.5308 | 0.47 [0.06, 1.30] | 14.6 / 18.1 | +0.4 [-20.4, +21.2] |
| mid, claude-sonnet-5.5 | 256 | 3/4 (30–95) | 3/4 (30–95) | +0 [-75, +75] | 0.0827 / 0.8758 | 0.09 [0.05, 0.17] | 9.2 / 47.8 | -31.8 [-48.3, -14.1] |
| mid, claude-sonnet-5.5 | 500 | 4/4 (51–100) | 3/4 (30–95) | +25 [+0, +75] | 0.0771 / 1.3628 | 0.06 [0.04, 0.08] | 11.1 / 34.1 | -29.1 [-51.9, -11.1] |
| **mid, claude-sonnet-5.5** | **256+500** | 7/8 (53–98) | 6/8 (41–93) | **+12 [-25, +50]** | 0.0799 / 1.1193 | **0.07 [0.05, 0.10]** | 9.2 / 43.9 | **-30.5 [-48.5, -12.0]** |
| top, claude-opus-5.5 | 64 | 3/3 (44–100) | 3/3 (44–100) | +0 [+0, +0] | 0.1326 / 0.3640 | 0.36 [0.23, 0.49] | 12.2 / 17.0 | -4.6 [-6.4, -2.5] |
| top, claude-opus-5.5 | 500 | 3/3 (44–100) | 3/3 (44–100) | +0 [+0, +0] | 0.2284 / 2.7237 | 0.08 [0.08, 0.09] | 13.5 / 70.2 | -41.9 [-61.9, -7.0] |

Read across the tiers:

- **Cost.** By reference's prompt stays at 12k to 50k tokens a step whatever the
  session's length. Inlining grows by 1,200 to 1,350 tokens a frame, to 600k to
  680k at 500 frames. At 500 frames by reference costs 2% (Haiku), 9% (Luna),
  6% (Sonnet) and 8% (Opus) of inlining. At 64 frames the gap is small, and on
  Luna there is none (0.93). Both cheap models bill long prompts at a higher rate
  (Haiku five times above 100k prompt tokens, Luna twice above about 272k), which
  widens the gap on the cheap tier.
- **Accuracy.** On the cheap models inlining is at its best at 128 frames (Haiku
  9/12, Luna 8/12, and 8/12 again at 256) and falls at 500 (1/12, 4/12), where the
  transcripts show the model losing the thread and guessing, for example by due
  date. By reference does not fall with length (Haiku 5/12 at every size, Luna 5/12
  rising to 8 to 9/12), but at 128 frames it is below inlining on both (5 against 9
  and 8), and at 64 below captions or half-size inlining. On Sonnet it is behind at
  64 and 128 frames (1/4 against 2/4 and 3/4) and level or ahead at 256 and 500,
  with four items a cell. On Opus both are perfect.
- **Latency.** On the cheap tier by reference's several steps (2.2 to 3.6 on average)
  cost about as much time as one long inline request saves. Luna in particular
  thinks at length on each step, so it is slower than inlining at 64 and 128 frames.
  On Sonnet and Opus a 350k to 680k-token request takes 34 to 70 s against 9 to 14 s
  by reference.
- **Sonnet 5.5 refused some long inline requests.** In 4 of its 12 inline requests
  at 128 frames or more, a reply came back with no text and no tool call. A
  diagnostic re-run of one of them gave the reason: Anthropic's `refusal` stop
  reason, which OpenRouter reports as `content_filter` (`runs/discarded/README.md`);
  the others look the same (some thinking, then nothing). The harness's one reminder
  recovered two of them. The other two (item 0 at 256 frames, item 3 at 500) ended
  unanswered and count as wrong. In item 0's case the model's thinking had already
  named the right three tasks. Without those two runs Sonnet inlining would be 3/3
  at 256 and 500. No by-reference request was refused, on any model, and Opus
  refused no inline request.

### Cheap tier: every arm

Right answers out of 12 (95% CI), and cost per request:

**Haiku 5.5**

| arm | 64 | 128 | 256 | 500 | cost per request at 64 / 128 / 256 / 500, $ |
|---|---|---|---|---|---|
| inline_all | 6/12 (25–75) | 9/12 (47–91) | 3/12 (9–53) | 1/12 (1–35) | 0.0098 / 0.0918 / 0.1794 / 0.3431 |
| inline_low | 5/12 (19–68) | 9/12 (47–91) | 5/12 (19–68) | 5/12 (19–68) | 0.0035 / 0.0059 / 0.0111 / 0.0962 |
| inline_sampled | 0/12 (0–24) | 1/12 (1–35) | 0/12 (0–24) | 0/12 (0–24) | 0.0059 / 0.0061 / 0.0062 / 0.0064 |
| captions | 11/12 (65–99) | 9/12 (47–91) | 8/12 (39–86) | 3/12 (9–53) | 0.0274 / 0.0536 / 0.1589 / 0.3147 |
| pipeline | 4/12 (14–61) | 2/12 (5–45) | 0/12 (0–24) | 1/12 (1–35) | 0.0144 / 0.0270 / 0.0506 / 0.0951 |
| by_reference | 5/12 (19–68) | 5/12 (19–68) | 5/12 (19–68) | 5/12 (19–68) | 0.0058 / 0.0048 / 0.0095 / 0.0069 |

**GPT-6 Luna**

| arm | 64 | 128 | 256 | 500 | cost per request at 64 / 128 / 256 / 500, $ |
|---|---|---|---|---|---|
| inline_all | 3/12 (9–53) | 8/12 (39–86) | 8/12 (39–86) | 4/12 (14–61) | 0.0105 / 0.0203 / 0.0798 / 0.1540 |
| inline_low | 9/12 (47–91) | 8/12 (39–86) | 6/12 (25–75) | 6/12 (25–75) | 0.0037 / 0.0063 / 0.0118 / 0.0216 |
| inline_sampled | 5/12 (19–68) | 0/12 (0–24) | 0/12 (0–24) | 0/12 (0–24) | 0.0062 / 0.0070 / 0.0074 / 0.0080 |
| captions | 8/12 (39–86) | 9/12 (47–91) | 9/12 (47–91) | 7/12 (32–81) | 0.0082 / 0.0165 / 0.0340 / 0.0722 |
| pipeline | 4/12 (14–61) | 5/12 (19–68) | 1/12 (1–35) | 4/12 (14–61) | 0.0068 / 0.0146 / 0.0240 / 0.0444 |
| by_reference | 5/12 (19–68) | 5/12 (19–68) | 9/12 (47–91) | 8/12 (39–86) | 0.0098 / 0.0138 / 0.0131 / 0.0141 |

The cost of captions and the pipeline includes their preprocessing: one caption
call per frame (Haiku $0.18 and Luna $0.06 at 500 frames, 1,100 to 1,500 s of
model time spread over the share) and one notes call per 24 frames. That work runs
while the share goes on, so it is not in request latency.

By reference minus each arm at the realistic sizes, 256 and 500 frames pooled (24
runs an arm):

| model | against | accuracy diff, points [95% CI] | cost ratio, by ref / other [95% CI] | latency diff, s [95% CI] |
|---|---|---|---|---|
| claude-haiku-5.5 | inline_all | +25 [-4, +54] | 0.03 [0.02, 0.04] | -5.2 [-12.0, +2.8] |
| claude-haiku-5.5 | inline_low | +0 [-33, +54] | 0.15 [0.10, 0.21] | -4.3 [-12.1, +3.5] |
| claude-haiku-5.5 | inline_sampled | +42 [+17, +67] | 1.30 [0.85, 1.84] | +0.9 [-5.0, +6.8] |
| claude-haiku-5.5 | captions | -4 [-37, +25] | 0.03 [0.02, 0.05] | +6.5 [+2.4, +10.7] |
| claude-haiku-5.5 | pipeline | +38 [+17, +67] | 0.11 [0.08, 0.16] | -5.5 [-9.3, -0.9] |
| gpt-6-luna | inline_all | +21 [+0, +50] | 0.12 [0.09, 0.14] | +0.1 [-3.0, +3.2] |
| gpt-6-luna | inline_low | +21 [+0, +42] | 0.81 [0.66, 0.97] | +8.0 [-1.2, +17.2] |
| gpt-6-luna | inline_sampled | +71 [+46, +92] | 1.77 [1.38, 2.19] | -2.0 [-9.0, +5.1] |
| gpt-6-luna | captions | +4 [-8, +17] | 0.26 [0.22, 0.29] | +26.6 [+20.9, +32.6] |
| gpt-6-luna | pipeline | +50 [+17, +75] | 0.40 [0.31, 0.48] | +2.1 [-7.0, +15.6] |

Captioning every frame is by reference's strongest rival on the cheap tier. It is
as accurate at long sessions and more accurate at short ones (Haiku 11/12 at 64
frames), and it answers faster because its work is done during the share. It costs
4 (Luna) to 33 (Haiku) times as much per request at 256 to 500 frames. Sampling 32
frames almost never catches the moment a pick is shown. The pipeline lost the picks
in its procedure step: Haiku's often wrote that nothing had been picked although
the chunk notes recorded the picks, and Luna's often listed every task the notes
mentioned (up to 37). It was the weakest arm that saw every frame.

### How by reference used its looks

| model | frames | frames looked at (mean, share) | looked at every pick's frame | right when it did | steps (mean) | prompt tokens per step |
|---|---|---|---|---|---|---|
| claude-haiku-5.5 | 64 | 28.4 (44%) | 12/12 | 5/12 | 2.2 | 23,144 |
| claude-haiku-5.5 | 128 | 13.4 (10%) | 11/12 | 5/11 | 2.4 | 14,972 |
| claude-haiku-5.5 | 256 | 21.0 (8%) | 8/12 | 5/8 | 3.2 | 23,840 |
| claude-haiku-5.5 | 500 | 15.2 (3%) | 8/12 | 5/8 | 2.7 | 22,103 |
| gpt-6-luna | 64 | 49.1 (77%) | 12/12 | 5/12 | 3.3 | 29,842 |
| gpt-6-luna | 128 | 68.7 (54%) | 12/12 | 5/12 | 3.6 | 42,514 |
| gpt-6-luna | 256 | 61.4 (24%) | 12/12 | 9/12 | 3.2 | 43,532 |
| gpt-6-luna | 500 | 64.0 (13%) | 12/12 | 8/12 | 3.4 | 50,049 |
| claude-sonnet-5.5 | 64 | 13.2 (21%) | 4/4 | 1/4 | 2.5 | 12,114 |
| claude-sonnet-5.5 | 128 | 26.2 (21%) | 4/4 | 1/4 | 3.8 | 23,370 |
| claude-sonnet-5.5 | 256 | 11.8 (5%) | 4/4 | 3/4 | 2.5 | 14,872 |
| claude-sonnet-5.5 | 500 | 7.2 (1%) | 4/4 | 4/4 | 2.2 | 15,771 |
| claude-opus-5.5 | 64 | 11.0 (17%) | 3/3 | 3/3 | 2.7 | 11,520 |
| claude-opus-5.5 | 500 | 10.3 (2%) | 3/3 | 3/3 | 3.0 | 18,249 |

The words did point at the frames that mattered. Every model looked at every
frame showing a pick in most runs, and the stronger the model the fewer other
frames it opened (Opus 10 of 500). Each of the 9 Haiku runs that missed a pick's
frame ended wrong. Most failures came after looking at all the right frames: they
were readings, not frames that never arrived.

### The pre-registered rules, applied

| model | 64 | 128 | 256 | 500 | 256+500 (primary) |
|---|---|---|---|---|---|
| Haiku 5.5 | loses (captions more accurate) | loses (inline_low more accurate) | no call | holds | no call: not reliably faster |
| GPT-6 Luna | loses (inline_low as accurate, cheaper and faster) | loses (captions more accurate; inline_low as accurate, cheaper and faster) | no call | holds | no call: not reliably faster |
| Sonnet 5.5 | no call | no call | no call | holds, and wins outright (only inline was run) | no call: accuracy interval [-25, +50] |
| Opus 5.5 | holds, wins outright | | | holds, wins outright | (256 not run) |

"Holds" means non-inferior in accuracy (the lower end of the difference above -10
points), cheaper and faster, each with its interval. "No call" means none of
holds, pulls ahead or loses applies. **It pulls ahead (accuracy higher, with the
interval above 0) nowhere**: the lower ends at 500 frames are exactly 0 on Haiku,
Luna and Sonnet.

### What the transcripts show

Read from every arm on both cheap models, and from both arms on the mid and top
tiers (`show.py`):

- **Frames reached the model matched to their ids.** `check_frames.py` walks every
  by-reference transcript (123 runs, 231 look steps, 53 of them with several looks
  at once, 4,408 images). Each look is answered by a tool message keyed by its own
  call, in call order. Each image follows the label of the frame asked for (its id,
  time and words), in the order asked. Nothing was shifted or dropped. Where the
  models said what a frame showed (for example "f314: FIN-2451 hovered in the list",
  "f041: Archive old wiki pages, checklist 3 items"), it matched that frame.
  Neither of the old code's bugs is present.
- **How readings went wrong, in every arm:**
  - remarks that say "this" without picking anything ("this is the one about the
    venue", "this one's from Priya") taken as picks;
  - a pick shown only by the pointer on a list row missed (the pointer is plain to
    see: item 2's f314);
  - "and then…" said over the list, with "yeah, this" as the page opens, read as two
    things or as none;
  - Sonnet by reference reading the tracker tab's title. The tab keeps the title of
    the last task opened, so a calendar frame shows "PPL-703 Translate…", and Sonnet
    took that for a pick.
- **Inlining 500 frames on the cheap models** lost the thread. Haiku built queues
  from due dates and the planning doc, not from what was said.
- **Captions were accurate frame by frame** in the runs read. Their failures came
  in reading the whole session's descriptions at the end (234k tokens on Haiku at
  500 frames), for example missing a pick split across two frames.

### Cost, from the ledger

`uv run python -m demos_common.ledger runs/ledger.jsonl`, on the merged ledger
(the cloud machine's copy, which holds the laptop's rows from before 00:38 UTC and
everything since; the laptop ran nothing after that). Total **$71.93**:

| what | $ |
|---|---|
| scored runs (the four rows of the table at the top) | 49.79 |
| Haiku runs on items 4 to 9, made before the tiers amendment and kept out of the tables | 1.18 |
| Sonnet's refusal diagnostic (two re-runs, discarded) | 1.36 |
| calls of runs stopped mid-flight (the batch stopped for the amendment) | 0.51 |
| sanity and smoke runs, every arm | 4.90 |
| pilots on development items, and probes of today's models | 11.17 |
| **today's models** | **68.91** |
| older models (GPT-5, Claude Sonnet and Opus 4.5: era pilot, era runs, probes), out of the analysis | 3.02 |

The aim was under $50, and the stop at $80 was kept. Most of the money went on
inlining long sessions (a single 500-frame request costs $0.34 on Haiku, $1.36 on
Sonnet and $2.72 on Opus) and on captioning them.

### Bugs found and fixed

In the order they were found. Discarded runs are in `runs/discarded/`, with a
README giving each file's reason.

1. **The old code's two bugs, avoided by design and tested.** By reference is
   offered `look` from its first step, and the pixels a look returns are in the
   very next request (`test_by_reference_first_step_has_lines_and_look_and_no_pixels`).
   Three looks in one step whose fetches finish in reverse order still come back
   as tool messages keyed by their own calls, each image after its own frame id
   (`test_by_reference_parallel_looks_stay_matched_when_they_finish_out_of_order`).
   On the real runs, `check_frames.py` checks the same on every transcript (above).
2. **Ambiguous data (pilots, development items).** The request said "the tasks I
   showed you", so every model queued every task it saw. Beats ran back to back, so
   "then this one" followed by "no, sorry, not this" (said over a different task)
   read as a retraction even to a careful reader. Filler "yeah" over a task opened
   just before "next up is…" read as a pick. Fixed before the plan: the request says
   "the tasks I picked out for next week", every beat has pauses around it,
   set-aside phrases say what they are about, and filler never affirms.
3. **Unbalanced items.** Drawing the set-aside kind and the correction at random
   gave 9 of 10 items "already done" and only one correction. They are assigned by
   item (amendment of 22:58 UTC, before any scored run).
4. **Harness: repeated reminders (smoke run).** A model that declined to answer was
   reminded up to the step limit, each reminder re-sending the whole request:
   inlining 500 frames cost $1.36 for four refusals. Now there is one reminder, then
   the request ends unanswered. The prompt's "When you know the answer" invited the
   refusals; it now asks for the best reading. Sanity and smoke runs from before the
   fix are in `runs/discarded/`.
5. **Data: the pointer crossed into the next row (smoke run).** The arrow's tip sat
   low in a row, so its body covered the row below, and several arms named the task
   under the arrow's body. The pointer now stays inside its row.
6. **Harness: work in the event loop (first cloud launch).** Resizing 500 frames
   for `inline_low` and encoding inline payloads held up every other run's calls,
   whose latency is timed around the request, so the stall would have counted as
   model latency. Half-size frames are now made by `make_data.py`, and payloads are
   built in a worker thread. The launch was stopped before any call returned.
7. **Platform: renders differ across machines.** The same Pillow, FreeType and
   libwebp on Linux give about 1.3% different pixels (glyph edges). The runs on the
   cloud machine use the laptop's frames, checked against the manifest
   (`make_data.py --verify`).
8. **Analysis: requests over a model's limits.** A provider refusing a request as
   over the context window or image cap is recorded as a failed request (`limit`):
   it counts as wrong and is kept out of cost and latency, rather than retried as a
   harness error (`test_a_request_over_the_models_limits_is_a_failed_request_not_an_error`).
   This mattered only for the older models.
9. **Analysis: "wins outright" did not require holding first** (found while writing
   up). The plan says by reference wins outright if it *also* is non-inferior to
   every arm and cheaper than its rivals. The code granted it when no arm was
   non-inferior to by reference, even where by reference did not hold against
   inlining (Luna, 256+500). Fixed, with
   `test_winning_outright_needs_holding_first`. No run was affected.
10. **Not a bug: Sonnet's empty replies.** They were first taken for a transport
    fault: the harness briefly made them an error, and the affected run was re-run
    twice. The re-runs showed they were refusals, so the change was reverted. The run that was scored
    is the original, made under the pre-registered harness (one reminder, then
    unanswered), and `test_an_empty_reply_is_the_models_reply_not_a_fault` pins that
    behaviour.

### Caveats and threats to validity

- **Small samples, especially above the cheap tier.** Four items with three seeds on
  the cheap tier (12 runs a cell, 24 pooled), one seed on four items on the mid
  tier, and three items on the top tier. Bootstrap intervals over four items are
  coarse, and over three identical items they collapse to a point. The mid and top
  tiers show direction, cost and latency, not significance. The budget set these
  sizes: most of it went on inlining long sessions and on captions, and two changes
  of plan midway (era models, then today's tiers) used some of it.
- **The cheap tier's ceiling is low.** No arm gets more than 11 of 12 in any cell.
  Many cells sit between 3 and 9 of 12, so differences of 2 or 3 runs are noise.
- **Synthetic sessions.** These are rendered screenshots of a fictional app, with
  scripted speech drawn from fixed phrase pools. Remarks repeat in long sessions and
  do not always match the screen ("this one's from Priya" over an email from someone
  else); a real narration would vary more. The sidebar's "Next week" view, never
  opened in any session, drew some runs into looking for it. The tracker tab keeps
  its last page's title, which is realistic, but it lured Sonnet.
- **One kind of demonstration.** Picking and ordering a few tasks by pointing and
  speaking. A demonstration of a procedure with many steps, or one whose key moments
  are not marked by words, could favour other arms.
- **Frames are given.** Which frames a capture policy keeps is not modelled; every
  arm gets the same frames at a constant rate.
- **Latency** counts the calls made after the share ends. Captions and chunk notes
  are assumed to be made as frames arrive; if they were made at the end, those arms'
  latency would grow by their preprocessing time (reported). Calls went through
  OpenRouter's standard tier from a Google Cloud machine in us-central1, at most 12
  at a time, and include upload time (a 500-frame inline request is about 17 MB).
  Other concepts' runs shared the key and the machine.
- **Prices** are OpenRouter's billed costs on 9 and 10 October 2026, including the
  cheap models' higher rates for long prompts, which weigh on inlining. Prompt
  caching was not used. It would cut the cost of by reference's later steps, which
  re-send the frame list and earlier images, more than inlining's single call.
- **Sonnet's refusals** are counted as the model's outcome (two inline runs wrong).
  A harness that retried refused requests more than once might recover them.
- **The baselines are simple versions.** Sampling is uniform (32 frames); a sampler
  that kept the frames where something was said would do better. The pipeline's
  notes see 24 frames at a time and lose which row was pointed at; a pipeline with
  per-frame notes would be the captions arm.
- **Excluded runs:** 28 Haiku runs on items 4 to 9 (made before the tiers amendment
  cut the design to items 0 to 3) stay in `runs/records.jsonl` but out of the tables
  (`--items 0-3`).
- **Not tested:** the post's other two properties (work that stays open for
  corrections, and storage that ordinary work writes to), and a capture policy.

### Files

- `src/demoref/world.py` (the fictional workplace), `render.py` (PIL screenshots),
  `session.py` (sessions, speech and ground truth), `arms.py` (the six arms and the
  one loop they share).
- `make_data.py` (sessions and frames from seeds; `--verify` checks the frames
  against `data/frames_manifest.json`), `run.py` (runs arms over sessions, one
  record per run), `analyze.py` (tables, intervals, verdict; writes
  `results.json`), `show.py` (prints one run's transcript), `check_frames.py`
  (checks that every by-reference image followed its own frame's label),
  `probe.py` (model limits and tokens per frame).
- `tests/` (71 tests, model stubbed): the data, every arm's request, the old code's
  two bugs, refused and empty replies, and the analysis.
- `data/sessions/` and `data/frames_manifest.json` are committed. `data/frames/`
  (248 MB) and `data/frames_half/` (391 MB) are regenerated from the seeds by
  `make_data.py` and are not committed.
- `runs/` (83 MB loose, out of git; committed as `runs.tar.zst`): `records.jsonl`
  (every scored run with its transcript, pixels omitted), `ledger.jsonl` (every
  model call and its cost), `tiers.sh` and `tiers.log` (the staged cloud run),
  `main-batch*.log`, `probes.jsonl`, `sanity-records.jsonl`, `smoke-records.jsonl`,
  `pilot*-records.jsonl` (development items), and `discarded/` (each file's reason
  in its README).
- `results.json`: every number in the tables.

### Re-running

```bash
cd ~/demos/concepts/demonstration-by-reference
zstd -dc runs.tar.zst | tar -xf -        # the recorded runs, to re-analyse without spending
uv sync && uv run pytest -q
uv run python make_data.py --items 0-9 --sizes 64 128 256 500   # sessions and frames from seeds
uv run python make_data.py --verify      # frames match the manifest byte for byte (macOS arm64 renders)
uv run python analyze.py --items 0-3     # results.json and every table
uv run python check_frames.py runs/records.jsonl
uv run python -m demos_common.ledger runs/ledger.jsonl

# the runs themselves (a run already in runs/records.jsonl is skipped)
K=../../bin/with-openrouter-key
$K uv run python run.py --sessions sanity-0{0..9} --seeds 0 --records runs/sanity-records.jsonl --tag demoref-sanity
$K uv run python run.py --items 0 --seeds 0 --records runs/smoke-records.jsonl --tag demoref-smoke
$K uv run python run.py --items 0-3 --seeds 0 1 2 --model anthropic/claude-haiku-5.5 --tag demoref-main
$K uv run python run.py --items 0-3 --seeds 0 1 2 --model openai/gpt-6-luna --tag demoref-main
$K uv run python run.py --items 0-3 --seeds 0 --arms by_reference inline_all \
   --model anthropic/claude-sonnet-5.5 --effort medium --tag demoref-mid
$K uv run python run.py --items 0-2 --seeds 0 --arms by_reference inline_all --sizes 64 500 \
   --model anthropic/claude-opus-5.5 --effort high --tag demoref-top
```

On the cloud machine the four stages ran in that order under `runs/tiers.sh` (one
model at a time, at most 12 calls in flight, each stage started only if the concept
stayed under $78), after the laptop's frames were copied over and checked with
`--verify`.
