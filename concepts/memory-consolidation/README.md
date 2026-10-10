# memory-consolidation

Does offline, hierarchical consolidation of an agent's activity let it answer
questions at every time scale within a fixed memory budget, without slowing
the conversation? Tested here against what a competent engineer would build
instead, on synthetic half-year histories whose answers are known exactly.

The post the idea comes from is "Offline hierarchical memory consolidation"
(`landing-page/content/blog/offline-hierarchical-memory-consolidation.mdx`).
The first experiment behind it ran the original code
(`landing-page/experiments/offline-hierarchical-memory-consolidation/`). That
code had two faults: the day level read only the action that triggered it, and
the levels above reread copies of one summary. It also ran at ten weeks, where
the whole raw record (7.7k tokens) still fitted in context and beat every
summary. This folder rebuilds the idea from scratch, with each window read
correctly and tested, at the scale the idea is meant for.

## Plan (written 9 October 2026, 22:40 UTC, before any experiment run)

### The idea

An agent logs every interaction (the request, its tool calls and results, and
its reply). Away from the conversation, a background process condenses the log
into two hierarchies of summaries. The time hierarchy has a day, a week (from
seven days), four weeks (from four weeks), twelve weeks (from three four-week
summaries) and 48 weeks (from four twelve-week summaries). The count
hierarchy has blocks of 10 interactions (from ten raw records), 40 (from four
of those), 120 (from three) and 480 (from four). Each summary reads a fixed
number of inputs, exactly the ones that tile its window, and is written as
soon as they exist. On every turn the agent reads a ready-made snapshot of the
latest summaries at every level, at the cost of a lookup. So questions at
every scale are meant to be answerable within a fixed memory budget, and the
conversation never waits on consolidation.

Two choices are fixed here where the post is loose. The post's 52 weeks from
four twelve-week summaries is 48 weeks, and 520 interactions is not a whole
number of 120s, so the top levels are 48 weeks and 480 interactions. The
post's snapshot is "the latest" summary of each level, which leaves a gap:
with whole-week blocks, the days since the last full week sit only in the
latest day summary. Here the snapshot shows, at each level, as many of the
latest summaries as make one summary of the level above. That is the last 7
days, the last 4 weeks, the last 3 four-week blocks and the last 4 twelve-week
blocks; and the last 10 raw interactions, the last 4 blocks of 10, the last 3
of 40, the last 4 of 120, and every block of 480. So each scale has a complete
view of its recent past, newest at the finest resolution. The code labels
every summary with its window, its number of interactions and their range;
the old code's summaries carried no time, which its README found left the
model unable to scope them.

### Conditions it is built for (from its own reasoning)

1. **A record far larger than the budget.** About half a year of an
   assistant's work for one person. At 26 weeks the raw record is 0.80M to
   1.44M characters (1,245 to 2,144 interactions; about 275k to 500k tokens
   for the primary model), 25 to 45 times the budget. It still fits a 1M-token
   context, so the long-context arm is a real competitor, not a strawman.
2. **Questions at every time and count scale**, plus details and gaps.
3. **Uneven activity.** Weekends at 15% of weekday activity; a busy, normal or
   quiet day chain (2x, 1x, 0.3x); weeks from 3 to 239 interactions; three
   silent breaks of 5 to 16 days; projects that pause for two to five weeks
   and resume.
4. **A conversation whose latency matters**: what each design puts on the
   path from question to reply is measured.

### Memory budget

**32,000 characters** of memory in the answer prompt for every fixed-budget
arm, about 8k tokens (11k for Haiku 5.5's tokenizer). It is a memory block
that rides on every turn, so it has to be cheap and quick to read each time:
at Haiku's price about $0.001 per turn, and it adds well under a second of
prefill. It is set in characters so it is the same for both model families,
whose tokenizers differ by half. It holds the hierarchy's whole snapshot at
the word limits below (about 25k to 31k characters at 26 weeks) without
truncation, so the budget does not cut the idea short. Every fixed-budget arm
gets the same 32,000 characters.

### Arms

Every arm uses the same model for every call (consolidation, extraction,
routing and answering), the same answer prompt (`src/memcon/answer.py`), and
the same data; only the memory block differs. Native reasoning is off for
every call; the answer may carry a brief "working" note before its answer.

| arm | write path (offline unless noted) | what the answer sees | on the path to the reply |
|---|---|---|---|
| `hierarchy` (the idea) | both hierarchies, word limits day 45, week 75, 4 weeks 100, 12 weeks 130, 48 weeks 160; blocks of 10: 45, 40: 75, 120: 100, 480: 130 | the snapshot above, within 32k | a lookup |
| `hierarchy_time`, `hierarchy_count` (ablations) | the same consolidation | one hierarchy's levels, plus the last 10 raw interactions | a lookup |
| `rolling` | one running summary, rewritten every 10 interactions from itself and those 10 raw records, at most 3,000 words (recursive summarisation as in MemoryBank's global summary and Wang et al.) | the latest summary plus the raw records since, within 32k | a lookup |
| `recent` | none | the newest raw records that fit 32k | a lookup |
| `retrieval` | each raw record embedded (`voyageai/voyage-4`) | hybrid search, BM25 and embeddings fused by reciprocal rank, over the raw records; the top records that fit 32k, in time order | one embedding call and a search |
| `mem0` | Mem0's pipeline per exchange, as it happens: an extraction call on the new interaction (with the two before it as context), then an update call that adds, updates or deletes memories against the five most related stored ones per new fact | hybrid search over the stored facts, the top ones that fit 32k, in date order | one embedding call and a search (the per-exchange write is measured separately; Mem0 runs it inside the exchange) |
| `query_time` | none | a router call reads the period or count the question asks about; the records in it are selected; if they exceed 32k they are summarised for the question in parallel chunks of 200k characters, and merged if the notes still overflow | route, map and merge calls |
| `full` (reference) | none | every raw record so far, in one prompt-cached long context | nothing extra, but a prompt of up to 500k tokens |
| `none` (floor) | none | nothing | nothing |

### Data

`src/memcon/world.py`, seeded and deterministic. Maya Chen runs a small design
studio; her assistant handles email, calendar, tasks, documents, quotes,
purchases and travel. Each history has 18 to 21 of 26 projects (the rest stand
as distractors in the answer options), each with three named people, and
about 20% one-off personal requests. Records are rendered as an agent logs
them (request, tool calls with results, reply; about 650 characters each), and
every project interaction names its project in the record. Quotes get
revised and meetings moved, so details change over time.

Sizes come from checkpoints within each half-year history: about 2, 8, 16 and
26 weeks (a random weekday from Monday to Thursday, 10:00 to 18:45), plus one
after the longest break, just after the third interaction of the first active
day after it. Generated with `run.py generate` (seeds 0 to 4); the
numbers are in the table below.

| seed | interactions at 2 / 8 / 16 / 26 weeks / after the break | characters at 26 weeks |
|---|---|---|
| 0 | 114 / 432 / 830 / 1,245 / 870 | 797,492 |
| 1 | 229 / 787 / 1,546 / 2,144 / 1,602 | 1,444,925 |
| 2 | 138 / 650 / 1,236 / 1,584 / 1,239 | 1,015,504 |
| 3 | 329 / 860 / 1,249 / 1,950 / 1,152 | 1,336,065 |
| 4 | 157 / 433 / 831 / 1,268 / 852 | 805,793 |

### Questions and scoring

About 20 questions per checkpoint (`src/memcon/questions.py`), every answer
computed exactly from the structured facts of the interactions before the
checkpoint, and scored without a judge:

- **T, time scales**: which projects were worked on today, in the past 7,
  28 and 84 days, ever, and in one past calendar week at least a week old.
  Scored by set F1 against the project list given in the prompt.
- **C, count scales**: which projects the last 10, 40, 120 and 480
  interactions involved (where that many exist). Set F1.
- **D, details**: the current quote for an item, the current date of a
  meeting, who a document was shared with, or the price of a purchase. Two
  per age bucket: up to 7 days old, 8 to 28 days, and older. Exact match.
- **N, counts**: how many new meetings, completed tasks, new quotes or shared
  documents there were in the past 7 and in the past 28 days. Exact.
- **G, gaps**: the date a project dormant for more than a week was last
  touched; the last project worked on before the most recent stretch of five
  or more silent days; and that stretch's length in days. Exact.

### Measures

- **Accuracy**: the mean score by family, by question type, by checkpoint,
  and the macro-average over the five families. Intervals come from
  `demos_common.stats`: a paired bootstrap over questions for differences
  between arms, and per-seed means as a robustness check.
- **Cost per question**: every call on the read path and the answer call, from
  the ledger (cache reads included as charged), plus the uncached list-price
  cost of the long context.
- **Latency on the conversation path**: the read path's model-call critical
  path (map calls counted in parallel) and local compute, plus the answer
  call's time to first token, give the time to the reply's first token. The
  long context is reported cold (the first question at each checkpoint, which
  writes the cache) and warm.
- **The idea's own**: consolidation cost and calls per history and per
  interaction for `hierarchy`, `rolling` and `mem0`; the input size of every
  summary; the hierarchy's cascade time per trigger, which is what a turn
  would wait if it consolidated inline; and Mem0's write latency per exchange.
- **Live loop** (`run.py live`): four busy days of seed 0, replayed as back-to-back
  turns that read the snapshot and stream a reply. Turn latency is compared
  with consolidation off, running in the background as the design does, run
  inline before the next turn, and with Mem0's per-exchange write inline.

### Models, items and seeds

- Primary: `anthropic/claude-haiku-5.5` for every call in every arm, at all
  five checkpoints of seeds 0 to 4: about 500 questions per arm.
- Second family on the headline comparison: `openai/gpt-6-luna`, every call
  in every arm, consolidation included, at the 26-week checkpoint of seeds 0
  to 2 (and 3 to 4 if the budget allows).
- Embeddings: `voyageai/voyage-4`, the same for both families. OpenRouter
  serves embedding models, and `src/memcon/client.py` adds an `embed` method
  to `demos_common.LLM`, with the same pool, key handling, retries and ledger.
  It is flagged for the lead to fold into `common/`.

### What counts as the idea winning

Primary comparison: the 26-week checkpoint, Haiku 5.5, macro-average over the
five families, paired bootstrap over questions.

- **H1, best fixed-budget memory.** `hierarchy` beats each of `recent`,
  `rolling`, `retrieval` and `mem0` overall, each difference's 95% interval
  above zero.
- **H2, its home ground.** On time and count questions (T and C together),
  `hierarchy` beats each of those four, with intervals above zero.
- **H3, close to the unbounded references at a fraction of the cost.** On T
  and C, `hierarchy` is within 0.10 of `query_time` and of `full` (the lower
  end of each difference's interval above -0.10). Its median time to the
  reply's first token must be lower, and its cost per question at least five
  times lower, than either.
- **H4, no slowing.** No model call on the path, by construction. In the live
  loop, median turn latency with background consolidation is within 10% of no
  consolidation, while inline consolidation adds measurable delay.

**The idea holds** if H1 to H4 hold at 26 weeks on Haiku and H1 and H2 hold
in direction on luna. **It holds in part** if H2 and H4 hold but H1 or H3
does not; the write-up then says where it falls short. **It does not hold**
if H2 fails, that is if a simple fixed-budget design does as well on the
questions the hierarchy exists for.

Also reported, with no threshold set: the gap between `hierarchy` and the
best fixed-budget baseline at 2, 8, 16 and 26 weeks (the idea predicts it
grows with history length); time against count views after the longest
break (the post's "time alone is not enough"); detail accuracy by age
(the idea predicts recent detail kept and old detail lost); and G questions
on their own.

### Procedure

1. Unit tests for the mechanism and every baseline, with model calls stubbed
   (`tests/`). The hierarchy tests reintroduce the old code's two faults to
   check that they are caught.
2. Sanity condition (`run.py sanity`): the first four days of seed 113, 38
   interactions and 26k characters, so every arm can hold everything. It
   was chosen by rule: of seeds 100 to 139 cut to three or four days, those
   under 30k characters with at least 35 interactions and three details in
   their last ten, the one with the most sanity questions. The
   questions are projects today, this week and ever, the last 10 and 40
   interactions, two counts, and details from the last ten interactions.
   Every memory arm should score at least 0.8; any arm that does not is
   investigated as a bug.
3. Smoke run: seed 0 at 2 and 8 weeks, every arm, then whole transcripts
   read from every arm, and the cache hits checked. Bugs are fixed and the
   affected runs repeated; discarded runs go to `runs/discarded/` with the
   reason.
4. Full runs on the cloud machine at concurrency 12 or less, with `runs/`
   copied back after every batch.
5. `analyse.py` writes `results.json` and the tables below.

### Budget

Estimated from the data sizes and list prices: Haiku consolidation about $7
(mostly `rolling` and `mem0`), long context about $9 with prompt caching (it
would be about $60 without), query-time summarisation about $6, the
fixed-budget arms about $4, and luna on the headline about $5, for about $31
in total plus the sanity, smoke and live runs. If the smoke run's measured
costs project the total above $50, the primary run drops to four seeds,
decided before it starts.

### Amendment 1 (10 October 2026, 00:30 UTC; replaced by Amendment 2): model strength becomes an axis

From Dan, through the lead, before any main answer was made. Smarter models
need thinner harnesses, so the fair test of a design is on the models it was
built for. Model strength becomes an axis, and the headline moves to the
models of the design's own time, July 2025.

- **Headline: the era of the design.** `openai/o4-mini` makes every call in
  every arm, consolidation and answering, as it did in the original code.
  The second family is `google/gemini-2.5-flash`: it costs a tenth of
  `anthropic/claude-sonnet-4` and holds the whole record in a 1M context.
  o4-mini always reasons; its effort is set to low for every call. In a
  probe at medium effort it spent its whole 8,000-token cap reasoning on one
  day summary and returned nothing, and the budget could not carry medium.
  Gemini's thinking is off, as for today's cheap models. The embedding model
  stays `voyageai/voyage-4` in every condition, so the language model is the
  only thing that changes between tiers.
- **Size.** o4-mini's context is 200k tokens, about 750k characters of these
  records (3.87 characters per token, measured). The largest checkpoint
  whose whole record fits for seeds 0 to 2 is 8 weeks: 432, 787 and 650
  interactions, about 66k, 137k and 108k tokens, 8 to 17 times the memory
  budget. At 16 weeks seeds 1 and 2 need 267k and 212k tokens. So the
  headline runs at the 2- and 8-week checkpoints of seeds 0 to 2,
  consolidating only up to 8 weeks. At 26 weeks the whole record (207k to
  374k tokens) does not fit o4-mini, which is the condition the design was
  built for. What the full size would cost on the era models is projected
  from the measured per-call costs and reported.
- **Today's cheap tier.** The runs so far on `anthropic/claude-haiku-5.5`
  stay as a labelled condition, not the headline. It covers seeds 0 to 4,
  answered at 26 weeks (the realistic size) and at 8 weeks (the headline's
  size, so the two tiers meet on the same questions), and at the other
  checkpoints if budget remains. The live loop (on Haiku) measures the
  mechanism, not the model, and stands for every tier. The luna runs
  (today's cheap tier, second family) were stopped at seed 0 and are
  discarded: one seed cannot replicate, and the budget goes to the headline.
- **Today's frontier.** If budget remains after the headline and the cheap
  tier: `anthropic/claude-opus-5.5` at high reasoning effort, or
  `openai/gpt-6.1-sol`, on the headline comparison only, with fewer seeds if
  need be. With a million-token context the whole-record baseline is
  stronger today, and its cost and latency are reported beside its accuracy.
- **Budget.** A hard stop at $80 for the whole concept, including the $9.40
  spent before this amendment. Projected: the headline about $46 (o4-mini
  about $35, of which its Mem0 chains are about $10, and Gemini about $11),
  the rest of the cheap tier's consolidation about $3.50, and its 26- and
  8-week answers about $9.50.
- **What counts as winning** is unchanged and is judged on the headline: H1
  to H3 at the 8-week checkpoint on o4-mini, with H1 and H2 replicated in
  direction on Gemini 2.5 Flash, and H4 from the live loop. The cheap tier's
  26-week results are reported as their own condition, with the same tests.
- **Code changes for it.** Each model gets its reasoning setting
  (`run.py: REASONING`), and a reasoning model's token caps get 16,000
  tokens added so reasoning can never crowd out the output (unused tokens
  are not billed). `consolidate --upto 8w` now stops the hierarchy at the
  checkpoint too, as it already stopped the rolling summary and Mem0.

### Amendment 2 (10 October 2026, 01:00 UTC): today's models in three tiers (replaces Amendment 1)

From Dan, through the lead, replacing Amendment 1 before any era run at
scale. Use only today's models that someone would reasonably choose today,
on the cost and quality frontier, and no era models. Model strength stays an
axis, in three tiers, so the results show how the idea's advantage changes as
models get stronger, with the whole-record long-context reference kept in
view at each tier (its cost and latency beside its accuracy). Every arm in a
condition uses the same model for every call.

- **Cheap tier, the full design on both families.**
  `anthropic/claude-haiku-5.5` and `openai/gpt-6-luna`, with every arm, all
  five checkpoints and seeds 0 to 4, native reasoning off as before. This is
  the design of the original plan, now run on both families. Luna's seed 0,
  stopped under Amendment 1, resumes from its saved state.
- **Mid tier, one family: `anthropic/claude-sonnet-5.5`.** It is the same
  family as Haiku and Opus, so the three tiers form one ladder. Its endpoint
  does not allow reasoning to be turned off, so it runs at low effort (a
  probe reported no reasoning tokens at low or medium on these prompts).
- **Top tier, one family: `anthropic/claude-opus-5.5` at high reasoning
  effort.**
- **What the mid and top tiers run, by cost.** Probed per-call costs and
  the measured record sizes give these projections. For one seed at 26
  weeks on Sonnet: the hierarchy's consolidation about $2.40; the rolling
  summary's about $15 and Mem0's about $13 (both write paths rewrite or
  re-read text with every interaction); answers about $0.03 per question per
  fixed-budget arm; the long context about $1.60 per checkpoint with caching;
  and query-time summarisation about $0.70 for each question about the whole
  record. On Opus at high effort every figure is two to five times higher.
  So:
  - Mid: seed 0 (the smallest 26-week history) at the 26-week checkpoint.
    The arms are the hierarchy (and its two views), recent, retrieval,
    full, none, and query-time on the time and count questions (T and C),
    where it is the strongest rival. About $9.
  - Top: seed 0 at the 8-week checkpoint, consolidating only up to it. The
    arms are hierarchy, recent, retrieval, full, and query-time on T and C.
    About $11.50. If at least $17 remains below $78 once the cheap and mid
    tiers are done, the top tier runs instead at seed 0's 26-week checkpoint
    (about $16), with the arms hierarchy, recent and full on the T and C
    questions, so that all four models meet on one checkpoint at the
    realistic size.
  - Rolling and Mem0 do not fit on either tier. What they, the other seeds
    and the 26-week size on the top tier would cost is projected from the
    measured per-call costs and reported.
- **The era runs** (o4-mini and Gemini 2.5 Flash: two cost probes and two
  sanity runs, $1.15) are kept in `runs/excluded-era-models/` and left out of
  the analysis.
- **Budget.** A hard stop at $80 for the whole concept, including $12.30
  spent by now (era runs and tier probes included). The order of spending
  is: (1) the cheap tier's full design (Haiku's remaining consolidation about
  $1, Haiku's answers about $24, luna's consolidation about $6 and answers
  about $12); (2) the mid tier; (3) the top tier. The projected total is
  about $75. Before each step the ledger is checked, and a step that would
  pass $78 is cut to fit, by fewer questions or arms, and reported as cut.
- **What counts as winning** is unchanged and is judged on the cheap tier:
  H1 to H3 at 26 weeks on Haiku, with H1 and H2 replicated in direction on
  luna, and H4 from the live loop. The mid and top tiers are read as a
  trend: the hierarchy's margin over the long context, and over the best
  fixed-budget arm that runs there, against the cheap tier's margin on the
  same seed, checkpoint and questions.
- **Code changes for it.** Each model gets its reasoning setting
  (`run.py: REASONING`), and reasoning models' token caps get 16,000 tokens
  added (unused tokens are not billed). `consolidate --upto <checkpoint>`
  now stops the hierarchy at the checkpoint too.

### Amendment 2, addendum (10 October 2026, 01:45 UTC): Sonnet and Opus refuse most questions

Running the mid tier showed that `anthropic/claude-sonnet-5.5`'s usage-policy
filter refuses most of this task's answer prompts. Of 157 answers at seed
0's 26-week checkpoint, 82 came back empty (finish reason `content_filter`,
a `refusal` field, no content, no usage), and 28 more were cut off before
they parsed. Every detail question was refused in all seven arms, most gap
and count questions were, and no question about which projects were worked
on in a period was. The refusal ("This request was blocked as it seems to
violate Anthropic's usage policy") came from every provider tried (Anthropic,
Google Vertex, Azure, Amazon Bedrock, Claude Platform on AWS), even with no
memory in the prompt. `anthropic/claude-opus-5.5` refused the same prompts in
7 of 8 tries, while Haiku 5.5 and `openai/gpt-6.1-sol` answered them. The
consolidation prompts were never refused (Sonnet wrote all 348 summaries).
The shared client's streaming also recorded the empty refusals as successful
calls with no tokens; the analysis treats any reply with no text as missing
and reports how many there were. So:

- **Mid tier: `openai/gpt-6.1-sol`** (the amendment's other mid-tier model),
  at low reasoning effort, since its endpoint cannot turn reasoning off.
  The scope and arms are the same as planned for Sonnet. Sonnet's runs
  (about $8) are kept in `runs/excluded-sonnet-content-filter/` and left out
  of the analysis.
- **Top tier: Opus 5.5 at high effort, on the questions its filter lets
  through**: the time-scale questions (T), and the count-scale ones (C) if
  budget allows, with refusals counted and left out of comparisons. It runs
  on seed 0 at the 8-week checkpoint, with the arms hierarchy, recent,
  retrieval and full. Query-time is left out at this tier (about $0.40 per
  whole-record question on Opus). About $6.50.
- With about $8 spent on Sonnet, the budget's order of cuts is unchanged:
  the top tier is cut first.

### Faults found before the main runs, and changes to the plan

These were all found in the sanity and smoke runs, before any main run. The
runs they affected are in `runs/discarded/`, each with its reason.

1. **Summaries cut off mid-sentence** (first sanity run). The hierarchy's
   summaries were capped at max(200, 4 x the word limit) tokens. Haiku writes
   about 3.2 tokens per word here and runs 20 to 50% over a word limit, so 3
   of 7 summaries were cut. Every cap is now generous (summaries get 10 x the
   word limit + 200 tokens), so only the prompt's word limit bounds a
   summary. The word limits are unchanged. Measured on seed 0's full half
   year, the snapshot then held 16k characters at 2 weeks and 30k at 26 weeks,
   inside the budget with nothing dropped.
2. **Ambiguous count questions** (first sanity run). "New quotes ... (not
   counting revised quotes)" and "new meetings ... (not counting meetings you
   moved)" were read by the summary arms as "leave out quotes later revised"
   and "meetings later moved". The answer key counts every first quote and
   every first booking, and the wording now says so explicitly.
3. **IDs that changed between related records** (smoke run, in the rolling
   summary). Completing a task or moving a meeting logged a fresh random task
   or event ID, not the one created earlier, and the summariser rightly
   flagged the mismatch. Budget-sheet rows were also random, and calendar
   busy blocks could fall on the hour being booked. IDs and rows are now set
   when a thing is created and reused (tested), and all data was
   regenerated. The sizes above are from the regenerated data, and the
   sanity history was re-chosen by the same rule (it had been seed 112's
   first three days).
4. **The contact-update record did not name its project** in one of its two
   phrasings (found by a unit test before any run). Its reply now names the
   project, as every other action's record does.
5. **Periods measured in hours** (smoke run, seed 0 at 2 weeks). "In the past
   7 days" was 7 x 24 hours back from the checkpoint in the answer key, while
   every arm, the long context included, read it as calendar days. Periods
   are now whole days ending today, and every question names them: "in the
   past 7 days (Thursday 15 January to today)". The interactions and
   checkpoints were unchanged (checked by hash), so no consolidation was
   redone; only the answers were.
6. **The query-time arm put the question before 70k characters of records**
   (smoke run). One map call answered "Nothing relevant" with five matching
   records in its input. The map and merge prompts now give the records
   first and the question and instructions after them, as is usual for long
   inputs.
7. **A slip in fixing 5**: a local name overwrote each question's checkpoint
   label, and the next sanity answers failed on it. It was caught at once,
   and a test now checks that every question carries its checkpoint.

Faults 5 to 7 changed only questions and the query-time arm, so they were
fixed before any main answer was made. The smoke answers that are kept
(`runs/haiku/answers_smoke.jsonl`, seed 0 at 2 and 8 weeks, every arm) are
from the final code and are not counted in the results.

Two behaviours seen in the smoke transcripts are left as they are, because
they belong to the arms, not to faults:

- A summary states dates without saying what kind they are, so a model
  sometimes takes a meeting's scheduled date for the last day work was done
  ("last did anything for the Milan Design Week trip": 2 February, a
  scheduled showroom visit, against 25 January). The rolling summary keeps a
  dated timeline and got that question right.
- Both models write past the summaries' word limits (Haiku by 20 to 50%,
  luna by up to 90% on the smallest windows). The snapshot's budget guard
  records any summary or raw record it has to drop, and the results report
  how often that happened.

### Checks during the main runs

- **Query-time rows with no map calls are correct** (checked 10 October,
  02:05 UTC, on 297 query-time answers from Haiku's main run). The first 16
  query-time rows to finish all showed zero map calls, which looked wrong at 26
  weeks. They were the questions whose routed selection fits the 32,000-character
  budget (today, the past 7 days, one past week, the last 10 or 40
  interactions: 0 to 28,436 characters), which the arm passes in full with no
  map call, by design. They finish first because they need only the route
  call. Every one of the 158 rows whose selection exceeded the budget made map
  calls (1 to 8, one per 200,000 characters; 8 for seed 1's whole record), so
  no run was affected and nothing was re-run. One whole-record transcript
  (seed 1, 26 weeks, "every project ever", 8 map calls, 1.44M characters into
  18k of notes) was read in full and scored 1.0. On the finished runs the
  same holds: of the query-time answers whose selection exceeded the budget
  (392 of 496 on Haiku, 395 of 496 on luna, 8 of 10 on sol), none skipped the
  map step. No merge round was ever needed, since 8 notes of at most 250
  words fit the budget, and no route failed to parse.
- **Every arm's transcripts were read** at 26 weeks on Haiku and luna (a time
  question, a count question, a detail, a count of actions and a gap question,
  each in all ten arms) and on the Opus answers. The answers follow from the
  memory each arm was given; the failures are the arms' own (a summary that
  never recorded a silent stretch, a retrieval that found the wrong 53
  records for "the last 480 interactions"), apart from the provider-filter
  blocks and the token-cap overruns listed under the caveats.
- **No run was restarted or discarded after the takeover** (10 October,
  02:00 UTC, after a usage limit stopped the first agent). The three jobs then
  in flight (luna's consolidation and two Haiku answer runs) ran to the end
  and were checked for duplicates and gaps: none.

## Results

Written 10 October 2026, 03:40 UTC, from `results.json` (`uv run python
analyse.py`). Every number below is in `results.json`; the analysis prints
the full tables for every checkpoint.

### Verdict

**The idea holds in part.** By the plan's own test, H2 and H4 hold and H1 and
H3 do not. On the questions it exists for (which projects were worked on in a
period of time, or in the last N interactions), the hierarchy beats every
fixed-budget design a competent engineer would build, on both cheap model
families, and matches the whole record in a million-token context at a
twentieth of the cost per question or less and with a faster first token; and
consolidating in the background costs the conversation nothing. But it is not
the best fixed-budget memory overall: once detail, count and gap questions
are included, Mem0-style fact extraction beats it on Haiku (by 0.14) and is
level with it on luna, and Haiku's rolling summary is narrowly ahead too,
because a summary cannot keep an old quote or a meeting date that a search
over facts or records still finds. Query-time summarisation of the raw
records beats it overall and on its own questions, at 17 times the cost per
question and five times the wait.

### What ran

| condition | model | seeds and checkpoints | arms | answers |
|---|---|---|---|---|
| cheap tier (primary) | `anthropic/claude-haiku-5.5`, reasoning off | 0 to 4, all five checkpoints | all ten | 4,960, none missing |
| cheap tier, second family | `openai/gpt-6-luna`, reasoning off | 0 to 4, all five checkpoints | all ten | 4,960, none missing |
| mid tier | `openai/gpt-6.1-sol`, low effort | 0, 26 weeks | all but rolling and Mem0; query-time on T and C | 157, none missing |
| top tier | `anthropic/claude-opus-5.5`, high effort | 0, 8 weeks (consolidated to 8 weeks) | hierarchy, recent, retrieval, full; T and C | 36, of which 20 blocked by Opus's filter |
| live loop | Haiku | three busy days of seed 0 (2 to 4 February), 69 turns per mode | off, background, inline, Mem0 inline | 345 turns |

Every planned run was made. Nothing was discarded after the main runs began.

### Headline: 26 weeks, Haiku 5.5 (five seeds, 105 questions per arm)

Accuracy is the mean score (set F1 for T and C, exact match otherwise);
"macro" is the mean of the five family means. Intervals are 95% bootstrap
intervals over questions. Cost is per question, every call on the read path
and the answer included, cache reads as charged. First token is the time from
the question to the reply's first token, read path included.

| arm | macro | T+C | T | C | D | N | G | $/question | first token, median (p90) | memory |
|---|---|---|---|---|---|---|---|---|---|---|
| hierarchy | 0.57 (0.50 to 0.65) | **0.91** (0.85 to 0.95) | 0.93 | 0.88 | 0.70 | 0.30 | 0.07 | 0.0015 | 0.96 s (1.26) | 28k chars |
| hierarchy, time view only | 0.58 (0.50 to 0.66) | 0.91 (0.87 to 0.95) | 0.91 | 0.90 | 0.57 | 0.40 | 0.13 | 0.0009 | 0.97 s (1.33) | 17k |
| hierarchy, count view only | 0.59 (0.51 to 0.68) | 0.89 (0.83 to 0.94) | 0.88 | 0.92 | 0.57 | 0.40 | 0.20 | 0.0010 | 0.97 s (1.24) | 18k |
| rolling summary | 0.66 (0.58 to 0.74) | 0.84 (0.77 to 0.89) | 0.83 | 0.86 | 0.80 | 0.70 | 0.13 | 0.0017 | 1.01 s (1.32) | 32k |
| recent records | 0.38 (0.31 to 0.46) | 0.66 (0.56 to 0.76) | 0.55 | 0.82 | 0.27 | 0.20 | 0.07 | 0.0014 | 0.95 s (1.26) | 32k |
| retrieval (hybrid) | 0.62 (0.54 to 0.70) | 0.60 (0.53 to 0.67) | 0.63 | 0.55 | 1.00 | 0.60 | 0.33 | 0.0015 | 1.41 s (1.67) | 32k |
| Mem0 | **0.72** (0.63 to 0.80) | 0.77 (0.70 to 0.83) | 0.86 | 0.63 | 1.00 | 0.70 | 0.40 | 0.0014 | 1.25 s (1.89) | 32k |
| query-time summarisation | 0.98 (0.95 to 1.00) | 0.99 (0.98 to 1.00) | 1.00 | 0.98 | 1.00 | 1.00 | 0.93 | 0.0245 | 5.23 s (8.50) | 2k |
| full record, cached | 0.84 (0.76 to 0.91) | 0.86 (0.81 to 0.91) | 0.89 | 0.83 | 1.00 | 0.80 | 0.67 | 0.0353 | 1.59 s (9.85) | 0.8 to 1.4M |
| none | 0.05 (0.00 to 0.10) | 0.02 (0.00 to 0.06) | 0.03 | 0.00 | 0.00 | 0.20 | 0.00 | 0.0001 | 0.92 s (1.17) | 0 |

Hierarchy minus each arm, paired by question (95% interval):

| arm | macro | T+C | D | N | G |
|---|---|---|---|---|---|
| rolling summary | -0.09 (-0.18 to -0.00) | +0.07 (0.01 to 0.14) | -0.10 (-0.30 to 0.10) | -0.40 (-0.70 to -0.10) | -0.07 (-0.27 to 0.13) |
| recent records | +0.19 (0.12 to 0.27) | +0.25 (0.15 to 0.35) | +0.43 (0.27 to 0.60) | +0.10 (0.00 to 0.30) | 0.00 (-0.20 to 0.20) |
| retrieval | -0.05 (-0.14 to 0.04) | +0.31 (0.22 to 0.39) | -0.30 (-0.47 to -0.13) | -0.30 (-0.60 to 0.00) | -0.27 (-0.53 to 0.00) |
| Mem0 | -0.14 (-0.24 to -0.06) | +0.14 (0.05 to 0.23) | -0.30 (-0.47 to -0.13) | -0.40 (-0.70 to -0.10) | -0.33 (-0.60 to -0.13) |
| query-time | -0.41 (-0.48 to -0.33) | -0.08 (-0.14 to -0.04) | -0.30 (-0.47 to -0.13) | -0.70 (-1.00 to -0.40) | -0.87 (-1.00 to -0.67) |
| full record | -0.26 (-0.36 to -0.15) | +0.04 (-0.02 to 0.11) | -0.30 (-0.47 to -0.13) | -0.50 (-0.90 to -0.10) | -0.60 (-0.87 to -0.33) |

Per seed, the hierarchy's T+C at 26 weeks is 0.94, 0.92, 0.93, 0.94 and 0.81;
the rolling summary's 0.88, 0.90, 0.82, 0.81 and 0.79; Mem0's 0.84, 0.62,
0.85, 0.72 and 0.80. The hierarchy leads the best fixed-budget arm on T+C in
every seed.

### Second family: 26 weeks, gpt-6-luna (five seeds, 105 questions per arm)

| arm | macro | T+C | D | N | G | $/question | first token, median |
|---|---|---|---|---|---|---|---|
| hierarchy | 0.65 (0.57 to 0.72) | **0.91** (0.88 to 0.95) | 0.80 | 0.20 | 0.40 | 0.00017 | 0.74 s |
| rolling summary | 0.49 (0.42 to 0.55) | 0.75 (0.67 to 0.82) | 0.73 | 0.10 | 0.13 | 0.00018 | 0.77 s |
| recent records | 0.40 (0.32 to 0.48) | 0.65 (0.55 to 0.75) | 0.27 | 0.20 | 0.20 | 0.00016 | 0.73 s |
| retrieval | 0.53 (0.45 to 0.60) | 0.57 (0.50 to 0.64) | 1.00 | 0.40 | 0.13 | 0.00112 | 1.02 s |
| Mem0 | **0.68** (0.59 to 0.76) | 0.76 (0.69 to 0.82) | 1.00 | 0.40 | 0.53 | 0.00116 | 1.08 s |
| query-time | 0.79 (0.72 to 0.86) | 0.98 (0.96 to 1.00) | 0.97 | 0.70 | 0.33 | 0.01747 | 3.54 s |
| full record, cached | 0.64 (0.57 to 0.71) | 0.91 (0.87 to 0.95) | 0.93 | 0.40 | 0.07 | 0.00652 | 1.27 s |

Hierarchy minus arm, paired: rolling +0.16 macro (0.08 to 0.25), +0.17 T+C
(0.09 to 0.24); recent +0.24 (0.17 to 0.32), +0.27 (0.17 to 0.37); retrieval
+0.12 (0.02 to 0.21), +0.34 (0.26 to 0.42); Mem0 -0.03 (-0.13 to 0.06), +0.16
(0.09 to 0.23); query-time -0.15 (-0.24 to -0.05), -0.07 (-0.11 to -0.03);
full record 0.00 (-0.08 to 0.09), 0.00 (-0.03 to 0.04). Luna's whole-record
long context is much weaker on luna than on Haiku (G 0.07, N 0.40), and luna
writes longer summaries, so its snapshot overflowed the budget in 227 of 496
answers (the guard dropped raw records, and in 145 answers up to 7 of the
oldest summaries); the hierarchy still led on T+C.

### The hypotheses

| | Haiku 5.5 (the test) | gpt-6-luna (direction) |
|---|---|---|
| H1, best fixed-budget memory overall | **fails**: beats recent only; Mem0 -0.14 and rolling -0.09 ahead of it, retrieval level | **fails** in direction against Mem0 (-0.03); beats rolling, recent and retrieval |
| H2, best on T and C | **holds** against all four (+0.07 to +0.31, every interval above zero) | **holds** against all four (+0.16 to +0.34) |
| H3, within 0.10 of the unbounded references on T and C, faster, 5x cheaper | **fails** against query-time (-0.08, lower end -0.14); **holds** against the full record (+0.04, lower end -0.02); faster and 17x and 24x cheaper than them | same pattern: query-time -0.07 (lower end -0.11), full record 0.00 |
| H4, no slowing | **holds**: background consolidation median first token 1.115 s against 1.108 s with none (0.6%); inline consolidation adds 0.29 s per turn on average, up to 4.3 s, and lifts p90 from 1.43 to 2.63 s; Mem0's inline write adds 3.8 s per turn | (mechanism, not model) |

### Where it pulls ahead: by history length

Macro / T+C, all five seeds at each checkpoint:

| arm | 2 weeks | 8 weeks | 16 weeks | 26 weeks | after the longest break |
|---|---|---|---|---|---|
| Haiku: hierarchy | 0.67 / 0.99 | 0.62 / 0.96 | 0.62 / 0.89 | 0.57 / 0.91 | 0.69 / 0.91 |
| Haiku: best fixed-budget arm | 0.92 / 0.95 (Mem0) | 0.77 / 0.92 (rolling) | 0.74 / 0.89 (rolling) | 0.72 (Mem0) / 0.84 (rolling) | 0.73 (Mem0) / 0.87 (rolling) |
| Haiku: full record | 0.95 / 0.99 | 0.93 / 0.92 | 0.87 / 0.92 | 0.84 / 0.86 | 0.89 / 0.91 |
| Haiku: query-time | 0.99 / 0.99 | 0.95 / 0.98 | 0.96 / 0.99 | 0.98 / 0.99 | 0.96 / 0.99 |
| luna: hierarchy | 0.62 / 0.97 | 0.61 / 0.93 | 0.64 / 0.94 | 0.65 / 0.91 | 0.68 / 0.88 |
| luna: best fixed-budget arm | 0.84 / 0.96 (rolling) | 0.69 / 0.84 (Mem0) | 0.70 / 0.87 (Mem0) | 0.68 / 0.76 (Mem0) | 0.75 (Mem0) / 0.76 (rolling) |
| luna: full record | 0.70 / 0.95 | 0.68 / 0.92 | 0.73 / 0.93 | 0.64 / 0.91 | 0.79 / 0.91 |

The hierarchy's T+C lead over the best fixed-budget arm, paired: Haiku +0.04
(0.00 to 0.09), +0.04 (0.01 to 0.07), 0.00 (-0.05 to 0.04), +0.07 (0.01 to
0.14) and +0.04 (-0.00 to 0.09); luna +0.01 (-0.05 to 0.05), +0.09 (0.01 to
0.16), +0.07 (0.02 to 0.13), +0.16 (0.09 to 0.23) and +0.12 (0.04 to 0.20). On
luna it grows with history length, as the idea predicts; on Haiku it is small
at every size, because Haiku's rolling summary stays good at time questions.
Overall (macro), the hierarchy trails the best fixed-budget arm at every size
on Haiku (by 0.24 at 2 weeks, 0.14 at 26 weeks) and is level with it on luna
from 8 weeks on.

Other measures the plan named:

- **Detail by age** (all checkpoints pooled, Haiku): the hierarchy keeps recent
  details (0.98 up to 7 days old) and loses old ones (0.84 at 8 to 28 days,
  0.41 older), as the idea predicts; the rolling summary keeps 0.95, 0.88,
  0.76; retrieval, Mem0 and the full record get 1.00 at every age. Luna: 1.00,
  0.82, 0.47 for the hierarchy.
- **Gaps** (G) are the hierarchy's weakest family: 0.07 on Haiku at 26 weeks
  (0.40 on luna), against 0.93 for query-time and 0.67 for the full record on
  Haiku. Summaries rarely say how long a silence lasted.
- **After the longest break**, the count view helps count questions a little
  over the time view alone (Haiku +0.05, -0.00 to 0.12; luna +0.04, -0.03 to
  0.10), and the two views together are level with the time view on time
  questions. Not enough to support "time alone is not enough" here; the two
  ablations are within a few points of the full hierarchy everywhere.

### Model strength (mid and top tiers)

Seed 0 at 26 weeks, the same 21 questions (query-time on its 10 T and C
questions), T and C only:

| model | hierarchy | recent | retrieval | query-time | full record | hierarchy - full | hierarchy - best fixed |
|---|---|---|---|---|---|---|---|
| Haiku 5.5 | 0.94 | 0.73 | 0.59 | 0.97 | 0.90 | +0.04 (-0.09 to 0.17) | +0.21 (0.06 to 0.36) |
| gpt-6-luna | 0.93 | 0.73 | 0.53 | 0.98 | 0.93 | 0.00 (-0.04 to 0.04) | +0.20 (0.05 to 0.38) |
| gpt-6.1-sol | 0.97 | 0.73 | 0.60 | 0.99 | 1.00 | -0.03 (-0.06 to 0.00) | +0.24 (0.07 to 0.44) |

On sol, every question about the whole record is answered perfectly by the
long context (1.00 on all 21, macro 1.00, $0.045 per question, 2.8 s to the
first token), against the hierarchy's 0.76 macro and 0.97 on T and C at $0.003
and 2.6 s. So the long context catches up as the model gets stronger, while
the hierarchy's lead over the fixed-budget designs holds. One seed, so the
intervals are wide.

The top tier could not be measured. Opus 5.5's usage-policy filter blocked 8
of the hierarchy's 9 answers (6 refused outright, 2 cut off mid-reply), 6 of 9
for recent records, 5 of 9 for retrieval and 1 of 9 for the full record, so no
question was answered by both the hierarchy and another arm. On what it did
answer: full record 1.00 (8 questions), retrieval 0.88 (4), recent 0.61 (3),
hierarchy 1.00 (1). It also refused 1 of its 123 consolidation calls (the day
summary of Thursday 19 February, 6 interactions), which is empty in its
hierarchy.

### Cost and latency

- **Per question, 26 weeks, Haiku**: hierarchy $0.0015; the other fixed-budget
  arms $0.0014 to $0.0017; query-time $0.0245 (17x); the full record $0.0353
  (24x) with caching, $0.28 for the first question at a checkpoint (which
  writes the cache) and $0.023 after it, against $0.22 per question at list
  price uncached. On luna: hierarchy $0.00017, full record $0.0065 (38x).
- **Time to first token, 26 weeks, Haiku**: hierarchy 0.96 s median; full
  record 9.0 s cold and 1.6 s warm; query-time 5.2 s (route, then parallel map
  calls of up to 200,000 characters).
- **Write paths, per 1,000 interactions** (all five histories): hierarchy
  $0.09 on Haiku and $0.07 on luna; rolling summary $0.81 and $0.43; Mem0
  $0.35 and $0.16; record embeddings $0.011. The hierarchy is the cheapest
  write path by 4 to 9 times. Done inline, the hierarchy's cascades would make
  16 to 21% of turns wait (median 1.6 s, p90 3.6 to 4.2 s, worst 9.8 s on
  Haiku; worst 16 s on luna); Mem0's per-exchange write takes 3.3 s median on
  Haiku.

### Caveats

- **The questions are synthetic and their answers exact.** The histories are
  generated, every record names its project, and the questions ask for lists,
  counts, dates and names. Real logs are messier; the direction of the
  results is more trustworthy than their size.
- **The long-context arm is strong because the record fits.** At 26 weeks the
  record is 0.8 to 1.44M characters, inside today's million-token windows.
  Beyond about a year of this assistant's work it would not fit, and that size
  was not tested.
- **The snapshot budget binds on luna.** Luna writes summaries up to 90% over
  their word limits, so the guard dropped raw records in 227 of luna's 496
  hierarchy answers and old summaries in 145. Haiku's snapshot never
  overflowed.
- **The rolling summary overflows too.** Both models write it past its
  3,000-word limit (Haiku 4,400 to 5,000 words, luna up to 6,100), so in 82 of
  Haiku's 496 rolling answers and 21 of luna's the summary alone exceeded the
  budget and its last 1,500 characters or so were cut.
- **Some long-context replies ran out of tokens.** Every answer has the same
  1,000-token cap. In 9 of Haiku's 496 full-record answers (and 1 Mem0 answer)
  the model scanned the log aloud past the cap and gave no answer, mostly on
  gap questions; they are scored 0. At most 0.02 of the full record's T+C mean
  at 26 weeks.
- **The mid and top tiers are one seed each,** and the top tier is unusable
  for the comparison (above).
- **The live loop covers three days, not four.** The plan says four busy
  days; `run.py live` replays the busiest three-day window of seed 0 (69
  turns per mode), which is what ran, on Haiku only.
- **Anthropic's mid and top models filter this task.** Sonnet 5.5 was replaced
  by gpt-6.1-sol (Amendment 2, addendum); Opus 5.5 kept its runs, with blocked
  answers left out. A reply counts as blocked if it is empty, or if it opens
  the JSON answer and stops far below the token cap without closing it
  (`analyse.py: cut_by_filter`; only Opus rows match).
- **Caching differs by provider.** OpenAI caches repeated prompt prefixes on
  its own, so on luna and sol the arms whose memory is the same for every
  question at a checkpoint (hierarchy, recent, rolling) are charged cache
  rates; on Haiku only the full record is cached. Costs are as charged.

### Bugs found and fixed

Before the main runs: the seven faults listed above under "Faults found
before the main runs". During them (none changed a counted answer):

1. **Mem0's log after a restart** (00:37 UTC). When a Mem0 chain resumed from
   its last saved state (saved every 50 exchanges), the exchanges it redid
   were logged a second time. The resume now drops log rows after the saved
   state. The store itself was always correct; the final logs have one row per
   exchange (checked).
2. **The answer runner loaded Mem0 snapshots for every checkpoint**, so it
   crashed when a later checkpoint's snapshot did not exist yet. It now loads
   only the checkpoints being answered.
3. **Empty streams were recorded as answers.** Sonnet's and Opus's
   usage-policy refusals arrive as a stream with no content, which the shared
   client logged as a successful call with no tokens. An empty stream is now
   followed by one non-streamed call that tells a refusal from a broken stream
   (`refused`, `stream_fallback` in each row). No Haiku, luna or sol answer
   needed it.
4. **The analysis double-counted missing replies** (it loaded each model's
   answers twice), and did not separate the excluded era models from today's
   in the ledger total. Both fixed on 10 October; it also now leaves out
   replies cut off by a provider's filter, and reports T and C margins in the
   tier trend.

### Projected cost of what did not run

From the measured per-call costs and token counts: the mid tier's rolling
summary and Mem0 for seed 0 (gpt-6.1-sol) about $11 and $6; the mid tier's
comparison on seeds 1 to 4 about $27; the top tier at seed 0's 26-week
checkpoint (hierarchy, recent and full on T and C; Amendment 2's alternative,
which needed $17 left below $78) about $18, of which the hierarchy's
consolidation at high effort is about $15 ($0.042 per summary); the top tier
on seeds 1 to 4 at 8 weeks about $48; and Opus's rolling summary and Mem0 to 8
weeks for seed 0 about $15 and $31, since every Mem0 call reasons for about
1,500 tokens at high effort.

### Spend

From both ledgers (`runs/ledger.jsonl`, the laptop's, and
`runs/ledger-vm.jsonl`, the cloud machine's; each call is in exactly one):
**$70.96 on today's models**, under the $80 stop and over the $50 aim. By
model: Haiku 5.5 $34.84, gpt-6-luna $13.83, Opus 5.5 $8.69 (hierarchy to 8
weeks $5.22, answers $3.28, probes $0.19), Sonnet 5.5 $7.98 (excluded runs),
gpt-6.1-sol $5.40, voyage-4 embeddings $0.22. The era models of Amendment 1
cost $1.15 more (o4-mini $0.92, Gemini 2.5 Flash $0.22), kept apart and
outside the cap.

## Re-running

From `concepts/memory-consolidation/` (on the cloud machine, add
`--ledger runs/ledger-vm.jsonl` to every command that calls a model, and copy
`runs/` back afterwards):

```
uv sync && uv run pytest -q                    # 33 tests, every model call stubbed
uv run python run.py generate                  # data/: five half-year histories, the sanity history, questions
K=../../bin/with-openrouter-key
$K uv run python run.py sanity --model haiku   # every arm on a history that fits the budget
$K uv run python run.py sanity --model luna
$K uv run python run.py embed --seeds 0 1 2 3 4                        # record embeddings for retrieval
# cheap tier, both families: every write path, every arm, every checkpoint
$K uv run python run.py consolidate --model haiku --seeds 0 1 2 3 4 --concurrency 6   # hierarchy, rolling, mem0
$K uv run python run.py consolidate --model luna --seeds 0 1 2 3 4 --concurrency 6
$K uv run python run.py answer --model haiku --seeds 0 1 2 3 4 --concurrency 4
$K uv run python run.py answer --model luna --seeds 0 1 2 3 4 --concurrency 4
# mid tier: seed 0 at 26 weeks
$K uv run python run.py consolidate --model sol --seeds 0 --what hierarchy --concurrency 2
$K uv run python run.py answer --model sol --seeds 0 --checkpoints 26w --arms hierarchy hierarchy_time hierarchy_count recent retrieval full none --concurrency 1 --out answers-main.jsonl
$K uv run python run.py answer --model sol --seeds 0 --checkpoints 26w --arms query_time --families T C --concurrency 1 --out answers-qt.jsonl
# top tier: seed 0 at 8 weeks, consolidated only to 8 weeks
$K uv run python run.py consolidate --model opus --seeds 0 --what hierarchy --upto 8w --concurrency 2
$K uv run python run.py answer --model opus --seeds 0 --checkpoints 8w --arms hierarchy recent retrieval full --families T C --concurrency 2 --out answers-main.jsonl
$K uv run python run.py live --model haiku --concurrency 2              # turn latency, consolidation on and off the path
uv run python analyse.py                       # results.json and the tables above
uv run python -m demos_common.ledger runs/ledger*.jsonl
```

The main answers were made by several processes side by side, each writing
its own `runs/<model>/answers-<part>.jsonl` (Haiku: `a`, `b`; luna: `l04`,
`l123`, `rest`; sol: `main`, `qt`; Opus: `main`), split by seeds and arms so
no two wrote the same answer. `answer` skips any question and arm already in
any of a model's `answers*.jsonl` files, and `analyse.py` reads them all. The
records are archived in `runs.tar.zst`, committed in three parts because the whole is over GitHub's 100 MB file limit (the embedding arrays in `runs/emb/`, 246 MB of float32, are most of it): restore with `cat runs.tar.zst.part-* | zstd -dc | tar -xf -`, and after a change rebuild with `tar -cf - runs | zstd -19 -T0 -o runs.tar.zst && split -b 45m runs.tar.zst runs.tar.zst.part-`.

`run.py show --model haiku --file answers-a.jsonl --qid <id> --arm <arm>`
prints one question's memory (query-time only, unless answered with
`--keep-memory`), reply and score. Consolidation resumes where it stopped
(the rolling summary and the Mem0 store are saved as they go).
