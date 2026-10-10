# linked-libraries

**The idea.** What an agent learns is kept in two libraries that are searched as
peers: procedures as functions (name, signature, docstring, code) and rules and
walkthroughs as written guidance (title, text). Each guidance entry is linked to
the functions it governs, many to many, and the link is followed both ways: a
guidance entry lists its functions, and a function shows the guidance linked to
it. A rule therefore lives in one place, so changing it is one edit, and a task
that chooses a procedure fetches every rule linked to that procedure, so no
rule is missed. (Background: the post "Functions and guidance, linked both
ways" and the earlier run on the old code, `landing-page/experiments/functions-and-guidance-linked-both-ways/`.)

## Plan (written 2026-10-09 22:25 UTC, before any experiment run)

### Conditions the idea is built for

From its own reasoning, the idea should matter when:

1. **The library is big.** Hundreds of procedures in tens of skill folders, and
   tens of rules, most of which govern procedures in several folders. With a
   handful of each (the earlier run: 20 procedures, 6 folders, 4 rules) every
   arrangement can be read whole and nothing separates them.
2. **Rules change, and not always on purpose.** Changes come as a direct
   instruction, and in passing in the middle of a task ("Invoice Harbour for
   the onboarding. Oh, and from now on we state amounts in euros."), where the
   task points at one folder and the rule lives in several.
3. **Tasks must find their procedure among many and every rule that applies,**
   including procedures that their folder's description does not mention.

### Arms

Same model, same system prompt apart from one paragraph describing the store
(and the store's own contents where an arm puts them in context), same agent
loop, same requests and tasks. Four ways of keeping the library, and two ways
of reading it at task time:

| Arm | How the knowledge is kept | What the agent sees and uses |
|---|---|---|
| `pasted` | Agent Skills folders (`SKILL.md` with name and description front matter, a list of scripts, `scripts/*.py`). Each rule is copied, word for word, into the `SKILL.md` of every skill whose procedures it governs, with a line naming those procedures. The prompt says copies must be kept identical. | Every skill's name and description in the prompt (as Agent Skills does); file tools: list, read, grep (and edit, write, delete when changing) |
| `rule-skills` | The same procedure skills without rules; each rule is a skill of its own whose description gives the rule's title and what it applies to. | Every skill's name and description in the prompt; the same file tools |
| `rules-file` | The same procedure skills; every rule once, in one `RULES.md`, each saying what it applies to. | Skill listing **and the whole `RULES.md` in the prompt** (the CLAUDE.md pattern); the same file tools |
| `long-context` | `rules-file`'s library (tasks only: it reads the library that `rules-file` changed). | **Everything in one context**: every procedure's signature and docstring grouped by skill, and the whole `RULES.md`; no tools |
| `linked` (the idea) | Two libraries: functions, and guidance entries each linked to the functions it governs. | Tools: search, get and list for each library, side by side; a function comes back with its linked guidance, a guidance entry with its functions; add, update and delete guidance when changing |
| `linked-in-context` (ablation) | `linked`'s library (tasks only). | The whole linked library rendered into the prompt; no tools. Separates the links from the retrieval |

Every rule has the same title and text in every arm: what it applies to (a
list of kinds of work), then the rule. Only the placement differs, plus the
per-skill "governs" line in `pasted` and the explicit links in `linked`.

Search in `linked` is BM25 over names, signatures and docstrings (functions)
and titles and texts (guidance), with no embedding model, so that every model
call goes through `demos_common.LLM`; the folder arms' grep is lexical too.
There is no duplicate check on adding guidance: the minimal idea, not the fix
the post proposes.

### Data

`src/linked_libraries/catalog.py` holds a hand-written company library: about
195 procedures (each a Python function with a docstring and a one-line task
that needs it) in 30 skill folders with descriptions that leave one or two
procedures per folder unmentioned ("hidden"), and 28 rules, each with what it
applies to, its text, one change (the old and the new fact), a direct request
and an in-passing phrasing. A rule's links are the procedures whose kind of
work its "applies to" list names; the catalog states them explicitly.
`generate.py` builds a dataset per size, deterministically from a data seed
(0): it chooses the folders, the rules (preferring rules that span several of
the chosen folders), the rules that change and their order, the procedure each
in-passing change arrives with, and a stratified task set (procedures governed
by a changed rule, by unchanged rules only, by none; hidden ones in
proportion). Sizes, a sweep:

| Size | Folders | Procedures | Rules | Changes | Tasks |
|---|---|---|---|---|---|
| sanity | 2 | ~12 | 2 | 1 | 4 |
| S | 5 | ~30 | 5 | 4 | 24 |
| M | 10 | ~65 | 10 | 6 | 48 |
| L (realistic) | 30 | ~195 | 25 | 10 | 72 |

Before any arm runs, the ground truth is checked for being decidable from what
every arm can see: a capable model of another family (`openai/gpt-6.1-sol`)
reads the whole catalog and each task and names the procedure, and reads every
procedure's docstring against every rule's "applies to" list and names the
rules that apply. I review every disagreement by hand and fix the catalog where
the text, not the model, is at fault. This can only help the baselines that
must judge applicability from prose.

### Procedure, per model, size, arm, change condition and seed

1. **Change phase** (`pasted`, `rule-skills`, `rules-file`, `linked`), from a
   freshly built library: the K changes in order, each handed to a fresh agent
   with read and write tools. *Direct*: an instruction to change the rule.
   *In passing*: a task for a procedure the rule governs, with the change
   mentioned in passing; the agent plans the task and keeps the library up to
   date. Each change is a separate loop.
2. **State**: for every rule and governed procedure, the text that governs the
   procedure afterwards (`pasted`: its skill's `SKILL.md`; `rule-skills`: the
   rule skills; `rules-file`: `RULES.md`; `linked`: the guidance linked to the
   function).
3. **Task phase** (all six arms) on the changed library, read-only: each task
   asks for the one procedure to call and every rule that applies, as JSON
   (procedure name; each rule's title and text). Nothing is executed.

### Measures

- **Per change:** writes (successful edit, write or delete of a file; add,
  update or delete of a guidance entry), places written, tool calls, tokens,
  cost, model latency; *completed* when every procedure the rule governs is
  governed by the new version alone; collateral damage (another rule's text no
  longer intact); for `linked`, links kept and duplicate entries.
- **Per task:** procedure found (exact name); for each rule that applies:
  *current* (new version only, or an unchanged rule reported), *stale* (old
  only), *conflict* (both), *vague* (reported without the changed fact) or
  *missing*; extra rules reported; fully correct (right procedure, every rule
  current, nothing stale); tool calls, prompt and completion tokens, cost,
  model latency.
- Rule versions are judged by `google/gemini-3.8-flash` (a third family) at
  temperature 0 from the rule's old and new fact; reported rules are matched to
  rules by title (exact, then fuzzy, then the judge). Every non-current
  judgment and a random sample of current ones are checked by hand.

### Models, items, seeds

- Primary: `anthropic/claude-haiku-5.5` for every agent, all sizes, provider
  defaults (no reasoning parameter), max 4,000 output tokens a call, at most 40
  steps a change loop and 25 a task loop.
- Second family on the headline (size L, all arms): `openai/gpt-6-luna`.
- Three seeds (independent samples of the same dataset; the libraries are
  rebuilt fresh per seed). Items are the tasks (and the changes), the same in
  every arm and condition, so comparisons are paired.
- Intervals: Wilson for proportions; `bootstrap_diff` over items (seeds kept
  together) for differences, linked minus each baseline.

### What counts as the idea winning

- **Primary outcome:** at size L, on the primary model, both change conditions
  pooled, the share of (task, rule that applies) pairs reported *current*.
  The idea **holds** if `linked` beats every baseline (`pasted`,
  `rule-skills`, `rules-file`, `long-context`) on it with the 95% interval of
  the difference above zero, without finding the right procedure less often
  than any of them (difference's interval not entirely below zero).
- It **holds partly** if it beats some baselines; the verdict names which, and
  in which condition. If the strongest simple baseline (`rules-file` or
  `long-context`) ties or wins, the idea does not hold at this scale, whatever
  it does against `pasted`.
- **Where it pulls ahead:** the same differences at S and M; the idea's
  reasoning predicts the gap grows with size.
- Cost per task and per change, and model latency, are reported beside every
  accuracy number; a win that costs more is reported as a trade.
- The idea's own predictions, checked separately: one write per change against
  one per copy for `pasted`; no stale copies in passing for `linked` but some
  for `pasted`; fewer missed rules for `linked` than for `rule-skills` and
  `rules-file`, more so as rules grow; hidden procedures found at least as often.
- The second family confirms the headline if the differences point the same way
  with intervals above zero.

### Checks before trusting

Unit tests for the stores, tools, search, the loop (model stubbed), state,
scoring and generator. A **sanity** size in which every arm should get every
change and task right. A **smoke** run (size S, one seed, both conditions,
every arm) whose whole transcripts I read in every arm. Any bug, in any arm, is
fixed and everything it touched re-run; discarded runs are kept in
`runs/discarded/` with the reason.

### Budget

Estimated from the probe's prices: about $15 for the primary model over all
sizes, $8 for the second family at L, $2 for judging and $1 for the data check;
aim under $50, stop before $80. If the smoke run projects the full design above
$45, the second family runs one change condition (in passing) only.

### Addendum: the data as built (2026-10-09 22:38 UTC, before any experiment run)

`python -m linked_libraries.generate` (data seed 0) gives:

| Size | Folders | Procedures (hidden) | Rules (spanning several folders) | Links | Changes, in order | Tasks: under a changed rule / unchanged rules only / none; hidden | Task-rule pairs |
|---|---|---|---|---|---|---|---|
| sanity | 2 | 11 (5) | 2 (2) | 9 | R06 | 2 / 0 / 2; 2 | 4 |
| S | 5 | 36 (12) | 5 (5) | 18 | R17, R22, R04, R25 | 12 / 2 / 10; 7 | 15 |
| M | 10 | 67 (24) | 10 (10) | 45 | R25, R04, R20, R18, R02, R03 | 20 / 9 / 19; 17 | 45 |
| L | 30 | 193 (65) | 25 (21) | 149 | R20, R03, R02, R24, R25, R14, R05, R01, R09, R11 | 36 / 18 / 18; 27 | 80 |

(Note added 2026-10-09 22:47 UTC: nine hidden labels were corrected after the
first smoke run, Bugs 5 below. The final counts are S 14 hidden procedures and
9 hidden tasks, M 26 and 19, L 74 and 29; nothing else in this table changed.)

The changed rules are a seeded draw among the rules that span several of the
chosen folders; at L it happens to leave out the widest one (R04, stating
money, 21 procedures in 9 folders), which stays as an unchanged rule that
tasks must still report. Each changed rule at L spans 2 to 7 folders.

**Ground-truth check** (`validate_data.py`, `openai/gpt-6.1-sol`, results in
`data/validation-round1.json` and `data/validation.json`): from the whole
catalog it named the right procedure for 193 of 193 tasks. From the rules'
"applies to" lists it named exactly the catalog's rules for 191 of 193
procedures. The two disagreements were the catalog's fault and were fixed: an
expense report read as personal data (its docstring now says it is totalled by
team and category) and comparing supplier quotes, which states money and now
falls under R04. A second check after the fix agreed on 193 of 193. Cost $0.80.

System prompts at L, in tokens (characters / 4): pasted 1.3K, rule-skills
2.5K, rules-file 3.0K, long-context 9.6K, linked 0.3K, linked-in-context
11.1K. The whole library fits in every context at every size.

### Addendum: a harness limit fixed before M and L ran (2026-10-09 22:56 UTC)

Every tool result was clipped at 12,000 characters (with a note saying how
much was cut). At L, `linked`'s `list_functions` is 16,359 characters, so the
clip would have cut its alphabetical list after about 140 of 193 functions,
and a broad grep over the `pasted` library at L (for example "Applies to",
14,397 characters) is cut too. The limit is now 20,000 characters for every
tool in every arm. It changed while the S runs were in flight (S ran 22:51 to
22:57 UTC); none of their tool results reached 12,000 characters (checked in
the records), so S is as it would have been. The M process started at 22:57:06
UTC, after the change reached the cloud machine, and M, L and the second family
run with the new limit throughout.

### Addendum: one exploratory run after the main results (2026-10-10 00:01 UTC, written just before it was launched)

Not in the plan, and it cannot change the verdict below: the main results
point opposite ways on the two cheap models (Claude Haiku 5.5 and GPT-6
Luna), so one seed of a capable model, `openai/gpt-6.1-sol` (the family whose
cheap model missed rules without links), runs at size L, both change
conditions, every arm, as run `explore-sol` (about $8; renamed `mid-sol` when
the second amendment made it the mid tier's seed 1). It asks whether a
capable model still misses rules when it must judge them from their "applies
to" lists. It is reported separately, as exploratory.

### Amendment from the lead: model strength as an axis (2026-10-10 00:13 UTC; superseded at 00:42, below)

Reason (Dan, through the lead): smarter models need thinner harnesses, so the
fair test of a design is on the models it was built for. From here:

1. **Headline, the design's own era (October 2025):** `openai/gpt-5` with
   reasoning effort high (the original code's setting) as the primary model,
   and `anthropic/claude-sonnet-4.5` (provider defaults) as the second family.
   Every arm in a condition uses the same model. The judge stays
   `google/gemini-3.8-flash`, a family different from both.
2. **Today's tier:** the runs so far (Claude Haiku 5.5 at S, M and L; GPT-6 Luna
   at L) stay, labelled as today's cheap tier, not the headline.
3. **Today's frontier, if the budget allows after the headline:**
   `openai/gpt-6.1-sol` (or `anthropic/claude-opus-5.5` at reasoning effort high)
   on the headline comparison only, fewer seeds if need be, to show whether the
   advantage shrinks with stronger models. The exploratory GPT-6.1 Sol run at L
   below is one seed of this.
4. **Budget:** a hard stop at $80 for the whole concept, including the $16.79
   already spent. If the realistic size (L) does not fit on the era models, the
   headline runs at the largest size that does, and the README reports what L
   would cost.

Settings for the era runs: GPT-5's reasoning effort is set to high through
OpenRouter's `reasoning` parameter; because reasoning tokens count against the
output cap, the cap rises from 4,000 to 32,000 tokens a call for both era
models (a cap, not a behaviour; no earlier call came near 4,000). Everything
else (data, prompts, tools, loop, steps, seeds, judge, measures and what
counts as winning) is as planned, with GPT-5 in the primary model's place and
Claude Sonnet 4.5 in the second family's. Before the main era runs, each era
model goes through the sanity size, and the cost per task and per change it
shows there sets the size, by the rule in point 4.

### Amendment from the lead, superseding the one above (2026-10-10 00:42 UTC)

Dan replaced the era-model amendment. Only today's models, ones someone would
reasonably choose today on the cost and quality frontier; no era models at
all. Model strength stays an axis, in three tiers:

1. **Cheap:** `anthropic/claude-haiku-5.5` and `openai/gpt-6-luna`. The full
   design, both families: Haiku already ran S, M and L; Luna ran L and now also
   runs S and M, three seeds each.
2. **Mid:** `openai/gpt-6.1-sol` (one family is enough), on the headline
   comparison at L. Its exploratory seed 1 at L is kept; seeds 2 and 3 are
   added, every arm, both change conditions.
3. **Top:** `anthropic/claude-opus-5.5` with reasoning effort high, on the
   headline comparison at L, with fewer seeds or items if the budget needs it.

Every arm in a condition uses the same model, and the judge stays
`google/gemini-3.8-flash`. What counts as the idea winning is as in the plan,
read at each tier, and the question becomes how the idea's advantage changes
as models get stronger. The runs on older models made under the superseded
amendment (GPT-5 at the sanity size and part of size S, Claude Sonnet 4.5 at
the sanity size: $18.66 in all, GPT-5 $17.49, Claude Sonnet 4.5 $1.14 and
judging $0.04) are kept in `runs/superseded-era-models/`
with a note, and left out of every analysis. Budget: the hard stop at $80 for
the whole concept stands, with $35.45 spent at this point; a tier that does not
fit at full size runs what fits, and the README reports what the rest would
cost.

**The top tier as run** (added 2026-10-10 01:26 UTC, before it ran). Spend stood at $46.00
after the cheap tier's full design (Luna's S and M added) and the mid tier's
three seeds at L. A probe of Opus 5.5 at reasoning effort high on four L tasks
and two L changes ($0.40) put a task in the five arms of the headline
comparison at about $0.34 (about $0.41 with the ablation arm) and the ten
in-passing changes at about $4.60, so the full design at L (every arm, both
conditions, three seeds) would cost about $200 and one seed of it about $68
(a rough figure, corrected from the run's own costs in the results). What fits under the hard stop, with a
margin: **one seed, the in-passing condition only** (where copies drift and
writers merge), **the five arms of the headline comparison** (`pasted`,
`rule-skills`, `rules-file`, `long-context`, `linked`; the ablation is left
out), **all ten changes, and 60 of the 72 tasks**, a seeded subset that keeps
the task set's make-up (`python -m linked_libraries.generate --subset L 60`:
t001 t002 t003 t004 t005 t006 t007 t008 t010 t011 t012 t013 t015 t017 t018 t019 t020 t021 t022 t023 t024 t025 t026 t027 t028 t030 t031 t032 t033 t035 t036 t037 t038 t039 t040 t041 t042 t043 t044 t047 t049 t050 t052 t053 t054 t055 t057 t059 t060 t061 t062 t063 t064 t065 t066 t067 t068 t069 t070 t071). It runs under a watchdog that stops it at $78.00 of total spend.

## Results

Every number here is in `results.json` and `results.md` (`analyze.py` over
`runs/main-haiku`, `runs/main-luna`, `runs/mid-sol` and `runs/top-opus`; the
superseded era runs are not included). Rates pool the change conditions and
seeds that ran; intervals are 95%: Wilson for a rate, and for a difference a
bootstrap over tasks with each task's seeds kept together. "Rules current" is
the share of (task, rule that applies) pairs for which the task reported the
rule in its current form.

### Verdict

**The idea's advantage on rules is a weak-model effect that disappears as
models get stronger; its mechanical advantages hold at every tier.** On the
cheap tier the explicit link kept every rule for GPT-6 Luna, which misses
rules when it must judge prose "applies to" lists (at L, 100% current against
92.5% to 98.8% for the baselines), but not for Claude Haiku 5.5, which read the
lists as well as the links and came out 2.3 to 2.5 points *below* three
baselines because its linked writer twice merged a change by keeping the old
sentence. On the mid tier (GPT-6.1 Sol, three seeds) the gap is gone (100% for
`linked`, `pasted` and `rules-file`; `long-context` 98.5%), and on the top tier
(Claude Opus 5.5, one seed in passing) every arrangement kept every rule and
found every procedure. At every tier the linked library changes a rule in one
write where pasted copies need about three, at 40% to 90% of their cost per
change, and it finds every procedure that its folder's description does not
mention (174 of 174 on each model at L) where the rules file found 156 to 169
below the top tier. Per task it is the cheapest arm only when prompts are not
cached and the model is cheap; with caching, the whole library in one context
is cheaper and, from the mid tier up, as accurate. The explicit
link is what carries the cheap-tier gain: the same linked library rendered
whole into the prompt did as well as searching it.

### Across model tiers, size L

| Tier | Model | Run | pasted | rule-skills | rules-file | long-context | linked | linked minus the best other arm | Right procedure, linked / rules-file | Writes per change, pasted / linked | Cost per change, pasted / linked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cheap | Claude Haiku 5.5 | 3 seeds, both conditions, 72 tasks | 100.0% | 98.1% | 99.8% | 100.0% | 97.5% | −2.5 [−4.6, −0.8] | 99.3% / 94.2% | 3.05 / 1.00 | $0.0033 / $0.0013 |
| cheap | GPT-6 Luna | 3 seeds, both conditions, 72 tasks | 98.8% | 95.4% | 95.0% | 92.5% | **100.0%** | **+1.2 [+0.2, +2.5]** | 100.0% / 94.9% | 2.98 / 1.00 | $0.0010 / $0.0006 |
| mid | GPT-6.1 Sol | 3 seeds, both conditions, 72 tasks | 100.0% | 99.4% | 100.0% | 98.5% | 100.0% | +0.0 [+0.0, +0.0] | 100.0% / 98.8% | 3.05 / 1.02 | $0.0125 / $0.0113 |
| top | Claude Opus 5.5, effort high | 1 seed, in passing, 60 tasks, 5 arms | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | +0.0 [+0.0, +0.0] | 100.0% / 100.0% | 3.00 / 1.00 | $0.140 / $0.091 |

Rules current is out of 480 (task, rule) pairs per arm on the cheap and mid
tiers and 67 on the top tier. Against each baseline separately, `linked` is:
on Haiku −2.5 (pasted), −0.6 (rule-skills), −2.3 (rules-file), −2.5
(long-context); on Luna +1.2, +4.6, +5.0, +7.5, every interval above zero; on
Sol +0.0, +0.6 [+0.0, +1.9], +0.0, +1.5 [+0.2, +3.3]; on Opus +0.0 throughout.

How the advantage moves with strength, from the records:

- **Missed rules**, the link's promise. Where an arm must judge from prose
  which rules apply, Luna missed 22 to 36 of 480 at L (and 27 to 51 of 270 at
  M); Sol missed 7 (`long-context`) and 3 (`rule-skills`); Opus none. `linked`
  missed none at L on any model.
- **The linked writer's own errors**, which the plan did not foresee: Haiku
  kept the old sentence beside the new one in 3 changes (R02); Luna, changing
  a rule in passing, twice replaced its links with only the procedures of the
  task at hand (R17 at S; R04 at M, where eight procedures lost the money rule).
  Neither happened on Sol or Opus.
- **Pasted copies** went stale once in 310 changes (Luna) and never on Sol or
  Opus; told to keep copies identical, every model grepped for every copy. The
  cost stayed: about three writes and eight tool calls a change at every tier.
- **Hidden procedures** (ones their folder's description does not mention):
  `rules-file` found 156, 159 and 169 of 174 on Haiku, Luna and Sol, and 24 of
  24 on Opus; `linked` found all of them on every model.
- **Cost per task** at L: without caching (Haiku, Opus), on Haiku `linked`
  sends the smallest prompts and is the cheapest arm ($0.0009 against $0.0014
  to $0.0018); Opus explores more in it (4.8 tool calls a task), and it comes
  second to `long-context` ($0.067 against $0.057; `pasted` $0.097). With the automatic caching of OpenAI's models (Luna, Sol) the arms
  that carry the library in the prompt are 82% to 98% cached and cheapest, and
  on Sol `linked` is the most expensive arm ($0.0082 against $0.0017 for
  `long-context`). Model time follows the number of calls: about 2 seconds a
  task for the one-call arms, 4 to 15 for the tool arms.

## The cheap tier in full: Claude Haiku 5.5 and GPT-6 Luna, every size

These were the plan's primary and second-family runs (Haiku at S, M and L and
Luna at L, 2026-10-09 22:51 to 23:50 UTC), with Luna's S and M added under the
second amendment (2026-10-10 00:45 to 01:09 UTC). Three seeds, both change
conditions, every arm.

#### Size L, both change conditions

Claude Haiku 5.5:

| Arm | Rules current | Missed / stale / conflict (of 480) | Right procedure | Hidden procedures found | Fully correct | Cost per task | Prompt tokens per task | Model seconds (median) |
|---|---|---|---|---|---|---|---|---|
| pasted | 100.0% [99.2, 100] | 0 / 0 / 0 | 99.5% [98.3, 99.9] | 172 / 174 | 99.5% | $0.00144 | 11,107 | 5.2 |
| rule-skills | 98.1% [96.5, 99.0] | 3 / 0 / 6 | 97.2% [95.2, 98.4] | 167 / 174 | 95.1% | $0.00175 | 13,769 | 5.0 |
| rules-file | 99.8% [98.8, 100] | 1 / 0 / 0 | 94.2% [91.6, 96.0] | 156 / 174 | 94.0% | $0.00150 | 11,885 | 4.4 |
| long-context | 100.0% [99.2, 100] | 0 / 0 / 0 | 99.5% [98.3, 99.9] | 172 / 174 | 99.5% | $0.00152 | 13,417 | 2.3 |
| **linked** | 97.5% [95.7, 98.6] | 0 / 0 / 12 | 99.3% [98.0, 99.8] | 174 / 174 | 96.5% | **$0.00093** | 6,610 | 4.4 |
| linked-in-context | 97.5% [95.7, 98.6] | 0 / 0 / 12 | 100.0% [99.1, 100] | 174 / 174 | 97.2% | $0.00176 | 16,250 | 1.9 |

GPT-6 Luna (its provider cached prompts automatically, see Cost):

| Arm | Rules current | Missed / stale / conflict (of 480) | Right procedure | Hidden procedures found | Fully correct | Cost per task | Prompt tokens per task | Model seconds (median) |
|---|---|---|---|---|---|---|---|---|
| pasted | 98.8% [97.3, 99.4] | 4 / 0 / 2 | 98.8% [97.3, 99.5] | 171 / 174 | 97.7% | $0.00054 | 8,646 | 8.7 |
| rule-skills | 95.4% [93.2, 97.0] | 22 / 0 / 0 | 96.5% [94.4, 97.9] | 162 / 174 | 91.9% | $0.00044 | 8,132 | 7.9 |
| rules-file | 95.0% [92.7, 96.6] | 24 / 0 / 0 | 94.9% [92.4, 96.6] | 159 / 174 | 89.8% | $0.00039 | 10,449 | 6.1 |
| long-context | 92.5% [89.8, 94.5] | 36 / 0 / 0 | 100.0% [99.1, 100] | 174 / 174 | 91.7% | $0.00019 | 8,795 | 2.6 |
| **linked** | **100.0% [99.2, 100]** | 0 / 0 / 0 | 100.0% [99.1, 100] | 174 / 174 | **100.0%** | $0.00044 | 3,874 | 5.4 |
| linked-in-context | 99.8% [98.8, 100] | 1 / 0 / 0 | 100.0% [99.1, 100] | 174 / 174 | 99.8% | $0.00018 | 10,586 | 2.3 |

Linked minus each arm, in percentage points (cost in thousandths of a dollar):

| linked minus | Haiku: rules current | Haiku: right procedure | Haiku: fully correct | Haiku: cost per task | Luna: rules current | Luna: right procedure | Luna: fully correct | Luna: cost per task |
|---|---|---|---|---|---|---|---|---|
| pasted | −2.5 [−4.6, −0.8] | −0.2 [−1.4, +0.9] | −3.0 [−5.8, −0.7] | −0.51 [−0.59, −0.44] | +1.2 [+0.2, +2.5] | +1.2 [+0.0, +2.8] | +2.3 [+0.7, +4.2] | −0.10 [−0.13, −0.07] |
| rule-skills | −0.6 [−2.7, +1.2] | +2.1 [+0.0, +4.4] | +1.4 [−1.6, +4.4] | −0.83 [−0.91, −0.75] | +4.6 [+2.3, +7.1] | +3.5 [+1.4, +6.0] | +8.1 [+4.9, +11.3] | +0.01 [−0.02, +0.03] |
| rules-file | −2.3 [−4.6, −0.4] | +5.1 [+1.9, +9.0] | +2.5 [−1.4, +6.9] | −0.57 [−0.64, −0.50] | +5.0 [+2.5, +8.1] | +5.1 [+2.3, +8.3] | +10.2 [+6.5, +14.6] | +0.06 [+0.04, +0.08] |
| long-context | −2.5 [−4.6, −0.8] | −0.2 [−1.4, +0.7] | −3.0 [−5.6, −0.7] | −0.59 [−0.63, −0.54] | +7.5 [+4.6, +10.8] | +0.0 [+0.0, +0.0] | +8.3 [+5.1, +12.0] | +0.25 [+0.23, +0.27] |
| linked-in-context | +0.0 [+0.0, +0.0] | −0.7 [−1.6, +0.0] | −0.7 [−1.6, +0.0] | −0.83 [−0.88, −0.78] | +0.2 [+0.0, +0.6] | +0.0 [+0.0, +0.0] | +0.2 [+0.0, +0.7] | +0.26 [+0.24, +0.28] |

Against the plan's criteria as first written (Haiku as the primary model,
Luna as the second family), the idea did not hold on Haiku (`linked` below
`pasted`, `rules-file` and `long-context`, level with `rule-skills`), and Luna
reversed rather than confirmed it (`linked` above all four, intervals above
zero).

#### Changing a rule, at L

Per change, mean over 30 changes per condition (10 rules, 3 seeds); writes
are successful edits, writes or deletes of a file, or adds, updates or
deletes of a guidance entry.

| Arm | Haiku: writes (range) | Haiku: tool calls | Haiku: cost | Haiku: completed, direct / in passing | Luna: writes (range) | Luna: tool calls | Luna: cost | Luna: completed, direct / in passing |
|---|---|---|---|---|---|---|---|---|
| pasted | 3.1 (2–7) / 3.0 (2–7) | 7.6 / 8.1 | $0.0031 / $0.0036 | 30 / 30 | 3.0 (2–7) / 3.0 (1–7) | 8.5 / 10.6 | $0.0008 / $0.0012 | 30 / 29 |
| rule-skills | 1.0 / 1.0 | 2.8 / 6.7 | $0.0018 / $0.0033 | 29 / 30 | 1.0 / 1.0 | 2.3 / 7.4 | $0.0003 / $0.0007 | 30 / 30 |
| rules-file | 1.0 / 1.1 | 2.2 / 4.1 | $0.0018 / $0.0029 | 30 / 30 | 1.0 / 1.0 | 2.8 / 5.7 | $0.0007 / $0.0011 | 30 / 30 |
| linked | 1.0 / 1.0 | 2.6 / 4.1 | $0.0011 / $0.0016 | 30 / 28 | 1.0 / 1.0 | 2.2 / 3.9 | $0.0005 / $0.0008 | 30 / 30 |

Paired over the same changes, `linked` minus `pasted`: 2.05 fewer writes
[1.40, 2.83], 4.5 fewer tool calls [3.2, 6.0] and $0.0020 less [0.0016,
0.0025] per change on Haiku; 1.98 fewer writes, 6.5 fewer tool calls and
$0.00035 less on Luna. Against `rules-file` and `rule-skills` the writes are
the same (one) and `linked` is cheaper on Haiku. No arm in any run, on any
tier, damaged a rule it was not asked to change (every unchanged (rule,
procedure) pair intact, 303 of 303 per arm and condition at L), and `linked`
never added a second entry for a rule (0 adds in 310 changes across the
tiers), the failure the earlier run saw.

#### Across sizes

| Model | Size (procedures, rules) | Rules current: linked / pasted / rule-skills / rules-file / long-context | linked minus rules-file | linked minus long-context | Right procedure, linked / rules-file | Writes per change, pasted / linked |
|---|---|---|---|---|---|---|
| Haiku | S (36, 5) | 100 / 100 / 100 / 100 / 100% | +0.0 | +0.0 | 97.9 / 92.4% | 2.25 / 1.00 |
| Haiku | M (67, 10) | 98.5 / 98.5 / 100 / 100 / 100% | −1.5 [−3.0, −0.4] | −1.5 [−3.0, −0.4] | 100.0 / 97.9% | 2.92 / 1.00 |
| Haiku | L (193, 25) | 97.5 / 100 / 98.1 / 99.8 / 100% | −2.3 [−4.6, −0.4] | −2.5 [−4.6, −0.8] | 99.3 / 94.2% | 3.05 / 1.00 |
| Luna | S (36, 5) | 100 / 100 / 100 / 98.9 / 98.9% | +1.1 [+0.0, +3.3] | +1.1 [+0.0, +3.3] | 100.0 / 97.2% | 2.25 / 1.00 |
| Luna | M (67, 10) | 97.0 / 99.6 / 98.5 / 90.0 / 81.1% | +7.0 [+1.5, +13.3] | +15.9 [+9.6, +23.0] | 100.0 / 95.1% | 2.83 / 1.00 |
| Luna | L (193, 25) | 100 / 98.8 / 95.4 / 95.0 / 92.5% | +5.0 [+2.5, +8.1] | +7.5 [+4.6, +10.8] | 100.0 / 94.9% | 2.98 / 1.00 |

On Luna the gap opens with size, as the idea predicts: nothing at S, then the
prose-judging arms miss the money and product-name rules, whose "applies to"
lists are the longest (R04 and R06 make up 69 of `rules-file`'s and
`long-context`'s 78 misses at M). At M `linked` lost eight pairs of its own
when a change in passing narrowed the money rule's links, and fell below
`pasted` there (−2.6 [−4.8, −0.7]). On Haiku no arm missed a rule at S or M.

## The mid tier: GPT-6.1 Sol, size L, three seeds

Every arm, both change conditions, all 72 tasks (seed 1 was the exploratory
run made before the amendments; seeds 2 and 3 were added after the second).

| Arm | Rules current | Missed / stale / conflict (of 480) | Right procedure | Hidden procedures found | Fully correct | Cost per task (cached share) | Model seconds (median) |
|---|---|---|---|---|---|---|---|
| pasted | 100.0% [99.2, 100] | 0 / 0 / 0 | 100.0% [99.1, 100] | 174 / 174 | 100.0% | $0.0050 (82%) | 6.4 |
| rule-skills | 99.4% [98.2, 99.8] | 3 / 0 / 0 | 98.4% [96.7, 99.2] | 167 / 174 | 97.7% | $0.0039 (89%) | 5.0 |
| rules-file | 100.0% [99.2, 100] | 0 / 0 / 0 | 98.8% [97.3, 99.5] | 169 / 174 | 98.8% | $0.0028 (93%) | 4.4 |
| long-context | 98.5% [97.0, 99.3] | 7 / 0 / 0 | 100.0% [99.1, 100] | 174 / 174 | 98.4% | $0.0017 (98%) | 1.7 |
| linked | 100.0% [99.2, 100] | 0 / 0 / 0 | 100.0% [99.1, 100] | 174 / 174 | 100.0% | $0.0082 (40%) | 5.9 |
| linked-in-context | 100.0% [99.2, 100] | 0 / 0 / 0 | 100.0% [99.1, 100] | 174 / 174 | 100.0% | $0.0021 (98%) | 1.8 |

Every arm completed every change (60 of 60); `pasted` wrote 3.05 times a change
against 1.02 for `linked`, and cost $0.0125 against $0.0113 a change. `linked`
minus `long-context` is +1.5 [+0.2, +3.3] on rules current and +0.0 against
the others; minus `rules-file` on the right procedure, +1.2 [+0.0, +3.0].

## The top tier: Claude Opus 5.5 at reasoning effort high, size L, one seed

As fitted to the budget (the plan's "top tier as run"): in passing only, the
five arms of the headline comparison, all ten changes, 60 of the 72 tasks.

| Arm | Rules current | Missed / stale / conflict (of 67) | Right procedure | Hidden procedures found | Cost per task | Model seconds (median) | Writes per change | Cost per change |
|---|---|---|---|---|---|---|---|---|
| pasted | 100.0% [94.6, 100] | 0 / 0 / 0 | 100.0% [94.0, 100] | 24 / 24 | $0.097 | 15.1 | 3.0 | $0.140 |
| rule-skills | 100.0% [94.6, 100] | 0 / 0 / 0 | 100.0% [94.0, 100] | 24 / 24 | $0.077 | 11.4 | 1.0 | $0.106 |
| rules-file | 100.0% [94.6, 100] | 0 / 0 / 0 | 100.0% [94.0, 100] | 24 / 24 | $0.072 | 10.6 | 1.0 | $0.093 |
| long-context | 100.0% [94.6, 100] | 0 / 0 / 0 | 100.0% [94.0, 100] | 24 / 24 | $0.057 | 4.4 | (rules-file's) | – |
| linked | 100.0% [94.6, 100] | 0 / 0 / 0 | 100.0% [94.0, 100] | 24 / 24 | $0.067 | 12.9 | 1.0 | $0.091 |

Every arm completed all ten changes and every task was fully correct. Opus
once rewrote more than the fact it was asked to change: in `linked` it widened
R02's "applies to" list ("anything public: ... website pages, event pages,
story pitches...") while leaving the entry's links as they were, which this
experiment does not score.

### What the reduced top tier leaves out, and what it would cost

The top tier ran one seed of one condition on 60 tasks and five arms, for
$26.45. From its own cost per loop (`estimate_full_size.py --run top-opus`,
with the ablation arm scaled from `long-context` by their ratio on Haiku), the
full design at L, every arm, both conditions, three seeds, all 72 tasks, would
cost about **$214**: about $0.44 a task across six arms and $0.43 a change
across four. The mid tier ran the full design at L ($12.63). Neither the mid
nor the top tier ran S or M, which the idea is not built for.

### Where each arm failed (read in the transcripts)

- **linked**, two kinds of writer error, both on the cheap tier. *Keep and
  append* (Haiku, 3 changes, all R02): the writer rewrote the entry's whole
  text keeping "Refer to a customer by its company name..." and adding "Do not
  name a customer... unless they have agreed in writing", so every task under
  R02 reported both. `rule-skills` did the same once, and `pasted` once in all
  its copies. The folder arms edit by replacing one sentence; `update_guidance`
  rewrites the whole text, as the original system's update does. *Narrowed
  links* (Luna, 2 changes, both in passing): asked to state money in euros
  while preparing an order form, Luna sent `update_guidance` with `functions:
  ["send_order_form", "create_quote"]`, replacing ten links with two, and eight
  procedures lost the rule; at S it relinked R17 to `reply_to_ticket` alone.
  This is the linked library's version of a copy left stale, and the earlier
  run had flagged it as a risk. No duplicate entry was ever added (0 adds in
  310 changes).
- **pasted**: told that copies must be kept identical, every model grepped for
  every copy and edited each, even in passing. One stale copy in 310 changes
  (Luna, in passing: R25 in `support`, the folder whose survey procedure its
  description does not mention). The drift that motivates the idea, 9 stale
  rules in 42 in the earlier run, nearly disappears with today's models and
  one sentence in the prompt; what remains is about three writes and eight
  tool calls a change.
- **rules-file, long-context, rule-skills**: misses concentrate on procedures
  governed by two or three rules. Asked for a press release, Luna with all 25
  rules in its prompt reported "Talking to the press" and left out "Naming
  customers in public material" and "Our product's name", whose lists both
  name press releases; for a job interview it reported the rules on meetings
  and on candidates and left out the third, on stating times. `linked`'s
  `get_function` returned every linked rule and Luna reported them all. Haiku
  made these misses 4 times in 1,440 pairs at L, Sol 10 times, Opus none.
- **rules-file** found the procedure least often below the top tier: it reads
  the skill listing and the rules in its prompt, opens the one folder the task
  points to, and stops there. Its misses are procedures in another folder than
  the obvious one (`send_csat_survey` in support, taken for customer success's
  NPS survey; `book_speaker_travel` in events, taken for plain flight and hotel
  bookings; `send_offer_letter` in HR, taken for a recruiting follow-up;
  investor and partner calls taken for the generic external meeting).

### R02, and a post hoc sensitivity

R02's request ends "The rest of the rule stays", and its old fact ("refer to a
customer by its company name") overlaps a sentence that can be read as
staying (how to refer to a customer who has agreed to be named). Under the
pre-registered definition a text that keeps that sentence beside the new
restriction "still states the old version so that a reader could follow it",
and the judge called it a conflict; read whole, such a text is coherent, if
clumsy. The judge itself was not consistent on it: one `pasted` copy at M
that put the new sentence first and kept the old one after was judged
current inside its `SKILL.md` and a conflict when a task quoted it alone. As a
check that is not in the plan, counting R02 keep-and-append conflicts as
current: on Haiku at L `linked` reaches 100.0% and ties every arm (differences
+0.0 to +0.6 points, intervals touching zero); Luna is unchanged. The verdict
stands on either reading: the idea does not win on Haiku, and wins on Luna.

### Judge checks

The judge (`google/gemini-3.8-flash`, temperature 0) made 766 version
judgments for Haiku, 828 for Luna, 286 for Sol and 94 for Opus, cached by rule
and text; mapping reported rules to rules by title never needed it. I read
every judgment that was not "new only" (24 for Haiku, all the R02 texts above;
7 for Luna: the stale R25 copy, two answers quoting both copies, and four
governing texts left empty or without the rule by the narrowed links; none for
Sol or Opus) and agree with each. I read a seeded sample of "new only"
judgments (25 for Haiku, 33 for Luna, 18 for Sol, 8 for Opus) and every R04
judgment for Haiku (72) against the sentence carrying the fact, and agree with
all.

### Cost

The ledgers (`runs/ledger.jsonl`, `runs/ledger-laptop.jsonl`) record **$72.59**
in all, every call included, under the $80 hard stop:

| What | Spend |
|---|---|
| Cheap tier: Claude Haiku 5.5 (all sizes, with sanity and smoke) | $7.86 |
| Cheap tier: GPT-6 Luna (all sizes, with sanity) | $2.29 |
| Mid tier: GPT-6.1 Sol (L, three seeds) | $12.63 |
| Top tier: Claude Opus 5.5 (L, reduced; and its $0.40 probe) | $26.85 |
| Judge: Gemini 3.8 Flash | $3.53 |
| Ground-truth check: GPT-6.1 Sol | $0.81 |
| Superseded era runs, left out of the analysis: GPT-5 $17.49, Claude Sonnet 4.5 $1.14 | $18.63 |

Prices are OpenRouter's. OpenAI's models cached repeated prompt prefixes
automatically; Anthropic's were not cached (no cache markers were sent).

### Caveats and threats to validity

- **The tiers are not one family.** Cheap has both families; mid is OpenAI
  only; top is Anthropic only and one seed of one condition on 60 tasks, so its
  intervals are wide (a 100% rate over 67 pairs has a lower bound of 94.6%).
- **A synthetic library.** One hand-written company, 193 procedures and 28
  rules with unusually explicit "applies to" lists, checked so that a careful
  reader could get every link from the text. Vaguer rules should favour links;
  links going stale as procedures are added was not tested.
- **Ten changes at L, mostly one fact each,** applied in one order, each by a
  fresh agent; one (R02) was badly specified (above). Rules here never
  contradict each other.
- **Tasks stop at naming the procedure and the rules;** nothing is executed,
  so whether an agent obeys the rule it reports is not measured.
- **Search is BM25,** not embeddings, so that every model call goes through the
  shared client; the folder arms' grep is lexical too. `linked` found 99% to
  100% of procedures anyway.
- **The prompt told `pasted` to keep copies identical,** the strongest version
  of that arm; without it, copies may drift as in the earlier run.
- **The folder arms edit by find and replace and `linked` rewrites a whole
  entry, links included.** That difference is the likely cause of both of
  `linked`'s writer errors; a guarded update (one that changes the text or the
  links, not both at once, or that refuses to drop links silently) was not
  tested and might remove them.
- **One judge,** checked by hand on every non-current judgment and a sample of
  the rest; R02 sits on the edge of its definition.
- **Three seeds of model sampling on one dataset per size** (one on the top
  tier); items are tasks, so intervals cover task-to-task variation, not other
  libraries.

## Bugs found and fixed

Every fault found, in any arm or in the harness, with what it touched. The
discarded runs are in `runs/discarded/` with their reasons
(`runs/discarded/README.md`).

1. **Catalog, before any run** (`validate_data.py`): two of 193 procedures had a
   rule set that a careful reader would not get from the text (an expense
   report read as personal data; comparing supplier quotes, which states
   money, was not linked to the money rule). Fixed in the catalog; the re-check
   agreed on 193 of 193.
2. **Scoring** (sanity run 1): an answer naming the right procedure by its file,
   `marketing/scripts/export_mailing_list.py (export_mailing_list)`, was
   reduced to `py` and counted wrong. A procedure now counts as found when the
   answer names exactly one procedure of the library, the right one. Only
   baselines had been hurt. Re-scored; the run was repeated anyway for (3).
3. **The `rule-skills` store paragraph** (sanity run 1) said "a procedure skill",
   unlike the other folder arms, and its agent twice answered with a skill's
   name (`marketing`) instead of a function's. The three folder paragraphs are
   now word for word the same apart from where rules live, and the shared prompt
   asks every arm for "the name of the procedure's function". Sanity re-run.
4. **A change request** (smoke run 1): R17's direct request said "sign with
   your own first name", which `rule-skills` and `linked` read as the
   assistant's own name; with none on record they declined the change (linked
   edited the entry and put it back). Reworded to "the first name of the person
   who writes them". Smoke re-run.
5. **Hidden labels** (smoke run 1): nine procedures that their folder's
   description does not mention were not labelled hidden, among them
   `respond_to_complaint`. Labels fixed; no task choice depends on them.
6. **Tool results clipped at 12,000 characters** (found before M and L ran):
   `linked`'s function list at L is 16,359 characters. Raised to 20,000 for
   every tool; see the plan's addendum. No S result had been clipped.
7. Test-only: a helper in the end-to-end test took a tool's `name` argument as
   its own, and a check compared records in arrival order.
8. Operational: when the superseded GPT-5 run was stopped on the cloud machine,
   the kill loop's pattern also matched the loop's own shell, which died after
   the run's processes (listed first) had been stopped. A listing with a
   single bracketed pattern then showed no other process touched. Later stops
   used `watchdog.sh`, which kills by PID from a single-pattern listing.

## Files

- `src/linked_libraries/catalog.py`: the hand-written library (30 skill
  folders, 193 procedures with a task each, 28 rules with one change each).
- `src/linked_libraries/generate.py`: seeded datasets per size, written to
  `data/{sanity,S,M,L}.json`; `data/validation*.json`: the ground-truth check.
- `src/linked_libraries/library.py`: the four ways of keeping the knowledge
  (pasted, rule skills, rules file, linked) and the two whole-library renders.
- `src/linked_libraries/tools.py`: the file tools, the linked libraries' tools
  and BM25 search. `agent.py`: the one tool loop. `arms.py`: the shared prompt,
  each arm's store paragraph and tools. `runner.py`: chains of changes then
  tasks, the governing texts, resumable records. `scoring.py`: matching
  reported rules, the judge's prompts, task scores.
- `run.py`, `judge.py`, `analyze.py`, `inspect_runs.py`, `validate_data.py`,
  `estimate_full_size.py` (what a reduced run would cost as the full design),
  `run_all.sh` (every run, in order), and `capped_run.sh` with `watchdog.sh` (a run
  stopped by PID if the concept's spend reaches a ceiling); `tests/` (pytest, the model stubbed, including a scripted oracle
  run through every arm and the failure each arm is prone to).
- `runs/main-haiku/`, `runs/main-luna/` (cheap tier), `runs/mid-sol/` (mid
  tier) and `runs/top-opus/` (top tier): `changes.jsonl` (every change loop,
  with every tool call and result), `state.jsonl` and `texts.jsonl` (what
  governs each procedure afterwards), `tasks.jsonl` (every task loop),
  `judgments.jsonl`, `chains.jsonl`, and `libs/` (every library as the changes
  left it). `runs/sanity*/`, `runs/smoke/`: the checks. `runs/discarded/`:
  runs set aside for a fault, with the reason. `runs/superseded-era-models/`:
  the runs on older models under the superseded amendment, kept and left out
  of every analysis. `runs/ledger.jsonl`
  (cloud machine) and `runs/ledger-laptop.jsonl` (laptop): every model call.
- `results.json`, `results.md`: every number below.

## Re-running

```sh
cd concepts/linked-libraries
uv sync && uv run pytest -q
uv run python -m linked_libraries.generate                 # data/*.json (deterministic)
../../bin/with-openrouter-key uv run python validate_data.py --ledger runs/ledger-laptop.jsonl
for r in sanity:anthropic/claude-haiku-5.5 sanity-luna:openai/gpt-6-luna; do
  ../../bin/with-openrouter-key uv run python run.py --run ${r%%:*} --model ${r#*:} --size sanity --seeds 1 --ledger runs/ledger-laptop.jsonl
  ../../bin/with-openrouter-key uv run python judge.py --run ${r%%:*} --ledger runs/ledger-laptop.jsonl
done
../../bin/with-openrouter-key uv run python run.py --run smoke --size S --seeds 1 --ledger runs/ledger-laptop.jsonl
../../bin/with-openrouter-key uv run python judge.py --run smoke --ledger runs/ledger-laptop.jsonl
./run_all.sh            # every tier's runs, judging, the analysis and the top tier's full-size estimate
uv run python inspect_runs.py judge --run main-haiku        # what the hand checks read
uv run python inspect_runs.py failures --run main-luna --size M --arm linked --kind change   # e.g. the narrowed links
```

The run takes about two hours at 12 calls in flight, most of it the top tier.
Runs on the laptop use `--ledger runs/ledger-laptop.jsonl`; on the cloud machine
the default `runs/ledger.jsonl`.
