"""Write the pool of filler exchanges for the conversations, once, with a model.

    ../../bin/with-openrouter-key uv run python scripts/make_filler.py            # batch 1, short replies
    ../../bin/with-openrouter-key uv run python scripts/make_filler.py --fuller   # batch 2, appended

Pool v2 (the one used): the assistant in filler never acts or looks anything up, since a
remembered "Done, I've moved it" with no tool call behind it made agents redo or retract it.
Pool v1 is kept in data/discarded/.

Asks claude-sonnet-5.5 for 13 exchanges on each of 16 everyday themes (batch 1:
texting-length replies; batch 2: fuller replies, such as drafts, summaries and
explanations, as a long day with an assistant also has), then
drops every exchange that matches any scenario's `avoid` fragments, so filler
never names a scenario's people, places or topic. The result,
data/filler_exchanges.json, is the canonical pool: re-running this script
gives a different pool, so the experiment reads the saved file.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

from demos_common import LLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from conversation_context.scenarios import SCENARIOS  # noqa: E402
from conversation_context.world import PEOPLE  # noqa: E402

MODEL = "anthropic/claude-sonnet-5.5"
OUT = Path(__file__).resolve().parents[1] / "data" / "filler_exchanges.json"
THEMES = [
    "quick advice on handling meetings and colleagues",
    "drafting short messages for her to send to colleagues",
    "what to wear for different kinds of June weather",
    "cooking and quick meal ideas",
    "running, cycling and yoga tips",
    "book, film, TV and podcast recommendations",
    "learning Spanish: words and phrases",
    "gardening and houseplant care",
    "household cleaning tips",
    "how-to answers about phone and laptop settings, explained step by step",
    "quick facts, unit conversions and trivia",
    "explaining economics and technology terms she hears about",
    "general advice for planning a countryside walk",
    "advice on prioritising work and keeping focus",
    "music recommendations",
    "small talk about her day and how she's feeling",
]
FULLER_THEMES = [
    "explanations of product management ideas and frameworks",
    "drafts of longer messages and updates for her to send",
    "explanations of product metrics and analytics ideas",
    "advice on planning a busy week, from what she tells it",
    "advice on a tricky situation at work that she describes",
    "short overviews of what is generally known about a product topic",
    "comparisons of well-known software tools",
    "tidying up notes she pastes into the chat",
    "recipes with steps",
    "a training plan for a half marathon",
    "book, film and TV recommendations with reasons",
    "explanations of economics and technology terms",
    "Spanish grammar explanations",
    "how-to answers about home and tech",
    "brainstorming names and ideas for a project",
    "feedback on a paragraph she pastes in",
]
FORBIDDEN = (
    "travel bookings of any kind (flights, airports, trains, hotels, seats, tickets, check-in), cars, tyres, "
    "garages or parking, insurance, money owed, payments, bills, fees, invoices, prices, quotes, orders, "
    "deliveries, parcels or furniture, children, kids, schools, nurseries, tutoring, babysitting or lessons, "
    "Maya's family (parents, mum, dad, siblings, partner), pets or vets, appointments, doctors, dentists or physio, "
    "birthdays, dinners, restaurants, parties, cakes or bakeries, gifts, presents or collections, weddings or "
    "leaving dos, codes, passwords or wi-fi, contracts, agreements, signing or anything legal, measurements, "
    "windows or curtains, design reviews, budgets or slides, conferences, Berlin, Lisbon, Edinburgh, Manchester, "
    "York, Bath or Bristol"
)
PROMPT = """Write {n} short, separate exchanges between Maya Brooks and her AI personal assistant, as they might appear scattered through one long day of chat. Maya is a product manager in London, and it is June. Theme for this batch: {theme}.

Each exchange is either one message from Maya and one reply from the assistant, or two of each. Maya writes like a busy person texting: short, casual, often no greeting. The assistant replies in one to three sentences. Make every exchange about something different, self-contained and specific.

In these exchanges the assistant never acts and never looks anything up: it never says it has done, sent, moved, set, saved, booked, scheduled or reminded anything, never offers to, and never mentions Maya's own calendar, messages, emails, files, notes, tasks or lists. It answers only from general knowledge and from what Maya says in her message: it gives advice, explains, recommends, or drafts text for her to send herself.

People who may appear (no one else): {people}.

Never mention any of the following, even in passing: {forbidden}.

Return only JSON: a list of exchanges, each a list of messages like {{"from": "Maya" or "Assistant", "text": "..."}}."""
FULLER = ("The assistant replies in one to three sentences.",
          "The assistant's reply is a fuller answer, as an assistant gives when asked for one: four to ten "
          "sentences, or a short draft, summary or list. If there is a second round, Maya's follow-up is short "
          "and the reply to it can be short.")


def avoid_patterns() -> list[re.Pattern]:
    return [re.compile(r"\b" + frag, re.I) for s in SCENARIOS for frag in s["avoid"]]


# The world is set in June, and the people above are colleagues and friends, not
# housemates: drop exchanges that say otherwise (frost, 4pm sunsets, single-digit
# temperatures, a cleaning rota with a colleague).
CONSISTENCY = re.compile(r"frost|puffer|gloves|winter|sunset is at [1-7]:\d\d|\b[0-9]°C|hoovering|did the bathroom", re.I)
# Pool v1 had the assistant say "Done, I've moved it" with no tool call behind it, and agents then
# redid or retracted those turns (runs/discarded/README.md). Pool v2 is written without actions;
# this catches any that slip through.
ACTION = re.compile(r"\b(done|sent|saved|blocked|booked|scheduled|reminder set|i've (set|moved|sent|added|saved|blocked|"
                    r"created|muted|cancelled|declined|booked|scheduled|messaged|emailed|told|let|made|put|queued|cleared|"
                    r"updated|shared|marked|turned|switched|changed|fixed)|i'll (let you know|remind|message|send|tell|set|"
                    r"ping|check)|want me to|shall i|according to your|your (calendar|inbox|notes?|messages?|emails?|"
                    r"files?|tasks?|list|to-do|rota|stats|battery|last \w+))\b", re.I)


def clean(exchanges: list) -> tuple[list, list]:
    pats = avoid_patterns() + [CONSISTENCY, ACTION]
    keep, drop = [], []
    for ex in exchanges:
        ok_shape = (isinstance(ex, list) and len(ex) in (2, 4)
                    and all(isinstance(m, dict) and m.get("text") for m in ex)
                    and [m.get("from") for m in ex] == ["Maya", "Assistant"] * (len(ex) // 2))
        text = " ".join(m.get("text", "") for m in ex) if isinstance(ex, list) else ""
        hits = [p.pattern for p in pats if p.search(text)] if ok_shape else ["bad shape"]
        (drop if hits else keep).append({"exchange": ex, "hits": hits} if hits else ex)
    return keep, drop


async def main(fuller: bool = False) -> None:
    themes = FULLER_THEMES if fuller else THEMES
    async with LLM(Path(__file__).resolve().parents[1] / "runs" / "ledger.jsonl", "data", max_concurrency=8) as llm:
        async def one(theme: str) -> list:
            msg = PROMPT.format(n=13 if fuller else 16, theme=theme, people=", ".join(PEOPLE), forbidden=FORBIDDEN)
            if fuller:
                msg = msg.replace(FULLER[0], FULLER[1])
                assert FULLER[1] in msg
            for attempt in range(3):  # a batch that is cut off or malformed is asked for again
                r = await llm.chat(MODEL, [{"role": "user", "content": msg}], max_tokens=12000, temperature=1.0)
                text = r.text.strip()
                try:
                    return list(json.loads(text[text.index("["): text.rindex("]") + 1]))
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"{theme}: unreadable batch ({e}), attempt {attempt + 1}")
            return []

        batches = await asyncio.gather(*(one(t) for t in themes))
    allx = [ex for b in batches for ex in b]
    keep, drop = clean(allx)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if fuller:
        d = json.loads(OUT.read_text())
        d.setdefault("batches", [{"name": "short", "themes": d.pop("themes"), "written": d.pop("written")}])
        d["batches"].append({"name": "fuller", "themes": themes, "written": len(allx), "kept": len(keep)})
        d["exchanges"] += keep
        d["dropped"] += drop
        d["kept"] = len(d["exchanges"])
    else:
        d = {"model": MODEL, "themes": themes, "written": len(allx), "kept": len(keep), "exchanges": keep, "dropped": drop}
    OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    print(f"written {len(allx)}, kept {len(keep)}, dropped {len(drop)} -> {OUT} ({d['kept']} in the pool)")


def reclean() -> None:
    """Re-apply the filters to the saved pool (used once after adding CONSISTENCY)."""
    d = json.loads(OUT.read_text())
    keep, drop = clean(d["exchanges"])
    d["exchanges"], d["dropped"], d["kept"] = keep, d["dropped"] + drop, len(keep)
    OUT.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    print(f"kept {len(keep)}, dropped {len(drop)} more")


if __name__ == "__main__":
    if sys.argv[1:] == ["--reclean"]:
        reclean()
    elif sys.argv[1:] == ["--fuller"]:
        asyncio.run(main(fuller=True))
    else:
        asyncio.run(main(fuller=False))
