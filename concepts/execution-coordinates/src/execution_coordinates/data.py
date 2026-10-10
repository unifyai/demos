"""Generate the clients, their exports, their requests and the true answers.

    python -m execution_coordinates.data --cond main --seeds 1-5
    python -m execution_coordinates.data --cond sanity --seeds 1-2

Per seed, eight fictional clients, each with a payments export (all text, in
one schema, formatted in the client's locale) and churn-risk scores. Each
client sends six requests that build on each other (or three in the sanity
condition). True answers come from the generator's own numbers and are checked
against an independent parse of the written CSV. For each request the
generator also computes the answers that leftover or clobbered state would
give (another client's data, rates or scores; the client's own what-if left in
place): `signatures`, used only to attribute wrong answers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[2]
DATA = HERE / "data"
MONTHS = ["January", "February", "March", "April"]
TO_EUR = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17, "CHF": 1.04, "SEK": 0.087, "CAD": 0.68}
COMPANIES = [
    "Brightwater Bikes", "Halden Dental", "Marigold Tea Co", "Northgate Freight", "Ostara Studios",
    "Pinecrest Outfitters", "Quillon Labs", "Redfern Bakery", "Saltmarsh Ceramics", "Tidewell Energy",
    "Umbra Optics", "Vantage Pet Supply", "Willowmere Hotels", "Yarrow Health", "Larkspur Books", "Kestrel Marine",
]
PEOPLE = ["Ana", "Ben", "Chloe", "Dev", "Elif", "Farid", "Greta", "Hiro", "Ines", "Jonas", "Kemi", "Luca",
          "Mei", "Nils", "Oona", "Priya"]
REGIONS = ["EMEA", "Americas", "APAC", "Nordics", "DACH", "Benelux", "Iberia", "UK & Ireland", "France", "Italy",
           "CEE", "Middle East"]
PREFIXES = ["TEST-", "QA-", "INT-", "SANDBOX-", "DEMO-", "STAFF-"]
# locale: (amount style, date format, how the request describes the date order)
LOCALES = {
    "us": ("comma_point", "%m/%d/%Y %H:%M", "month first"),
    "uk": ("comma_point", "%d/%m/%Y %H:%M", "day first"),
    "de": ("point_comma", "%d.%m.%Y %H:%M", "day first"),
    "fr": ("space_comma", "%d/%m/%Y %H:%M", "day first"),
    "ch": ("apos_point", "%d.%m.%Y %H:%M", "day first"),
    "iso": ("plain_point", "%Y-%m-%d %H:%M:%S", "year first"),
}
WHATIFS = ["top_customers", "rate", "pending"]
# A robustness check added after the plan (see README): clients in pairs that
# share an initial and a prefix, as real client lists do, so that abbreviated
# session names can collide. Everything else is generated as in "main".
SHARED_INITIALS = ["Saltmarsh Ceramics", "Saltash Bakery", "Halden Dental", "Halford Optics",
                   "Pinecrest Outfitters", "Pinewood Pet Supply", "Marigold Tea Co", "Marlow Energy"]
CONDITIONS = {
    # rows per export, load and scoring seconds, requests per client
    "main": {"n_rows": 100_000, "n_customers": 3_000, "load_s": 20.0, "score_s": 20.0, "requests": 6},
    "sanity": {"n_rows": 2_000, "n_customers": 300, "load_s": 1.0, "score_s": 1.0, "requests": 3},
    "initials": {"n_rows": 100_000, "n_customers": 3_000, "load_s": 20.0, "score_s": 20.0, "requests": 6,
                 "companies": SHARED_INITIALS},
}
KS = (1, 2, 4, 8)
N_CLIENTS = 8


def fmt_amount(x: float, style: str) -> str:
    s = f"{x:,.2f}"  # 1,234.50
    if style == "comma_point":
        return s
    if style == "point_comma":
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    if style == "space_comma":
        return s.replace(",", " ").replace(".", ",")
    if style == "apos_point":
        return s.replace(",", "'")
    if style == "plain_point":
        return s.replace(",", "")
    raise ValueError(style)


def slug(name: str) -> str:
    return name.lower().split()[0]


@dataclass
class Client:
    idx: int
    id: str
    name: str
    person: str
    locale: str
    rep: str
    currencies: list[str]
    rates: dict[str, float]
    prefix: str
    regions: list[str]


def make_clients(seed: int, companies: list[str] | None = None) -> list[Client]:
    rng = random.Random(f"clients-{seed}")
    names = rng.sample(companies or COMPANIES, N_CLIENTS)
    people = rng.sample(PEOPLE, N_CLIENTS)
    locales = list(LOCALES)
    rng.shuffle(locales)
    out = []
    for i, name in enumerate(names):
        rep = rng.choices(["EUR", "USD", "GBP", "CHF", "SEK", "CAD"], [4, 4, 2, 1, 1, 1])[0]
        others = rng.sample([c for c in TO_EUR if c != rep], rng.choice([1, 2]))
        currencies = [rep] + others
        rates = {rep: 1.0}
        for c in others:
            rates[c] = round(TO_EUR[c] / TO_EUR[rep] * (1 + rng.uniform(-0.03, 0.03)), 4)
        out.append(Client(
            idx=i, id=slug(name), name=name, person=people[i], locale=locales[i % len(locales)],
            rep=rep, currencies=currencies, rates=rates, prefix=rng.choice(PREFIXES),
            regions=rng.sample(REGIONS, rng.choice([3, 4, 5])),
        ))
    return out


def make_frame(client: Client, seed: int, n_rows: int, n_customers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The client's transactions as numbers (frame) and its risk scores."""
    rng = np.random.default_rng([seed, client.idx, 7])
    cust_ids = np.array([f"CUS-{i:05d}" for i in range(1, n_customers + 1)])
    test_ids = np.array([f"{client.prefix}{i:03d}" for i in range(1, 21)])
    region_w = rng.dirichlet(np.ones(len(client.regions)) * 3)
    cur_w = np.array([0.6] + [0.4 / (len(client.currencies) - 1)] * (len(client.currencies) - 1))
    cust_region = rng.choice(client.regions, size=n_customers, p=region_w)
    cust_cur = rng.choice(client.currencies, size=n_customers, p=cur_w)
    test_region = rng.choice(client.regions, size=20)
    test_cur = rng.choice(client.currencies, size=20)
    size = rng.lognormal(0, 1.2, size=n_customers)
    p = size / size.sum()

    is_test = rng.random(n_rows) < 0.02
    ci = rng.choice(n_customers, size=n_rows, p=p)
    ti = rng.integers(0, 20, size=n_rows)
    customer = np.where(is_test, test_ids[ti], cust_ids[ci])
    region = np.where(is_test, test_region[ti], cust_region[ci])
    currency = np.where(is_test, test_cur[ti], cust_cur[ci])
    start = np.datetime64("2025-01-01T00:00")
    minutes = rng.integers(0, int((np.datetime64("2025-05-01T00:00") - start) / np.timedelta64(1, "m")), size=n_rows)
    created = start + minutes.astype("timedelta64[m]")
    typ = np.where(rng.random(n_rows) < 0.08, "refund", "charge")
    method = rng.choice(["card", "bank_transfer", "wallet"], size=n_rows, p=[0.6, 0.3, 0.1])
    u = rng.random(n_rows)
    status = np.where(typ == "charge", np.where(u < 0.85, "settled", np.where(u < 0.93, "failed", "pending")),
                      np.where(u < 0.9, "settled", "pending"))
    amount = np.round(np.minimum(rng.lognormal(4.4, 1.1, size=n_rows), 25_000.0) + 0.5, 2)
    order = np.argsort(created, kind="stable")
    frame = pd.DataFrame({
        "txn_id": [f"T{i:07d}" for i in range(1, n_rows + 1)],
        "created": created[order],
        "customer_id": customer[order],
        "region": region[order],
        "type": typ[order],
        "method": method[order],
        "status": status[order],
        "currency": currency[order],
        "amount": amount[order],
    })
    frame["month"] = pd.DatetimeIndex(frame["created"]).month.map(lambda m: MONTHS[m - 1])
    risk = np.round(rng.beta(2.5, 4.0, size=n_customers + 20), 3)
    scores = pd.DataFrame({"customer_id": np.concatenate([cust_ids, test_ids]), "risk": risk})
    return frame, scores


def to_text(frame: pd.DataFrame, client: Client) -> pd.DataFrame:
    style, datefmt, _ = LOCALES[client.locale]
    return pd.DataFrame({
        "txn_id": frame["txn_id"],
        "created_at": pd.DatetimeIndex(frame["created"]).strftime(datefmt),
        "customer_id": frame["customer_id"],
        "region": frame["region"],
        "type": frame["type"],
        "method": frame["method"],
        "status": frame["status"],
        "currency": frame["currency"],
        "amount": [fmt_amount(x, style) for x in frame["amount"]],
    })


# ───────────────────────────── the questions ─────────────────────────────────
def _mask(frame, prefix, month, statuses=("settled",), customers=None, exclude=None):
    m = (frame["month"] == month) & ~frame["customer_id"].str.startswith(prefix) & frame["status"].isin(statuses)
    if customers is not None:
        m &= frame["customer_id"].isin(customers)
    if exclude is not None:
        m &= ~frame["customer_id"].isin(exclude)
    return m


def _signed(frame, rates):
    conv = frame["amount"] * frame["currency"].map(rates)
    return conv * np.where(frame["type"] == "charge", 1.0, -1.0)


def net_revenue(frame, rates, prefix, month, **kw) -> float:
    return round(float(_signed(frame, rates)[_mask(frame, prefix, month, **kw)].sum()), 2)


def net_revenue_by_region(frame, rates, prefix, month, **kw) -> dict[str, float]:
    m = _mask(frame, prefix, month, **kw)
    g = _signed(frame, rates)[m].groupby(frame["region"][m]).sum()
    return {str(k): round(float(v), 2) for k, v in g.items()}


def top_customers(frame, rates, prefix, month, n) -> list[str]:
    m = _mask(frame, prefix, month) & (frame["type"] == "charge")
    conv = (frame["amount"] * frame["currency"].map(rates))[m]
    return list(conv.groupby(frame["customer_id"][m]).sum().nlargest(n).index)


def avg_card_charge(frame, rates, prefix, month, statuses=("settled",), exclude=None) -> float:
    m = _mask(frame, prefix, month, statuses=statuses, exclude=exclude) & (frame["type"] == "charge") & (
        frame["method"] == "card")
    return round(float((frame["amount"] * frame["currency"].map(rates))[m].mean()), 2)


def high_risk(scores, prefix, thr) -> list[str]:
    s = scores[~scores["customer_id"].str.startswith(prefix)]
    return list(s.loc[s["risk"] > thr, "customer_id"])


def make_requests(client: Client, frame: pd.DataFrame, scores: pd.DataFrame, seed: int,
                  n_requests: int, load_s: float = 20.0, score_s: float = 20.0) -> tuple[list[dict], dict]:
    rng = random.Random(f"requests-{seed}-{client.idx}")
    load_t = f"about {load_s:g} second{'s' if load_s != 1 else ''}"
    score_t = f"about {score_s:g} second{'s' if score_s != 1 else ''}"
    c = client
    style, _, order_note = LOCALES[c.locale]
    datefmt = LOCALES[c.locale][1]
    example_amount = fmt_amount(1234.5, style)
    example_date = pd.Timestamp("2025-03-14 09:05").strftime(datefmt)
    m1, m2, m3 = rng.choice(MONTHS), rng.choice(MONTHS), rng.choice(MONTHS)
    rates_text = " and ".join(f"1 {cur} = {c.rates[cur]} {c.rep}" for cur in c.currencies if cur != c.rep)
    head = f"[{c.name} · {c.person}]"
    kind = WHATIFS[(seed + c.idx) % 3]
    thr = rng.choice([0.7, 0.75, 0.8])
    # no customer sits exactly on the threshold
    scores.loc[(scores["risk"] - thr).abs() < 1e-9, "risk"] = round(thr + 0.001, 3)

    reqs: list[dict] = []
    reqs.append({
        "id": "r1_net_revenue", "answer_type": "number",
        "text": (f"{head} Hi! Could you pull our payments export with load_export(\"{c.id}\")? It takes {load_t}. "
                 f"Everything in it is text: amounts are written like {example_amount} and timestamps "
                 f"like {example_date} ({order_note}). What was our net revenue for {m1} 2025 in {c.rep}? Net "
                 f"revenue is settled charges minus settled refunds, for transactions created in {m1}, leaving out "
                 f"our internal test accounts (customer ids starting with \"{c.prefix}\"). Convert at {rates_text}. "
                 f"Round to 2 decimals and answer with the number."),
        "truth": net_revenue(frame, c.rates, c.prefix, m1),
    })
    reqs.append({
        "id": "r2_by_region", "answer_type": "regions",
        "text": (f"{head} Thanks! Could you split that {m1} net revenue by region? Same rules and rates. Answer "
                 f"with a JSON object mapping each region to its net revenue, rounded to 2 decimals."),
        "truth": net_revenue_by_region(frame, c.rates, c.prefix, m1),
    })
    whatif_state = {}
    if n_requests >= 6:
        if kind == "top_customers":
            n = rng.choice([3, 5, 10])
            top = top_customers(frame, c.rates, c.prefix, m1, n)
            text = (f"{head} Out of curiosity, what would {m1}'s net revenue have been without our {n} biggest "
                    f"customers, by their settled charges in {m1} (in {c.rep})? Leave out all of their "
                    f"transactions. Just a what-if. Answer with the number, rounded to 2 decimals.")
            truth = net_revenue(frame, c.rates, c.prefix, m1, exclude=top)
            whatif_state = {"exclude": top}
        elif kind == "rate":
            cur = rng.choice([x for x in c.currencies if x != c.rep])
            factor = rng.choice([rng.uniform(1.05, 1.12), rng.uniform(0.88, 0.95)])
            alt = round(c.rates[cur] * factor, 4)
            text = (f"{head} Quick what-if: if {cur} had been at {alt} {c.rep} per {cur} instead, what would "
                    f"{m1}'s net revenue have been? Everything else the same. Answer with the number, rounded to "
                    f"2 decimals.")
            truth = net_revenue(frame, {**c.rates, cur: alt}, c.prefix, m1)
            whatif_state = {"rates": {**c.rates, cur: alt}}
        else:
            text = (f"{head} What would {m1}'s net revenue have been if every charge and refund still pending "
                    f"had settled? Just a what-if. Answer with the number, rounded to 2 decimals.")
            truth = net_revenue(frame, c.rates, c.prefix, m1, statuses=("settled", "pending"))
            whatif_state = {"statuses": ("settled", "pending")}
        reqs.append({"id": f"r3_whatif_{kind}", "answer_type": "number", "text": text, "truth": truth,
                     "whatif": kind})
    reqs.append({
        "id": "r4_avg_card", "answer_type": "number",
        "text": (f"{head} Back to the actual numbers: what was the average settled card charge created in {m2} "
                 f"2025, in {c.rep} at the same rates, leaving out test accounts? Round to 2 decimals and answer "
                 f"with the number."),
        "truth": avg_card_charge(frame, c.rates, c.prefix, m2),
        "params": {"month": m2},
    })
    if n_requests >= 6:
        hr = high_risk(scores, c.prefix, thr)
        reqs.append({
            "id": "r5_risk_count", "answer_type": "count",
            "text": (f"{head} Next, please run our churn-risk model with score_customers(\"{c.id}\") (it takes "
                     f"{score_t}). How many of our customers, not counting test accounts, have a risk score "
                     f"above {thr}? Answer with the count."),
            "truth": len(hr), "params": {"thr": thr},
        })
        reqs.append({
            "id": "r6_risk_revenue", "answer_type": "number",
            "text": (f"{head} Last one: how much net revenue did those high-risk customers bring in during {m3} "
                     f"2025, in {c.rep} at the same rates? Round to 2 decimals and answer with the number."),
            "truth": net_revenue(frame, c.rates, c.prefix, m3, customers=hr),
            "params": {"month": m3, "thr": thr},
        })
    for r in reqs:
        r.setdefault("params", {})
        r["params"].setdefault("month", m1)
    return reqs, {"m1": m1, "m2": m2, "m3": m3, "thr": thr, "kind": kind, "whatif_state": whatif_state}


def signatures(c: Client, req: dict, p: dict, frames: dict, scores: dict, clients: list[Client]) -> list[dict]:
    """Answers that leftover or clobbered state would give for this request."""
    out: list[dict] = []
    rid = req["id"]
    fr = frames[c.id]

    def add(cause, how, fn):
        try:
            v = fn()
        except Exception:
            return
        if isinstance(v, float) and not math.isfinite(v):
            return
        out.append({"cause": cause, "how": how, "value": v})

    # the client's own what-if left in place (only requests after the what-if)
    ws = p["whatif_state"]
    if rid in ("r4_avg_card", "r6_risk_revenue") and ws:
        rates = ws.get("rates", c.rates)
        st = ws.get("statuses", ("settled",))
        ex = ws.get("exclude")
        if rid == "r4_avg_card":
            add("own_whatif", p["kind"], lambda: avg_card_charge(fr, rates, c.prefix, p["m2"], statuses=st, exclude=ex))
        else:
            hr = high_risk(scores[c.id], c.prefix, p["thr"])
            add("own_whatif", p["kind"],
                lambda: net_revenue(fr, rates, c.prefix, p["m3"], statuses=st, customers=hr, exclude=ex))
    # another client's data, rates or scores
    for d in clients:
        if d.id == c.id:
            continue
        fd = frames[d.id]
        tag = f"leak:{d.id}"
        if rid == "r1_net_revenue" or rid.startswith("r3_"):
            add(tag, "their data and rates", lambda: net_revenue(fd, d.rates, d.prefix, p["m1"]))
            add(tag, "their data, our rates", lambda: net_revenue(fd, c.rates, c.prefix, p["m1"]))
            add(tag, "our data, their rates", lambda: net_revenue(fr, d.rates, c.prefix, p["m1"]))
        elif rid == "r2_by_region":
            add(tag, "their data and rates", lambda: net_revenue_by_region(fd, d.rates, d.prefix, p["m1"]))
            add(tag, "their data, our rates", lambda: net_revenue_by_region(fd, c.rates, c.prefix, p["m1"]))
        elif rid == "r4_avg_card":
            add(tag, "their data and rates", lambda: avg_card_charge(fd, d.rates, d.prefix, p["m2"]))
            add(tag, "their data, our rates", lambda: avg_card_charge(fd, c.rates, c.prefix, p["m2"]))
            add(tag, "our data, their rates", lambda: avg_card_charge(fr, d.rates, c.prefix, p["m2"]))
        elif rid == "r5_risk_count":
            add(tag, "their scores", lambda: len(high_risk(scores[d.id], c.prefix, p["thr"])))
        elif rid == "r6_risk_revenue":
            hr_d = high_risk(scores[d.id], d.prefix, p["thr"])
            hr_c = high_risk(scores[c.id], c.prefix, p["thr"])
            add(tag, "our data, their scores", lambda: net_revenue(fr, c.rates, c.prefix, p["m3"], customers=hr_d))
            add(tag, "their data, our scores", lambda: net_revenue(fd, c.rates, c.prefix, p["m3"], customers=hr_c))
            add(tag, "their data and scores", lambda: net_revenue(fd, d.rates, d.prefix, p["m3"], customers=hr_d))
            add(tag, "our data, their rates", lambda: net_revenue(fr, d.rates, c.prefix, p["m3"], customers=hr_c))
    return out


def interleave(n_per_client: list[int], seed: int, k: int) -> list[list[int]]:
    """A uniformly random merge of the clients' request sequences."""
    rng = random.Random(f"order-{seed}-{k}")
    left = list(n_per_client)
    nxt = [0] * len(left)
    out = []
    while sum(left):
        i = rng.choices(range(len(left)), weights=left)[0]
        out.append([i, nxt[i]])
        nxt[i] += 1
        left[i] -= 1
    return out


# ───────────────────────── an independent reference parse ────────────────────
def reference_parse(text: pd.DataFrame, client: Client) -> pd.DataFrame:
    """Parse an export from its text alone, as an analyst would."""
    style, datefmt, _ = LOCALES[client.locale]
    a = text["amount"]
    if style == "comma_point":
        num = a.str.replace(",", "", regex=False)
    elif style == "point_comma":
        num = a.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif style == "space_comma":
        num = a.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    elif style == "apos_point":
        num = a.str.replace("'", "", regex=False)
    else:
        num = a
    out = text.copy()
    out["amount"] = num.astype(float)
    out["created"] = pd.to_datetime(text["created_at"], format=datefmt)
    out["month"] = out["created"].dt.month.map(lambda m: MONTHS[m - 1])
    return out


def generate(cond: str, seed: int, out_root: Path | None = None) -> dict:
    cfg = CONDITIONS[cond]
    out = (out_root or DATA) / cond / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    clients = make_clients(seed, cfg.get("companies"))
    frames, scores, texts = {}, {}, {}
    for c in clients:
        frames[c.id], scores[c.id] = make_frame(c, seed, cfg["n_rows"], cfg["n_customers"])
        texts[c.id] = to_text(frames[c.id], c)
    made = {c.id: make_requests(c, frames[c.id], scores[c.id], seed, cfg["requests"], cfg["load_s"], cfg["score_s"])
            for c in clients}
    spec_clients = []
    for c in clients:
        reqs, p = made[c.id]
        for r in reqs:
            r["signatures"] = signatures(c, r, p, frames, scores, clients)
        # the reference parse must reproduce every truth that depends only on the export
        ref = reference_parse(texts[c.id], c)
        for col in ["customer_id", "region", "type", "method", "status", "currency"]:
            assert (ref[col] == frames[c.id][col]).all(), (c.id, col)
        assert np.allclose(ref["amount"].to_numpy(), frames[c.id]["amount"].to_numpy())
        assert (ref["created"].to_numpy() == frames[c.id]["created"].to_numpy()).all()
        assert reqs[0]["truth"] == net_revenue(ref, c.rates, c.prefix, p["m1"])
        texts[c.id].to_csv(out / f"{c.id}.csv", index=False)
        scores[c.id].to_csv(out / f"{c.id}_risk.csv", index=False)
        spec_clients.append({
            "idx": c.idx, "id": c.id, "name": c.name, "person": c.person, "locale": c.locale, "rep": c.rep,
            "currencies": c.currencies, "rates": c.rates, "prefix": c.prefix, "regions": c.regions,
            "params": {k: v for k, v in p.items() if k != "whatif_state"}, "requests": reqs,
        })
    ks = list(KS)
    spec = {
        "cond": cond, "seed": seed, **{k: v for k, v in cfg.items() if k != "companies"},
        "clients": spec_clients,
        "orders": {str(k): interleave([cfg["requests"]] * k, seed, k) for k in ks},
    }
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    return spec


def load_spec(cond: str, seed: int, root: Path | None = None) -> dict:
    return json.loads(((root or DATA) / cond / f"seed{seed}" / "spec.json").read_text())


def parse_seeds(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="main", choices=sorted(CONDITIONS))
    ap.add_argument("--seeds", default="1-5")
    args = ap.parse_args()
    for s in parse_seeds(args.seeds):
        spec = generate(args.cond, s)
        print(f"{args.cond} seed {s}: {len(spec['clients'])} clients, "
              f"{sum(len(c['requests']) for c in spec['clients'])} requests")


if __name__ == "__main__":
    main()
