"""The analysis: derived measures, paired comparisons and the verdict rules, on synthetic records."""

import analyze


def rec(arm, item, size, seed, correct, cost, lat, looked=()):
    return {"model": "m", "arm": arm, "session": f"item{item:02d}-n{size}", "item": item, "size": size, "seed": seed,
            "correct": correct, "truth": ["A", "B"], "answer": ["A", "B"] if correct else ["B", "A"],
            "steps": 2, "looked": list(looked),
            "calls": [{"phase": "pre", "cost": 0.5 * cost, "latency_s": 9.0, "prompt_tokens": 10, "provider": "p"},
                      {"phase": "request", "cost": 0.5 * cost, "latency_s": lat, "prompt_tokens": 100, "provider": "p"}]}


def test_derive(monkeypatch):
    monkeypatch.setattr(analyze, "evidence_of", lambda s: ["f001", "f002"])
    d = analyze.derive(rec("by_reference", 1, 64, 0, False, 0.02, 3.0, looked=["f001", "f002", "f003"]))
    assert d["correct"] == 0 and d["set_correct"] == 1 and d["positions"] == 0
    assert d["cost"] == 0.02 and d["cost_pre"] == 0.01 and d["latency"] == 3.0 and d["pre_compute_s"] == 9.0
    assert d["looked"] == 3 and d["looked_evidence"] == 1 and abs(d["looked_share"] - 3 / 64) < 1e-12


def test_verdict_rules(monkeypatch):
    monkeypatch.setattr(analyze, "evidence_of", lambda s: [])
    rows = []
    for item in range(10):
        for seed in range(3):
            rows.append(analyze.derive(rec("by_reference", item, 500, seed, True, 0.01, 5.0)))
            rows.append(analyze.derive(rec("inline_all", item, 500, seed, item % 2 == 0, 0.30, 40.0)))
            rows.append(analyze.derive(rec("captions", item, 500, seed, True, 0.005, 2.0)))
    ref = [r for r in rows if r["arm"] == "by_reference"]
    cmps = {a: analyze.compare(ref, [r for r in rows if r["arm"] == a]) for a in ("inline_all", "captions")}
    assert abs(cmps["inline_all"]["acc_diff"] - 0.5) < 1e-9 and cmps["inline_all"]["acc_lo"] > 0
    assert cmps["inline_all"]["cost_ratio"] < 0.05
    v = analyze.verdict(cmps)
    assert v["holds_vs_inline_all"] and v["pulls_ahead_of_inline_all"]
    # captions is as accurate, cheaper and faster: by_reference does not win outright and loses
    assert not v["wins_outright"] and v["noninferior_cheaper_and_faster"] == ["captions"] and v["loses"]


def test_refused_requests_count_as_wrong_and_carry_no_cost(monkeypatch):
    monkeypatch.setattr(analyze, "evidence_of", lambda s: [])
    rows = []
    for item in range(6):
        for seed in range(3):
            rows.append(analyze.derive(rec("by_reference", item, 500, seed, item != 0, 0.05, 20.0)))
            refused = rec("inline_all", item, 500, seed, False, 0.0, 0.0)
            refused["limit"] = "LLMError: Too much media: 0 document pages + 500 images > 100"
            refused["calls"] = []
            refused["answer"] = None
            rows.append(analyze.derive(refused))
    inline = [r for r in rows if r["arm"] == "inline_all"]
    c = analyze.cell(inline)
    assert c["fits"] is False and c["correct"] == 0 and c["refused"] == 18 and c["error_kinds"] == {"over the limit": 18}
    ref = [r for r in rows if r["arm"] == "by_reference"]
    cmp = analyze.compare(ref, inline)
    assert cmp["other_fits"] is False and "cost_ratio" not in cmp and cmp["acc_diff"] > 0.8
    v = analyze.verdict({"inline_all": cmp})
    assert v["holds_vs_inline_all"] and v["inline_all_does_not_fit"]


def test_winning_outright_needs_holding_first(monkeypatch):
    """Found in the analysis of the cheap tier: an arm can be non-inferior to every other arm and
    cheaper than every rival while not answering faster than inline_all; it then does not hold, so
    it cannot win outright."""
    monkeypatch.setattr(analyze, "evidence_of", lambda s: [])
    rows = []
    for item in range(8):
        for seed in range(3):
            rows.append(analyze.derive(rec("by_reference", item, 500, seed, True, 0.01, 40.0 + item)))
            rows.append(analyze.derive(rec("inline_all", item, 500, seed, item % 2 == 0, 0.30, 40.0 + item)))
    ref = [r for r in rows if r["arm"] == "by_reference"]
    v = analyze.verdict({"inline_all": analyze.compare(ref, [r for r in rows if r["arm"] == "inline_all"])})
    assert not v["holds_vs_inline_all"] and not v["wins_outright"]
