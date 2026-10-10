import json
from pathlib import Path

from linked_libraries import catalog
from linked_libraries.generate import SIZES, build

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_is_consistent():
    catalog.check()
    assert len(catalog.FOLDERS) == 30 and len(catalog.PROCEDURES) >= 190 and len(catalog.RULES) == 28
    # every change names a distinct old and new fact and both requests
    for r in catalog.RULES.values():
        ch = r["change"]
        assert ch["old"] and ch["new"] and ch["old"] != ch["new"] and ch["direct"] and ch["passing"]


def test_generator_is_deterministic_and_matches_saved_data():
    for size in SIZES:
        a, b = build(size), build(size)
        assert a == b
        saved = ROOT / "data" / f"{size}.json"
        if saved.exists():
            assert json.loads(saved.read_text()) == a, f"data/{size}.json is out of date"


def test_sizes_and_links(ds_l, ds_s):
    for ds, cfg in ((ds_l, SIZES["L"]), (ds_s, SIZES["S"])):
        assert len(ds["folders"]) == cfg["folders"] and len(ds["rules"]) == cfg["rules"]
        assert len(ds["changes"]) == cfg["changes"] and len(ds["tasks"]) == cfg["tasks"]
        # links only to chosen rules, procedures only from chosen folders
        for n, p in ds["procedures"].items():
            assert p["folder"] in ds["folders"]
            assert set(p["rules"]) <= set(ds["rules"])
            assert p["rules"] == [r for r in next(x for x in catalog.PROCEDURES if x["name"] == n)["rules"] if r in ds["rules"]]
        # the changed rules are distinct and each governs a procedure in the size
        changed = [c["rule"] for c in ds["changes"]]
        assert len(set(changed)) == len(changed)
        for c in ds["changes"]:
            assert c["rule"] in ds["procedures"][c["passing_procedure"]]["rules"]
            assert c["passing_message"].startswith(ds["procedures"][c["passing_procedure"]]["task"])
        # each task's expected rules are its procedure's links
        for t in ds["tasks"]:
            assert t["rules"] == ds["procedures"][t["procedure"]]["rules"]
            assert t["changed_rules"] == [r for r in t["rules"] if r in changed]
        assert len({t["procedure"] for t in ds["tasks"]}) == len(ds["tasks"])


def test_realistic_size_has_the_conditions_the_idea_is_for(ds_l):
    spans = {}
    for p in ds_l["procedures"].values():
        for r in p["rules"]:
            spans.setdefault(r, set()).add(p["folder"])
    multi = [r for r in ds_l["rules"] if len(spans[r]) >= 2]
    assert len(multi) >= 0.8 * len(ds_l["rules"])  # most rules span several folders
    assert all(len(spans[c["rule"]]) >= 2 for c in ds_l["changes"])  # every changed rule does
    assert sum(t["hidden"] for t in ds_l["tasks"]) >= 12  # tasks for procedures their folder does not mention
    assert sum(bool(t["changed_rules"]) for t in ds_l["tasks"]) == 36


def test_task_subset_keeps_the_make_up(ds_l):
    from linked_libraries.generate import task_subset
    sub = task_subset(ds_l, 36)
    assert len(sub) == len(set(sub)) == 36 and sub == task_subset(ds_l, 36)
    by = {t["id"]: t for t in ds_l["tasks"]}
    assert sum(bool(by[i]["changed_rules"]) for i in sub) == 18  # half, as in the full set
    assert abs(sum(by[i]["hidden"] for i in sub) - 36 * 29 / 72) <= 1.5
