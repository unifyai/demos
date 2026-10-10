import json

from linked_libraries.arms import system_prompt, toolbox
from linked_libraries.library import (build_folders, build_linked, guidance_for, render_linked_context,
                                      render_long_context, rule_body, rules_md, save_linked, script_source, skill_listing,
                                      slug, split_params)
from linked_libraries.tools import BM25, FolderTools, LinkedTools, tokens


def test_split_params_and_script_compiles(ds_l):
    assert split_params("a: int, b: dict[str, int], c: list[str]") == ["a: int", "b: dict[str, int]", "c: list[str]"]
    for n, p in ds_l["procedures"].items():
        src = script_source(n, p)
        compile(src, n, "exec")
        assert f"def {n}(" in src and p["doc"].split()[0] in src


def test_pasted_copies_sit_in_exactly_the_governed_skills(ds_l, tmp_path):
    root = tmp_path / "lib"
    build_folders(ds_l, "pasted", root)
    for rid, r in ds_l["rules"].items():
        folders_governed = {p["folder"] for p in ds_l["procedures"].values() if rid in p["rules"]}
        holding = {f for f in ds_l["folders"] if rule_body(r) in (root / f / "SKILL.md").read_text()}
        assert holding == folders_governed, rid
    # the line naming the governed procedures lists exactly them
    text = (root / "billing" / "SKILL.md").read_text()
    assert "In this skill it governs `create_invoice`, `send_payment_reminder`, `issue_refund`, `issue_credit_note`." in text
    assert not (root / "RULES.md").exists()


def test_rule_skills_and_rules_file(ds_l, tmp_path):
    rs = tmp_path / "rs"
    build_folders(ds_l, "rule-skills", rs)
    names = dict(skill_listing(rs))
    for r in ds_l["rules"].values():
        assert slug(r["title"]) in names
        assert r["applies"] in names[slug(r["title"])]
        assert rule_body(r) in (rs / slug(r["title"]) / "SKILL.md").read_text()
    # no rule text inside the procedure skills
    for f in ds_l["folders"]:
        body = (rs / f / "SKILL.md").read_text()
        assert all(r["text"] not in body for r in ds_l["rules"].values())
    rf = tmp_path / "rf"
    build_folders(ds_l, "rules-file", rf)
    text = (rf / "RULES.md").read_text()
    assert text == rules_md(ds_l)
    assert all(rule_body(r) in text and text.count(r["text"]) == 1 for r in ds_l["rules"].values())


def test_linked_library_links_both_ways(ds_l):
    lib = build_linked(ds_l)
    assert set(lib["functions"]) == set(ds_l["procedures"])
    for n, p in ds_l["procedures"].items():
        got = sorted(g["title"] for _, g in guidance_for(lib, n))
        assert got == sorted(ds_l["rules"][r]["title"] for r in p["rules"])
    assert len(lib["guidance"]) == len(ds_l["rules"])


def test_prompts_differ_only_in_the_store(ds_s, tmp_path):
    libs = {}
    for arm in ("pasted", "rule-skills", "rules-file"):
        libs[arm] = tmp_path / arm
        build_folders(ds_s, arm, libs[arm])
    libs["linked"] = tmp_path / "library.json"
    save_linked(build_linked(ds_s), libs["linked"])
    prompts = {a: system_prompt(a, ds_s, libs[a]) for a in libs}
    prompts["long-context"] = system_prompt("long-context", ds_s, libs["rules-file"])
    prompts["linked-in-context"] = system_prompt("linked-in-context", ds_s, libs["linked"])
    heads = {p.split("\n\n")[0] for p in prompts.values()}
    tails = {p[p.index("A message may give you"):] for p in prompts.values()}
    assert len(heads) == 1 and len(tails) == 1
    # the in-context arms hold every procedure and every rule
    for arm in ("long-context", "linked-in-context"):
        for n in ds_s["procedures"]:
            assert n in prompts[arm]
        for r in ds_s["rules"].values():
            assert r["text"] in prompts[arm]
    assert all(r["text"] in prompts["rules-file"] for r in ds_s["rules"].values())
    assert not any(r["text"] in prompts["pasted"] for r in ds_s["rules"].values())
    assert toolbox("long-context", libs["rules-file"], write=False) is None


def test_folder_tools(ds_s, tmp_path):
    root = tmp_path / "lib"
    build_folders(ds_s, "pasted", root)
    t = FolderTools(root, write=True)
    assert "SKILL.md" in t.call("list_files", {"path": "sales"})
    assert t.call("read_file", {"path": "sales/SKILL.md"}).startswith("---\nname: sales")
    assert t.call("read_file", {"path": "../etc/passwd"}).startswith("Error")
    hits = t.call("grep", {"pattern": "US dollars"})
    assert "sales/SKILL.md" in hits and "procurement/SKILL.md" in hits
    assert t.call("grep", {"pattern": "("}).startswith("Error: invalid regular expression")
    rule = ds_s["rules"]["R04"]["text"]
    assert "Error" in t.call("edit_file", {"path": "sales/SKILL.md", "old_text": "nope", "new_text": "x"})
    out = t.call("edit_file", {"path": "sales/SKILL.md", "old_text": rule, "new_text": rule.replace("US dollars", "euros")})
    assert out.startswith("Edited") and t.writes == [{"op": "edit", "path": "sales/SKILL.md", "replaced": 1}]
    assert "euros" in (root / "sales" / "SKILL.md").read_text()
    t.call("write_file", {"path": "notes/x.md", "content": "hi"})
    t.call("delete_file", {"path": "notes/x.md"})
    assert not (root / "notes").exists()
    assert [w["op"] for w in t.writes] == ["edit", "create", "delete"]
    ro = FolderTools(root, write=False)
    assert "edit_file" not in ro.names() and ro.call("edit_file", {}).startswith("Error")


def test_bm25_and_linked_tools(ds_l, tmp_path):
    assert tokens("Scheduling meetings") == tokens("schedule meeting")
    idx = BM25({"a": "create an invoice for a customer", "b": "book a flight", "c": "refund an invoice"})
    assert idx.search("invoice customer", 3)[0][0] == "a"
    path = tmp_path / "library.json"
    save_linked(build_linked(ds_l), path)
    t = LinkedTools(path, write=True)
    res = json.loads(t.call("search_functions", {"query": "invoice a customer for hours of work"}))
    assert res[0]["name"] == "create_invoice"
    assert any(g["title"] == "Stating amounts of money" for g in res[0]["guidance"])
    f = json.loads(t.call("get_function", {"name": "issue_refund"}))
    assert {g["title"] for g in f["guidance"]} == {"Stating amounts of money", "Refunds and credits"}
    assert ds_l["rules"]["R09"]["text"] in json.dumps(f, ensure_ascii=False)
    g = json.loads(t.call("search_guidance", {"query": "refund"}))
    assert g[0]["title"] == "Refunds and credits" and "issue_refund" in g[0]["functions"]
    gid = g[0]["id"]
    # update content only: links kept
    t.call("update_guidance", {"guidance_id": gid, "content": "Applies to refunds.\n\nNEW"})
    lib = json.loads(path.read_text())
    assert lib["guidance"][str(gid)]["content"].endswith("NEW")
    assert lib["guidance"][str(gid)]["functions"] == g[0]["functions"]
    # functions given: replaces the links; unknown names are refused
    assert t.call("update_guidance", {"guidance_id": gid, "functions": ["nope"]}).startswith("Error: no function")
    t.call("update_guidance", {"guidance_id": gid, "functions": ["issue_refund"]})
    assert json.loads(path.read_text())["guidance"][str(gid)]["functions"] == ["issue_refund"]
    added = json.loads(t.call("add_guidance", {"title": "T", "content": "C", "functions": ["create_invoice"]}))
    assert added["added"]["id"] == len(ds_l["rules"]) + 1
    assert "T" in [x["title"] for x in json.loads(t.call("get_function", {"name": "create_invoice"}))["guidance"]]
    assert t.call("delete_guidance", {"guidance_id": added["added"]["id"]}).startswith("Deleted")
    assert [w["op"] for w in t.writes] == ["update", "update", "add", "delete"]
    assert "add_guidance" not in LinkedTools(path, write=False).names()
    assert "[1]" in render_linked_context(json.loads(path.read_text()))


def test_long_context_render_groups_by_skill(ds_s, tmp_path):
    text = render_long_context(ds_s, rules_md(ds_s))
    for f in ds_s["folders"]:
        assert f"### {f}: " in text
    assert text.count("## RULES.md") == 1


def test_listings_fit_the_result_limit_at_every_size(tmp_path):
    from pathlib import Path
    from linked_libraries.tools import MAX_RESULT_CHARS
    root = Path(__file__).resolve().parents[1] / "data"
    for size in ("S", "M", "L"):
        ds = json.loads((root / f"{size}.json").read_text())
        path = tmp_path / f"{size}.json"
        save_linked(build_linked(ds), path)
        t = LinkedTools(path, write=False)
        assert len(t.list_functions()) < MAX_RESULT_CHARS and len(t.list_guidance()) < MAX_RESULT_CHARS
        assert "more characters not shown" not in t.call("list_functions", {})
