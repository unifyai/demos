"""The four ways of keeping the same knowledge, built from a dataset.

Folder libraries (Agent Skills format) are directories:

    <skill>/SKILL.md              front matter (name, description), then the
                                  instructions and the list of scripts
    <skill>/scripts/<name>.py     one procedure per file

  pasted       each rule copied into the SKILL.md of every skill whose
               procedures it governs, with a line naming those procedures
  rule-skills  each rule a skill of its own (<slug>/SKILL.md)
  rules-file   every rule once, in RULES.md at the root

The linked library is one JSON file with two tables, functions and guidance;
each guidance entry lists the functions it governs, and a function's guidance
is found by following those lists backwards.
"""

from __future__ import annotations

import json
import re
import shutil
import textwrap
from pathlib import Path

FOLDER_ARMS = ("pasted", "rule-skills", "rules-file")


# ---------------------------------------------------------------------------
# Shared rendering
# ---------------------------------------------------------------------------

def split_params(params: str) -> list[str]:
    """Split a parameter list at top-level commas: 'a: int, b: dict[str, int]'."""
    out, depth, cur = [], 0, ""
    for ch in params:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def signature(name: str, p: dict) -> str:
    return f"{name}({p['params']}) -> {p['returns']}"


def script_source(name: str, p: dict) -> str:
    args = [a.split(":")[0].strip() for a in split_params(p["params"])]
    call = f"primitives.{p['folder']}.{name}({', '.join(f'{a}={a}' for a in args)})"
    doc = textwrap.fill(p["doc"], width=84, initial_indent="    ", subsequent_indent="    ").strip()
    body = f"return {call}" if p["returns"] != "None" else call
    return f'def {signature(name, p)}:\n    """{doc}"""\n    {body}\n'


def rule_body(rule: dict) -> str:
    """A rule as every arm states it: what it applies to, then the rule."""
    return f"Applies to {rule['applies']}.\n\n{rule['text']}"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def rule_skill_description(rule: dict) -> str:
    t = rule["title"]
    return f"Rule on {t[0].lower() + t[1:]}. Applies to {rule['applies']}."


def front_matter(name: str, description: str) -> str:
    return f"---\nname: {name}\ndescription: {' '.join(description.split())}\n---\n"


def parse_front_matter(text: str) -> dict[str, str]:
    m = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    out: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                out[k.strip()] = v.strip()
    return out


def procs_in(ds: dict, folder: str) -> list[str]:
    return [n for n, p in ds["procedures"].items() if p["folder"] == folder]


def governed(ds: dict, rid: str) -> list[str]:
    return [n for n, p in ds["procedures"].items() if rid in p["rules"]]


def skill_md(ds: dict, folder: str, *, paste_rules: bool) -> str:
    f = ds["folders"][folder]
    names = procs_in(ds, folder)
    lines = [front_matter(folder, f["description"]), f"# {f['title']}", "",
             "Each script in `scripts/` defines one procedure; import it and call it.", "",
             "## Scripts", ""]
    for n in names:
        p = ds["procedures"][n]
        lines.append(f"- `scripts/{n}.py`: `{signature(n, p)}`. {p['doc']}")
    if paste_rules:
        rids = [r for r in ds["rules"] if any(r in ds["procedures"][n]["rules"] for n in names)]
        if rids:
            lines += ["", "## Rules", ""]
            for r in rids:
                here = ", ".join(f"`{n}`" for n in names if r in ds["procedures"][n]["rules"])
                lines += [f"### {ds['rules'][r]['title']}", "", f"In this skill it governs {here}.", "",
                          rule_body(ds["rules"][r]), ""]
    return "\n".join(lines).rstrip() + "\n"


def rule_skill_md(ds: dict, rid: str) -> str:
    r = ds["rules"][rid]
    return f"{front_matter(slug(r['title']), rule_skill_description(r))}# {r['title']}\n\n{rule_body(r)}\n"


def rules_md(ds: dict) -> str:
    parts = ["# Rules", "", "Every rule for how work is done at Fernhill, each kept once here. "
             "Each rule says what it applies to.", ""]
    for rid, r in ds["rules"].items():
        parts += [f"## {r['title']}", "", rule_body(r), ""]
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Folder libraries
# ---------------------------------------------------------------------------

def build_folders(ds: dict, arm: str, root: Path) -> None:
    assert arm in FOLDER_ARMS, arm
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for folder in ds["folders"]:
        d = root / folder
        (d / "scripts").mkdir(parents=True)
        (d / "SKILL.md").write_text(skill_md(ds, folder, paste_rules=(arm == "pasted")))
        for n in procs_in(ds, folder):
            (d / "scripts" / f"{n}.py").write_text(script_source(n, ds["procedures"][n]))
    if arm == "rule-skills":
        for rid in ds["rules"]:
            d = root / slug(ds["rules"][rid]["title"])
            assert not d.exists(), d
            d.mkdir()
            (d / "SKILL.md").write_text(rule_skill_md(ds, rid))
    if arm == "rules-file":
        (root / "RULES.md").write_text(rules_md(ds))


def skill_listing(root: Path) -> list[tuple[str, str]]:
    """(name, description) of every skill, as Agent Skills shows them."""
    out = []
    for p in sorted(root.glob("*/SKILL.md")):
        fm = parse_front_matter(p.read_text())
        out.append((fm.get("name", p.parent.name), fm.get("description", "")))
    return out


# ---------------------------------------------------------------------------
# Linked library
# ---------------------------------------------------------------------------

def build_linked(ds: dict) -> dict:
    functions = {}
    for n, p in ds["procedures"].items():
        functions[n] = {"signature": signature(n, p), "docstring": p["doc"], "code": script_source(n, p)}
    guidance = {}
    for i, (rid, r) in enumerate(ds["rules"].items(), 1):
        guidance[str(i)] = {"title": r["title"], "content": rule_body(r), "functions": governed(ds, rid)}
    return {"functions": functions, "guidance": guidance, "next_id": len(guidance) + 1}


def save_linked(lib: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lib, indent=1, ensure_ascii=False) + "\n")


def load_linked(path: Path) -> dict:
    return json.loads(path.read_text())


def guidance_for(lib: dict, name: str) -> list[tuple[str, dict]]:
    """The reverse link: every guidance entry that lists this function."""
    return [(gid, g) for gid, g in lib["guidance"].items() if name in g["functions"]]


# ---------------------------------------------------------------------------
# Everything in one context
# ---------------------------------------------------------------------------

def render_long_context(ds: dict, rules_file_text: str) -> str:
    """Every procedure's signature and docstring, by skill, and the whole RULES.md."""
    parts = ["## Procedures, by skill", ""]
    for folder, f in ds["folders"].items():
        parts += [f"### {folder}: {f['description']}", ""]
        for n in procs_in(ds, folder):
            p = ds["procedures"][n]
            parts.append(f"- `{signature(n, p)}`: {p['doc']}")
        parts.append("")
    parts += ["## RULES.md", "", rules_file_text.strip(), ""]
    return "\n".join(parts)


def render_linked_context(lib: dict) -> str:
    """The whole linked library: functions with their linked guidance, then guidance with its functions."""
    parts = ["## Functions", ""]
    for n in sorted(lib["functions"]):
        f = lib["functions"][n]
        gl = guidance_for(lib, n)
        linked = "; ".join(f"[{gid}] {g['title']}" for gid, g in gl) or "none"
        parts.append(f"- `{f['signature']}`: {f['docstring']} Guidance: {linked}.")
    parts += ["", "## Guidance", ""]
    for gid, g in lib["guidance"].items():
        parts += [f"### [{gid}] {g['title']}", f"Governs: {', '.join(g['functions']) or 'no functions'}.", "",
                  g["content"].strip(), ""]
    return "\n".join(parts)
