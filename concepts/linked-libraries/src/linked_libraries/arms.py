"""The arms: one shared prompt, a paragraph per arm on how the library is kept,
and the tools (or the whole library, for the arms that read everything at once).

Change arms build and change a library; task arms read one. long-context reads
the library rules-file changed, and linked-in-context the one linked changed.
"""

from __future__ import annotations

from pathlib import Path

from .library import (load_linked, render_linked_context, render_long_context, skill_listing)
from .tools import FolderTools, LinkedTools, Toolbox

CHANGE_ARMS = ("pasted", "rule-skills", "rules-file", "linked")
TASK_ARMS = ("pasted", "rule-skills", "rules-file", "long-context", "linked", "linked-in-context")
SOURCE = {"pasted": "pasted", "rule-skills": "rule-skills", "rules-file": "rules-file",
          "long-context": "rules-file", "linked": "linked", "linked-in-context": "linked"}

SYSTEM = """You are the assistant of Fernhill, a company of about 80 people that sells logistics planning software. You work from a library that holds the procedures you have learned and the rules for how work is done here.

{store}

A message may give you a task, tell you about a lasting change to how things are done, or both.
- For a task, do not carry it out. Find the one procedure in the library that the task should call, and every rule in the library that applies to it.
- For a lasting change, make the library reflect it before you reply.

When you are done, reply with JSON only, in this form:
{{"function": "<the name of the procedure's function, or null if there is no task>", "rules": [{{"title": "<the rule's title as the library gives it>", "text": "<the rule as the library states it>"}}], "library_changed": "<one line on what you changed in the library, or an empty string>"}}"""

STORE = {
    "pasted": "The library is a folder of skills in the Agent Skills format. Each skill is a folder holding a SKILL.md (front matter with the skill's name and description, then its instructions, the list of its scripts and the rules that govern them) and a scripts/ folder with one Python file per procedure. A rule that governs procedures in several skills is copied into the SKILL.md of each of those skills, and the copies must be kept identical. Use the file tools to read the library.",
    "rule-skills": "The library is a folder of skills in the Agent Skills format. Each skill is a folder holding a SKILL.md (front matter with the skill's name and description, then its instructions and the list of its scripts) and a scripts/ folder with one Python file per procedure. Each rule is a skill of its own: a folder holding only a SKILL.md that states the rule, whose description says what it applies to. Use the file tools to read the library.",
    "rules-file": "The library is a folder of skills in the Agent Skills format, and one rules file. Each skill is a folder holding a SKILL.md (front matter with the skill's name and description, then its instructions and the list of its scripts) and a scripts/ folder with one Python file per procedure. Every rule is kept once, in RULES.md at the root of the library, which is shown in full below; each rule says what it applies to. Use the file tools to read the library.",
    "long-context": "The whole library is below. Procedures are grouped by skill, each with its signature and docstring. Every rule is kept once, in RULES.md, shown in full after them; each rule says what it applies to.",
    "linked": "The library has two parts that you search side by side. Functions holds the procedures, each with a name, a signature, a docstring and its code. Guidance holds the rules and walkthroughs, each an entry with an id, a title and a text, linked to the functions it governs. Links go both ways: a function shows the guidance linked to it, and a guidance entry lists the functions it governs. A rule is kept in one entry, linked to every function it governs.",
    "linked-in-context": "The whole library is below, in two parts. Functions: every procedure with its signature, its docstring and the guidance linked to it. Guidance: every rule and walkthrough with its id, title and text and the functions it governs. Links go both ways, and a rule is kept in one entry, linked to every function it governs.",
}


def library_path(lib_root: Path, arm: str) -> Path:
    """Where a change arm's library lives inside a run's library folder."""
    return lib_root / "library.json" if arm == "linked" else lib_root / "files"


def system_prompt(arm: str, ds: dict, lib: Path) -> str:
    """The prompt for an arm reading (or changing) the library at lib (library_path of its source arm)."""
    store = STORE[arm]
    if arm in ("pasted", "rule-skills", "rules-file"):
        listing = "\n".join(f"- {n}: {d}" for n, d in skill_listing(lib))
        store += f"\n\nSkills in the library (name: description):\n{listing}"
        if arm == "rules-file":
            rules = (lib / "RULES.md").read_text() if (lib / "RULES.md").exists() else "(RULES.md does not exist)"
            store += f"\n\nRULES.md:\n\n{rules.strip()}"
    elif arm == "long-context":
        rules = (lib / "RULES.md").read_text() if (lib / "RULES.md").exists() else "(RULES.md does not exist)"
        store += "\n\n" + render_long_context(ds, rules)
    elif arm == "linked-in-context":
        store += "\n\n" + render_linked_context(load_linked(lib))
    return SYSTEM.format(store=store)


def toolbox(arm: str, lib: Path, *, write: bool) -> Toolbox | None:
    if arm in ("pasted", "rule-skills", "rules-file"):
        return FolderTools(lib, write=write)
    if arm == "linked":
        return LinkedTools(lib, write=write)
    return None  # the in-context arms read everything at once
