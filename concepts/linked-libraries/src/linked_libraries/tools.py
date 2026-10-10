"""The tools agents use on each kind of library.

FolderTools are the file tools an agent with Agent Skills has (list, read,
grep; edit, write and delete when it may change the library). LinkedTools are
the two libraries' own operations, searched side by side: a function comes
back with the guidance linked to it, and a guidance entry with the functions
it governs. Every tool returns a string; a failure is returned as an
"Error: ..." string for the model to read, never raised.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import snowballstemmer

from .library import guidance_for, load_linked, save_linked

MAX_RESULT_CHARS = 20000
MAX_GREP = 100


def _fn(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


def _clip(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS] + f"\n... ({len(text) - MAX_RESULT_CHARS} more characters not shown)"


class ToolError(Exception):
    pass


class Toolbox:
    """Dispatch by name; record every call and every successful write."""

    def __init__(self) -> None:
        self.writes: list[dict] = []
        self._tools: dict[str, Callable[..., str]] = {}
        self._schemas: list[dict] = []

    def register(self, schema: dict, fn: Callable[..., str]) -> None:
        self._schemas.append(schema)
        self._tools[schema["function"]["name"]] = fn

    def schemas(self) -> list[dict]:
        return list(self._schemas)

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: Any) -> str:
        fn = self._tools.get(name)
        if fn is None:
            return f"Error: there is no tool named {name!r}."
        if not isinstance(args, dict):
            return "Error: the arguments must be a JSON object."
        try:
            return _clip(fn(**args))
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except ToolError as e:
            return f"Error: {e}"
        except Exception as e:  # noqa: BLE001 - every failure goes back to the model as text
            return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Folder tools
# ---------------------------------------------------------------------------

class FolderTools(Toolbox):
    def __init__(self, root: Path, *, write: bool) -> None:
        super().__init__()
        self.root = root.resolve()
        self.register(_fn("list_files", "List the files and folders directly inside a folder of the library. Use '.' for the root.",
                          {"path": {"type": "string", "description": "Folder path relative to the library root, such as '.' or 'billing/scripts'."}},
                          ["path"]), self.list_files)
        self.register(_fn("read_file", "Read a file of the library, such as 'billing/SKILL.md' or 'billing/scripts/create_invoice.py'.",
                          {"path": {"type": "string"}}, ["path"]), self.read_file)
        self.register(_fn("grep", "Search the library's files for a regular expression, ignoring case. Returns up to 100 matching lines as 'path:line: text'. 'path' limits the search to one folder or file; by default it searches the whole library.",
                          {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]), self.grep)
        if write:
            self.register(_fn("edit_file", "Replace text in a file of the library. old_text must match the file exactly and occur once, unless replace_all is true, which replaces every occurrence.",
                              {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"},
                               "replace_all": {"type": "boolean"}}, ["path", "old_text", "new_text"]), self.edit_file)
            self.register(_fn("write_file", "Create a file in the library, or overwrite a whole file, with this content. Folders are created as needed.",
                              {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]), self.write_file)
            self.register(_fn("delete_file", "Delete a file of the library. A folder left empty is removed.",
                              {"path": {"type": "string"}}, ["path"]), self.delete_file)

    def _resolve(self, path: str) -> Path:
        path = (path or ".").strip()
        if path.startswith("/"):
            path = path.lstrip("/")
        p = (self.root / path).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"{path!r} is outside the library")
        return p

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) if p != self.root else "."

    def list_files(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ToolError(f"no folder {path!r}")
        entries = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty)"

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"no file {path!r}")
        return p.read_text()

    def grep(self, pattern: str, path: str = ".") -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            raise ToolError(f"invalid regular expression: {e}") from e
        base = self._resolve(path)
        files = [base] if base.is_file() else sorted(x for x in base.rglob("*") if x.is_file())
        hits = []
        for f in files:
            for i, line in enumerate(f.read_text().splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self._rel(f)}:{i}: {line.strip()}")
        if not hits:
            return "No matches."
        more = f"\n({len(hits) - MAX_GREP} more matches not shown)" if len(hits) > MAX_GREP else ""
        return "\n".join(hits[:MAX_GREP]) + more

    def edit_file(self, path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"no file {path!r}")
        text = p.read_text()
        n = text.count(old_text) if old_text else 0
        if n == 0:
            raise ToolError(f"old_text was not found in {path}")
        if n > 1 and not replace_all:
            raise ToolError(f"old_text occurs {n} times in {path}; give more context or set replace_all")
        p.write_text(text.replace(old_text, new_text) if replace_all else text.replace(old_text, new_text, 1))
        self.writes.append({"op": "edit", "path": self._rel(p), "replaced": n if replace_all else 1})
        return f"Edited {self._rel(p)} ({n if replace_all else 1} replacement{'s' if (n if replace_all else 1) > 1 else ''})."

    def write_file(self, path: str, content: str) -> str:
        p = self._resolve(path)
        if p == self.root or p.is_dir():
            raise ToolError(f"{path!r} is a folder")
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.writes.append({"op": "overwrite" if existed else "create", "path": self._rel(p)})
        return f"{'Overwrote' if existed else 'Created'} {self._rel(p)}."

    def delete_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"no file {path!r}")
        p.unlink()
        parent = p.parent
        while parent != self.root and parent.is_dir() and not any(parent.iterdir()):
            shutil.rmtree(parent)
            parent = parent.parent
        self.writes.append({"op": "delete", "path": self._rel(p)})
        return f"Deleted {self._rel(p)}."


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

_STEM = snowballstemmer.stemmer("english")
_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were will with "
            "we our us you your i me my do does how what which who when where".split())


def tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return _STEM.stemWords([w for w in words if w not in _STOP])


class BM25:
    def __init__(self, docs: dict[str, str], k1: float = 1.2, b: float = 0.75) -> None:
        self.ids = list(docs)
        self.tf = [Counter(tokens(docs[i])) for i in self.ids]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(self.ids)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q = tokens(query)
        scored = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scored.append((self.ids[i], s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


# ---------------------------------------------------------------------------
# Linked tools
# ---------------------------------------------------------------------------

def _first_sentence(text: str) -> str:
    m = re.match(r"(.+?\.)(\s|$)", text.strip())
    return m.group(1) if m else text.strip()


class LinkedTools(Toolbox):
    def __init__(self, path: Path, *, write: bool) -> None:
        super().__init__()
        self.path = path
        self.lib = load_linked(path)
        self._fidx: BM25 | None = None
        self._gidx: BM25 | None = None
        self.register(_fn("search_functions", "Search the functions library by words in their names, signatures and docstrings. Returns the best matches, each with its signature, docstring and the guidance linked to it (id and title).",
                          {"query": {"type": "string"}, "k": {"type": "integer", "description": "How many results (default 8)."}},
                          ["query"]), self.search_functions)
        self.register(_fn("get_function", "Get one function by name: its signature, docstring and code, and the full text of every guidance entry linked to it.",
                          {"name": {"type": "string"}}, ["name"]), self.get_function)
        self.register(_fn("list_functions", "List every function in the library by name, with the first sentence of its docstring.",
                          {}, []), self.list_functions)
        self.register(_fn("search_guidance", "Search the guidance library by words in titles and texts. Returns the best matches, each with its id, title, text and the functions it governs.",
                          {"query": {"type": "string"}, "k": {"type": "integer", "description": "How many results (default 8)."}},
                          ["query"]), self.search_guidance)
        self.register(_fn("get_guidance", "Get one guidance entry by id: its title, text and the functions it governs.",
                          {"guidance_id": {"type": "integer"}}, ["guidance_id"]), self.get_guidance)
        self.register(_fn("list_guidance", "List every guidance entry by id and title, with the number of functions it governs.",
                          {}, []), self.list_guidance)
        if write:
            self.register(_fn("add_guidance", "Add a guidance entry: a title, its text, and the names of the functions it governs. Returns the new entry.",
                              {"title": {"type": "string"}, "content": {"type": "string"},
                               "functions": {"type": "array", "items": {"type": "string"}}},
                              ["title", "content"]), self.add_guidance)
            self.register(_fn("update_guidance", "Change a guidance entry. Only the fields you give change; functions, when given, replaces the list of functions it governs.",
                              {"guidance_id": {"type": "integer"}, "title": {"type": "string"}, "content": {"type": "string"},
                               "functions": {"type": "array", "items": {"type": "string"}}},
                              ["guidance_id"]), self.update_guidance)
            self.register(_fn("delete_guidance", "Delete a guidance entry.",
                              {"guidance_id": {"type": "integer"}}, ["guidance_id"]), self.delete_guidance)

    # indexes are rebuilt after a write
    def _f(self) -> BM25:
        if self._fidx is None:
            self._fidx = BM25({n: f"{n} {f['signature']} {f['docstring']}" for n, f in self.lib["functions"].items()})
        return self._fidx

    def _g(self) -> BM25:
        if self._gidx is None:
            self._gidx = BM25({gid: f"{g['title']} {g['content']}" for gid, g in self.lib["guidance"].items()})
        return self._gidx

    def _gid(self, guidance_id: Any) -> str:
        gid = str(guidance_id).strip().lstrip("[").rstrip("]")
        if gid not in self.lib["guidance"]:
            raise ToolError(f"no guidance entry with id {guidance_id}")
        return gid

    def _check_functions(self, functions: Any) -> list[str]:
        if not isinstance(functions, list) or not all(isinstance(x, str) for x in functions):
            raise ToolError("functions must be a list of function names")
        unknown = [x for x in functions if x not in self.lib["functions"]]
        if unknown:
            raise ToolError(f"no function named {', '.join(unknown)}")
        return list(dict.fromkeys(functions))

    def _dump(self, obj: Any) -> str:
        return json.dumps(obj, indent=1, ensure_ascii=False)

    def search_functions(self, query: str, k: int = 8) -> str:
        hits = self._f().search(query, max(1, min(int(k), 20)))
        if not hits:
            return "No matches."
        return self._dump([{"name": n, "signature": self.lib["functions"][n]["signature"],
                            "docstring": self.lib["functions"][n]["docstring"],
                            "guidance": [{"id": int(gid), "title": g["title"]} for gid, g in guidance_for(self.lib, n)]}
                           for n, _ in hits])

    def get_function(self, name: str) -> str:
        name = name.strip().removesuffix("()")
        f = self.lib["functions"].get(name)
        if f is None:
            raise ToolError(f"no function named {name!r}")
        return self._dump({"name": name, "signature": f["signature"], "docstring": f["docstring"], "code": f["code"],
                           "guidance": [{"id": int(gid), "title": g["title"], "content": g["content"]}
                                        for gid, g in guidance_for(self.lib, name)]})

    def list_functions(self) -> str:
        return "\n".join(f"{n}: {_first_sentence(f['docstring'])}" for n, f in sorted(self.lib["functions"].items()))

    def search_guidance(self, query: str, k: int = 8) -> str:
        hits = self._g().search(query, max(1, min(int(k), 20)))
        if not hits:
            return "No matches."
        return self._dump([{"id": int(gid), "title": self.lib["guidance"][gid]["title"],
                            "content": self.lib["guidance"][gid]["content"],
                            "functions": self.lib["guidance"][gid]["functions"]} for gid, _ in hits])

    def get_guidance(self, guidance_id: int) -> str:
        gid = self._gid(guidance_id)
        g = self.lib["guidance"][gid]
        return self._dump({"id": int(gid), "title": g["title"], "content": g["content"], "functions": g["functions"]})

    def list_guidance(self) -> str:
        return "\n".join(f"[{gid}] {g['title']} ({len(g['functions'])} functions)" for gid, g in self.lib["guidance"].items())

    def _saved(self) -> None:
        save_linked(self.lib, self.path)
        self._gidx = None

    def add_guidance(self, title: str, content: str, functions: list[str] | None = None) -> str:
        fns = self._check_functions(functions or [])
        gid = str(self.lib["next_id"])
        self.lib["next_id"] += 1
        self.lib["guidance"][gid] = {"title": title, "content": content, "functions": fns}
        self._saved()
        self.writes.append({"op": "add", "id": gid, "functions": fns})
        return self._dump({"added": {"id": int(gid), "title": title, "content": content, "functions": fns}})

    def update_guidance(self, guidance_id: int, title: str | None = None, content: str | None = None,
                        functions: list[str] | None = None) -> str:
        gid = self._gid(guidance_id)
        g = self.lib["guidance"][gid]
        changed = []
        if title is not None:
            g["title"] = title
            changed.append("title")
        if content is not None:
            g["content"] = content
            changed.append("content")
        if functions is not None:
            g["functions"] = self._check_functions(functions)
            changed.append("functions")
        if not changed:
            raise ToolError("nothing to change: give title, content or functions")
        self._saved()
        self.writes.append({"op": "update", "id": gid, "fields": changed})
        return self._dump({"updated": {"id": int(gid), **g}})

    def delete_guidance(self, guidance_id: int) -> str:
        gid = self._gid(guidance_id)
        g = self.lib["guidance"].pop(gid)
        self._saved()
        self.writes.append({"op": "delete", "id": gid, "title": g["title"]})
        return f"Deleted guidance entry {gid} ({g['title']})."
