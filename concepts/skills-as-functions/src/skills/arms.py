"""What each arm keeps between tasks, and how it shows it to the model.

none       nothing.
notes      one notes document, rewritten by the model after every task.
retrieval  the code of every task checked correct; the three most similar are
           shown as worked examples (BM25 over the task text).
fullctx    one continuous conversation: every earlier task's turns stay in it.
library    the idea: after a task is checked correct, the model saves how it
           did it as functions, which are stored only if re-running them on
           the workspace as it was before the task reproduces the checked
           result; they are listed by signature and docstring, defined in the
           namespace of every later task, and replaced when saved again.
"""

from __future__ import annotations

import ast
import builtins
import dis
import importlib
import math
import re
import types
from collections import Counter
from typing import Any

import json

from .api import OPS_BY_SIZE, READ_ONLY
from .sandbox import BLOCKED, IMPORTABLE, PRELOADED, Namespace, Saved
from .tasks import Task

NOTES_WORDS = 1500
NOTES_HARD_CAP = 2000
NOTES_MAX_TOKENS = 8192  # 1,500 words of prose and code can run past 4,000 tokens


def _finish_reason(reply: Any) -> str | None:
    try:
        return reply.raw["choices"][0].get("finish_reason")
    except (AttributeError, KeyError, IndexError, TypeError):
        return None
TOP_K = 3


class Arm:
    name = "base"

    def __init__(self, size: str):
        self.size = size
        self.last_passed: bool | None = None

    # what the next task sees
    def memory_section(self, task: Task) -> str | None:
        return None

    def history(self) -> list[dict[str, Any]]:
        return []

    def history_cache_marks(self) -> set[int]:
        return set()

    def user_message(self, task: Task) -> str:
        return f"Task: {task.text}"

    def namespace_extra(self) -> dict[str, Any]:
        return {}

    def library_sources(self) -> dict[str, str]:
        return {}

    def traceback_note(self, names: list[str], ns: Namespace) -> str:
        return ""

    # after the task
    def record_turns(self, user: dict[str, Any], messages: list[dict[str, Any]]) -> None:
        pass

    async def after_task(self, llm: Any, model: str, task: Task, run: Any, ctx: Any, meta: dict[str, Any]) -> dict[str, Any]:
        """Update memory from the finished task and its checked outcome. Returns
        what to add to the task's record."""
        self.last_passed = run.passed
        return {}

    def snapshot(self) -> Any:
        return None


class NoMemory(Arm):
    name = "none"


# -- notes --------------------------------------------------------------------------------
NOTES_SYSTEM = """You keep the notes of an assistant that carries out tasks in Brightdesk, a company's customer workspace, by writing Python against its API. The assistant reads the notes at the start of every task, together with the current API reference. The notes are all it keeps from earlier tasks."""


def transcript_for_notes(run: Any, limit: int = 1500) -> str:
    parts = []
    for i, c in enumerate(run.cells, start=1):
        if c.get("tool_submit"):
            parts.append(f"<run {i}>\nIt submitted {run.answer!r} with the submit tool.\n</run {i}>")
            continue
        if c["code"] is None:
            continue
        seen = c["seen"]
        if len(seen) > limit:
            seen = seen[:limit] + f"\n... [{len(seen) - limit} characters cut]"
        parts.append(f"<run {i}>\n```python\n{c['code'].rstrip()}\n```\nOutput:\n{seen}\n</run {i}>")
    return "\n\n".join(parts) if parts else "(it ran no code)"


class Notes(Arm):
    name = "notes"

    def __init__(self, size: str):
        super().__init__(size)
        self.notes = ""

    def memory_section(self, task: Task) -> str:
        body = self.notes.strip() or "(no notes yet)"
        return ("Your notes from earlier tasks in this workspace (you wrote and updated them after each task; "
                f"the API may have changed since they were written):\n<notes>\n{body}\n</notes>")

    async def after_task(self, llm, model, task, run, ctx, meta):
        self.last_passed = run.passed
        outcome = "The result was checked and was correct." if run.passed else "The result was checked and was NOT correct."
        prompt = f"""The current notes:
<notes>
{self.notes.strip() or "(no notes yet)"}
</notes>

The task the assistant has just done: {task.text}

What it ran, and what each run printed:
{transcript_for_notes(run)}

Outcome: {outcome}

Rewrite the notes so that they help with later tasks: how to carry out tasks of this kind and the steps they share with other tasks, the shapes of the records, rules and pitfalls learned from errors, and corrections to anything in the notes that proved wrong or out of date. Keep what is still useful. You may include short code snippets. At most {NOTES_WORDS:,} words. Reply with the complete notes and nothing else."""
        from .agent import call_record

        messages = [{"role": "system", "content": NOTES_SYSTEM}, {"role": "user", "content": prompt}]
        calls, cut = [], False
        for attempt in range(2):
            reply = await llm.chat(model, messages, temperature=0, max_tokens=NOTES_MAX_TOKENS,
                                   meta={**meta, "role": "notes", "attempt": attempt})
            calls.append(call_record(reply, "notes"))
            cut = _finish_reason(reply) == "length"
            if not cut:
                break
            # the rewrite ran past the output limit: ask once for a shorter one
            messages = messages + [{"role": "assistant", "content": reply.text},
                                   {"role": "user", "content": f"Those notes were cut off at the length limit. Reply with the complete notes again, shorter: at most {NOTES_WORDS:,} words in all, code included."}]
        new = reply.text.strip()
        n_words = len(new.split())
        shortened = False
        if n_words > NOTES_HARD_CAP and not cut:
            # far over the limit: ask once for a shorter rewrite rather than cut off its end
            messages = messages + [{"role": "assistant", "content": reply.text},
                                   {"role": "user", "content": f"Those notes are {n_words:,} words, over the limit. Reply with the complete notes again in at most {NOTES_WORDS:,} words, code included: merge and shorten, keeping what matters most for later tasks."}]
            reply = await llm.chat(model, messages, temperature=0, max_tokens=NOTES_MAX_TOKENS,
                                   meta={**meta, "role": "notes", "attempt": "shorten"})
            calls.append(call_record(reply, "notes"))
            new, shortened = reply.text.strip(), True
        truncated = False
        if new:
            words = list(re.finditer(r"\S+", new))
            if len(words) > NOTES_HARD_CAP:  # still too long: keep the first 2,000 words
                new = new[: words[NOTES_HARD_CAP - 1].end()]
                truncated = True
            self.notes = new
        return {"memory_calls": calls, "notes_words": len(self.notes.split()), "notes_shortened": shortened,
                "notes_truncated": truncated, "notes_cut_at_limit": cut}

    def snapshot(self):
        return self.notes


# -- retrieval ----------------------------------------------------------------------------------
def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 score of each doc for the query."""
    toks = [tokens(d) for d in docs]
    n = len(docs)
    if n == 0:
        return []
    avg = sum(len(t) for t in toks) / n or 1.0
    df = Counter(w for t in toks for w in set(t))
    q = tokens(query)
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return scores


def solution_code(run: Any) -> str:
    """The code of a task's cells that ran without an error, in order (a
    submission made with the submit tool is shown as the call it stands for)."""
    parts = []
    for c in run.cells:
        if c.get("tool_submit"):
            parts.append(f"submit({run.answer!r})  # with the submit tool")
        elif c["code"] is not None and c["error_type"] is None:
            parts.append(c["code"].rstrip())
    return "\n\n".join(parts)


class Retrieval(Arm):
    name = "retrieval"

    def __init__(self, size: str):
        super().__init__(size)
        self.store: list[dict[str, Any]] = []  # {"position", "text", "code"}

    def retrieve(self, text: str) -> list[dict[str, Any]]:
        scores = bm25_rank(text, [s["text"] for s in self.store])
        ranked = sorted(range(len(self.store)), key=lambda i: (-scores[i], -self.store[i]["position"]))
        return [self.store[i] for i in ranked if scores[i] > 0][:TOP_K]

    def memory_section(self, task: Task) -> str:
        hits = self.retrieve(task.text)
        head = ("Code that solved similar earlier tasks in this workspace (each was checked correct when it ran; "
                "the API may have changed since):")
        if not hits:
            return head + "\n(no earlier solutions yet)"
        blocks = [f"<example task={s['text']!r}>\n```python\n{s['code']}\n```\n</example>" for s in hits]
        return head + "\n" + "\n\n".join(blocks)

    async def after_task(self, llm, model, task, run, ctx, meta):
        self.last_passed = run.passed
        hits = [s["position"] for s in self.retrieve(task.text)]
        if run.passed:
            self.store.append({"position": task.position, "text": task.text, "code": solution_code(run)})
        return {"retrieved": hits}

    def snapshot(self):
        return self.store


# -- one long context ---------------------------------------------------------------------------
class FullContext(Arm):
    name = "fullctx"

    def __init__(self, size: str):
        super().__init__(size)
        self.conversation: list[dict[str, Any]] = []
        self.task_starts: list[int] = []  # index of each task's user message in the conversation

    def memory_section(self, task: Task) -> str:
        return ("Earlier tasks in this workspace are in this conversation, with the code you ran and whether each "
                "result was checked correct. The API may have changed since.")

    def user_message(self, task: Task) -> str:
        if self.last_passed is None:
            return f"Task: {task.text}"
        outcome = "correct" if self.last_passed else "NOT correct"
        return f"The result of the previous task was checked and was {outcome}.\n\nTask: {task.text}"

    def history(self):
        return self.conversation

    def history_cache_marks(self):
        return {self.task_starts[-1]} if self.task_starts else set()

    def record_turns(self, user, messages):
        self.task_starts.append(len(self.conversation))
        self.conversation.extend([user] + list(messages))

    def snapshot(self):
        return {"messages": len(self.conversation)}


# -- the library ----------------------------------------------------------------------------------
LIBRARY_HOWTO = """How the library works:
- Call library functions wherever they fit, instead of writing those steps again. If one does this kind of task, call it and submit what it returns.
- If a library function fails or is out of date, repair it: define it again in your code, with the same name and arguments.
- After each task that is checked correct, you will be asked to save how you did it as library functions. They are tested by re-running them before they are stored.
- library_source(name) returns the source of a library function."""

SAVE_PROMPT = """The result was checked and is correct. Now save how you did it to your library, in one run_python call:
- Define a function for this kind of task that takes the task's values (names, dates, messages and so on) as arguments and does the whole task, returning the answer, or None for a task that only does something. Build it from library functions where they fit, and put each step that other tasks could share in a function of its own, with an expressive name, arguments and a docstring.
- If a library function already does this kind of task, end with a call of it. To improve or repair a library function, define it again with the same name and arguments: the new version replaces the old one for every function that calls it. Do not add a second version under a new name.{failed}
- A library function can use its arguments, the API, builtins, imported modules and other library functions, and no other names defined outside it.
- End the code with save(your_function, <this task's values>), for example save(company_city, "Ada Lovelace"). The call is re-run on the workspace as it was before this task, and the functions it runs are stored only if it gives the same answer and makes the same changes as your checked result. Do not call submit."""

SAVE_ATTEMPTS = 2

RESERVED = {"submit", "save", "library_source", "ApiError", "print"}


def _save(fn: Any, *args: Any, **kwargs: Any) -> None:
    """save(your_function, <this task's values>): store it if a re-run reproduces the checked result."""
    raise Saved(fn, args, kwargs)


def global_names(code: types.CodeType) -> set[str]:
    """Names a function's code (and the code nested in it) looks up as globals."""
    out = set()
    for ins in dis.get_instructions(code):
        if ins.opname in ("LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL"):
            out.add(ins.argval if isinstance(ins.argval, str) else str(ins.argval))
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out |= global_names(c)
    return out


def listing_entry(source: str) -> str:
    """`def name(args) -> ret:` and the docstring, without the body."""
    fn = next(n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    doc = ast.get_docstring(fn) or "(no docstring)"
    if len(doc) > 600:
        doc = doc[:600] + " ..."
    stub = ast.FunctionDef(name=fn.name, args=fn.args, body=[ast.Pass()], decorator_list=[], returns=fn.returns,
                           type_params=[], lineno=1, col_offset=0)
    head = ast.unparse(stub).splitlines()[0]
    doc_lines = doc.splitlines()
    doc_text = "\n    ".join(doc_lines)
    return f'{head}\n    """{doc_text}"""'


def _binding(name: str, value: Any) -> str | None:
    """A line that recreates a global a stored function needs: an import of an
    allowed module or of a class or function from one, or a literal constant."""
    if isinstance(value, types.ModuleType):
        mod = value.__name__
        if mod.split(".")[0] not in IMPORTABLE:
            return None
        return f"import {mod}" if mod == name else f"import {mod} as {name}"
    mod = getattr(value, "__module__", None)
    attr = getattr(value, "__name__", None)
    if isinstance(mod, str) and isinstance(attr, str) and mod.split(".")[0] in IMPORTABLE:
        try:
            module = importlib.import_module(mod)
        except ImportError:
            module = None
        if module is not None and getattr(module, attr, None) is value:
            return f"from {mod} import {attr}" if attr == name else f"from {mod} import {attr} as {name}"
    if _literal(value):
        return f"{name} = {value!r}"
    return None


def _literal(value: Any) -> bool:
    """A constant worth carrying with a function: a scalar, or a small flat
    list, tuple, set or dict of scalars. A nested record (task data) is not."""
    scalar = (str, int, float, bool, type(None))
    if isinstance(value, scalar):
        return not isinstance(value, str) or len(value) <= 200
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value) <= 20 and all(isinstance(v, scalar) for v in value)
    if isinstance(value, dict):
        return len(value) <= 20 and all(isinstance(k, scalar) and isinstance(v, scalar) for k, v in value.items())
    return False


class Library(Arm):
    name = "library"

    def __init__(self, size: str):
        super().__init__(size)
        self.functions: dict[str, dict[str, Any]] = {}  # name -> {"source", "created", "updated"}

    def memory_section(self, task: Task) -> str:
        if self.functions:
            body = "\n\n".join(listing_entry(self.functions[n]["source"]) for n in sorted(self.functions))
        else:
            body = "(empty)"
        return ("Your function library: functions you wrote in earlier tasks that were checked correct. They are "
                "already defined in this task's namespace; call them directly. The API may have changed since they "
                f"were written.\n<library>\n{body}\n</library>\n\n{LIBRARY_HOWTO}")

    def library_sources(self) -> dict[str, str]:
        return {n: f["source"] for n, f in self.functions.items()}

    def namespace_extra(self) -> dict[str, Any]:
        lib = self

        def library_source(name: str) -> str:
            """The source of a library function."""
            f = lib.functions.get(name)
            if f is None:
                raise KeyError(f"no library function named {name!r}; the library has: {', '.join(sorted(lib.functions))}")
            return f["source"]

        return {"library_source": library_source}

    def traceback_note(self, names: list[str], ns: Namespace) -> str:
        shown = []
        for n in names:
            obj = ns.g.get(n)
            # only if the namespace still holds the library's version (not one redefined in this task)
            if n in self.functions and getattr(getattr(obj, "__code__", None), "co_filename", "") == f"<library:{n}>":
                shown.append(self.functions[n]["source"].rstrip())
        if not shown:
            return ""
        return "\n\nSource of the library functions in this traceback:\n```python\n" + "\n\n".join(shown) + "\n```"

    def candidates(self, ns: Namespace) -> tuple[dict[str, str], dict[str, str]]:
        """The functions this task defined that can be stored, each with the
        bindings it needs prepended (an import, or a literal constant); and those
        that cannot, with the reason."""
        defined = ns.task_functions()
        ops = set(OPS_BY_SIZE[self.size])
        builtin_names = set(dir(builtins)) - BLOCKED
        rejected: dict[str, str] = {}
        ready: dict[str, str] = {}
        for name, src in defined.items():
            if name in ops or name in RESERVED or name in PRELOADED or name in builtin_names:
                rejected[name] = "its name is taken by the API, a builtin or a preloaded name"
                continue
            fn = ns.g[name]
            bindings, bad = [], []
            for g in sorted(global_names(fn.__code__)):
                if g == name or g in defined or g in self.functions:
                    continue  # a function stored with it, or already in the library
                if g in PRELOADED:
                    if ns.g.get(g) is not PRELOADED[g]:
                        bad.append(g)  # rebound in this task, e.g. by `from datetime import datetime`
                    continue
                if g in ops or g in RESERVED:
                    continue
                if g not in ns.g:
                    if g not in builtin_names:
                        bad.append(g)
                    continue
                value = ns.g[g]
                binding = _binding(g, value)
                if binding is None:
                    bad.append(g)
                else:
                    bindings.append(binding)
            if bad:
                rejected[name] = f"uses names defined outside it: {', '.join(bad)}"
                continue
            ready[name] = ("\n".join(bindings) + "\n\n" + src) if bindings else src
        # a function that calls a rejected one (with no stored version to fall back on) is rejected too
        changed = True
        while changed:
            changed = False
            for name in list(ready):
                calls = global_names(ns.g[name].__code__)
                missing = sorted(c for c in calls if c in rejected and c not in self.functions)
                if missing:
                    rejected[name] = f"calls functions that were not stored: {', '.join(missing)}"
                    del ready[name]
                    changed = True
        return ready, rejected

    def covered_by_library(self, task: Task, run: Any, ns: Namespace) -> bool:
        """The library did this task, so there is nothing new to save: no function
        was defined, the model's own top-level code changed nothing, a library
        function ran, and for a question, the answer submitted is a value a
        library function returned to the model's code."""
        from .tasks import KINDS

        if ns.task_functions() or not ns.library_calls:
            return False
        if any(op not in READ_ONLY and caller.startswith("<cell") for op, caller, _ in ns.op_log):
            return False
        if KINDS[task.kind].answer == "none":
            return True
        from .tasks import answer_ok

        for _, value in ns.library_returns:  # the answer submitted is a value a library function returned
            try:
                if value == run.answer or answer_ok(KINDS[task.kind].answer, value, run.answer):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def verify(self, task: Task, saved: Any, ns: Namespace, ctx: Any, answer: Any) -> tuple[bool, str, set[str], dict[str, str]]:
        """Re-run the saved call on the workspace as it was before the task, in a
        fresh namespace holding the library and this task's functions. Returns
        whether it reproduced the checked result, a message for the model, the
        task functions that ran, and the storable sources."""
        from .agent import TaskContext  # noqa: F401  (for readers: ctx is a TaskContext)
        from .api import Api
        from .tasks import KINDS, answer_ok
        from .world import clone, diff, observable

        name = getattr(saved.fn, "__name__", None)
        ready, rejected = self.candidates(ns)
        if not callable(saved.fn) or name not in ready and name not in self.functions:
            why = f" ({rejected[name]})" if name in rejected else ""
            return False, f"Not saved: save() needs a function defined in this task or in the library{why}.", set(), ready
        world = clone(task.before)
        vns = Namespace(Api(world, task.version, self.size).functions(), extra=self.namespace_extra())
        vns.load_library(self.library_sources())
        vns.exec_sources(ready, "task")
        vns.g["__saved_call__"] = (saved.args, saved.kwargs)
        r = vns.run(f"__result__ = {name}(*__saved_call__[0], **__saved_call__[1])")
        ran = {fn[len("<task:"):-1] for fn, _ in vns.ran if fn.startswith("<task:")}
        notes = [f"{n} was not stored: {why}" for n, why in rejected.items()]
        if r.error:
            return False, "Not saved: the re-run raised an error.\n" + r.error + ("\n" + "\n".join(notes) if notes else ""), ran, ready
        got = vns.g.get("__result__")
        problems = []
        if KINDS[task.kind].answer != "none" and not answer_ok(KINDS[task.kind].answer, task.expected, got):
            problems.append(f"it returned {got!r}, but your checked answer was {answer!r}")
        changes = diff(ctx.checked_state, observable(world))
        if changes:
            problems.append("it changed the workspace differently from your checked result: " + "; ".join(changes[:6]))
        if problems:
            return False, "Not saved: " + ", and ".join(problems) + ".", ran, ready
        return True, "", ran, ready

    async def after_task(self, llm, model, task, run, ctx, meta):
        from .agent import MAX_TOKENS, RUN_PYTHON, call_record

        self.last_passed = run.passed
        ns = ctx.ns
        out: dict[str, Any] = {"library_size_before": len(self.functions), "stored": [], "replaced": [],
                               "rejected": {}, "save": None, "memory_calls": []}
        if not run.passed:
            out["save"] = "not attempted (task failed)"
            return out
        if self.covered_by_library(task, run, ns):
            out["save"] = "not needed (done by library functions)"
            out["library_size"] = len(self.functions)
            return out
        failed = ""
        if run.library_failed:
            failed = ("\n- These library functions failed during this task: " + ", ".join(run.library_failed)
                      + ". If they are out of date, repair them here, so every function that calls them is fixed.")
        messages = ctx.request() + [{"role": "user", "content": SAVE_PROMPT.format(failed=failed)}]
        ns.g["save"] = _save
        save_turns: list[dict[str, Any]] = [messages[-1]]
        for attempt in range(SAVE_ATTEMPTS):
            reply = await llm.chat(model, messages, tools=[RUN_PYTHON], tool_choice="required", temperature=0,
                                   max_tokens=MAX_TOKENS, meta={**meta, "role": "save", "attempt": attempt})
            out["memory_calls"].append(call_record(reply, "save"))
            tool_calls = [{"id": tc.get("id"), "type": "function",
                           "function": {"name": (tc.get("function") or {}).get("name", "run_python"),
                                        "arguments": (tc.get("function") or {}).get("arguments") or ""}}
                          for tc in reply.tool_calls]
            if not tool_calls:
                turn = {"role": "assistant", "content": reply.text or "(no reply)"}
                nudge = {"role": "user", "content": "Use the run_python tool, and end the code with save(...)."}
                messages += [turn, nudge]
                save_turns += [turn, nudge]
                continue
            turn = {"role": "assistant", "content": reply.text or None, "tool_calls": tool_calls}
            messages.append(turn)
            save_turns.append(turn)
            verdict = None
            for tc in tool_calls:
                if verdict is not None and verdict[0]:
                    seen = "Not run: already saved."
                else:
                    try:
                        code = json.loads(tc["function"]["arguments"])["code"]
                        assert isinstance(code, str)
                    except Exception:  # noqa: BLE001
                        code = None
                    if code is None:
                        seen = 'Error: the arguments must be a JSON object with a string field "code".'
                    else:
                        r = ns.run(code)
                        seen = ns.format(r)
                        if r.saved is not None:
                            ok, msg, ran, ready = self.verify(task, r.saved, ns, ctx, run.answer)
                            if ok:
                                stored, replaced, rejected = self.store(task, ran, ready, ns)
                                out.update(stored=stored, replaced=replaced, rejected=rejected)
                                msg = "Saved." + (f" New: {', '.join(stored)}." if stored else "") + (
                                    f" Replaced: {', '.join(replaced)}." if replaced else "") + (
                                    "" if stored or replaced else " Nothing new: the library already had these functions.")
                            verdict = (ok, msg)
                            seen = (seen + "\n" + msg) if seen != "(no output)" else msg
                        elif r.submitted:
                            seen += "\nThis step does not submit: end the code with save(your_function, <this task's values>)."
                        elif not r.error:
                            seen += "\nNothing was saved: end the code with save(your_function, <this task's values>)."
                tool_msg = {"role": "tool", "tool_call_id": tc["id"], "content": seen}
                messages.append(tool_msg)
                save_turns.append(tool_msg)
            if verdict is not None and verdict[0]:
                out["save"] = f"saved on attempt {attempt + 1}"
                break
        else:
            out["save"] = "failed"
        ns.g.pop("save", None)
        out["rejected"] = self.candidates(ns)[1]
        out["save_turns"] = save_turns
        out["library_size"] = len(self.functions)
        return out

    def store(self, task: Task, ran: set[str], ready: dict[str, str], ns: Namespace) -> tuple[list[str], list[str], dict[str, str]]:
        """Store the task's functions that ran in the verified re-run."""
        stored, replaced = [], []
        for n in sorted(ran):
            if n not in ready:
                continue
            if n in self.functions:
                if self.functions[n]["source"] != ready[n]:
                    replaced.append(n)
                    self.functions[n] = {**self.functions[n], "source": ready[n], "updated": task.position}
            else:
                stored.append(n)
                self.functions[n] = {"source": ready[n], "created": task.position, "updated": task.position}
        return stored, replaced, self.candidates(ns)[1]

    def snapshot(self):
        return self.functions


ARMS = {"none": NoMemory, "notes": Notes, "retrieval": Retrieval, "fullctx": FullContext, "library": Library}
