"""Running the model's code: one namespace per task, with limits.

A task's namespace holds the API operations, a few preloaded modules,
submit(answer) and, in the library arm, the library's functions. Each
run_python call is one cell: it is compiled under the name "<cell N>", run with
its output captured, and stopped if it runs away. submit raises Submitted, a
BaseException, so a bare `except Exception` in the model's code cannot swallow
it, and the same holds for the limits.
"""

from __future__ import annotations

import ast
import builtins
import collections
import contextlib
import datetime
import functools
import io
import itertools
import json
import linecache
import math
import re
import statistics
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from .api import ApiError, OpLimit

PRELOADED = {"re": re, "math": math, "json": json, "datetime": datetime, "collections": collections,
             "itertools": itertools, "functools": functools, "statistics": statistics}
IMPORTABLE = set(PRELOADED) | {"string", "operator", "typing", "dataclasses", "decimal", "fractions",
                               "heapq", "bisect", "copy", "time", "textwrap"}
BLOCKED = {"open", "input", "exec", "eval", "compile", "breakpoint", "exit", "quit", "help", "memoryview"}
OUTPUT_LIMIT = 4000
TRACKED = ("<cell", "<library:", "<task:")  # code the model wrote: traced, and shown in tracebacks
LINE_LIMIT = 3_000_000


class Submitted(BaseException):
    def __init__(self, answer: Any):
        super().__init__("submitted")
        self.answer = answer


class LineLimit(BaseException):
    pass


class Saved(BaseException):
    """Raised by save(fn, *args, **kwargs) in the library arm's save step."""

    def __init__(self, fn: Any, args: tuple, kwargs: dict):
        super().__init__("saved")
        self.fn, self.args, self.kwargs = fn, args, kwargs


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root not in IMPORTABLE:
        raise ImportError(f"import of {name!r} is not allowed here; allowed: {', '.join(sorted(IMPORTABLE))}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def safe_builtins() -> dict[str, Any]:
    b = {k: v for k, v in vars(builtins).items() if k not in BLOCKED and k != "__import__"}
    b["__import__"] = _safe_import
    return b


def register_source(filename: str, source: str) -> None:
    """Make tracebacks show the lines of code that has no file."""
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)


@dataclass
class CellResult:
    output: str
    error: str | None = None  # the formatted traceback, if the cell raised
    error_type: str | None = None
    submitted: bool = False
    answer: Any = None
    saved: Saved | None = None
    library_in_traceback: list[str] = field(default_factory=list)


def _truncate(s: str, limit: int = OUTPUT_LIMIT) -> str:
    if len(s) <= limit:
        return s
    head, tail = s[: limit - 1000], s[-800:]
    return f"{head}\n... [{len(s) - len(head) - len(tail)} characters cut] ...\n{tail}"


def format_traceback(exc: BaseException) -> tuple[str, list[str]]:
    """The frames of the model's code and the library only, then the error.
    Returns the text and the library functions that appear in it, innermost last."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename.startswith(TRACKED)]
    lines = ["Traceback (most recent call last):"]
    lib: list[str] = []
    for f in frames:
        lines.append(f'  File "{f.filename}", line {f.lineno}, in {f.name}')
        if f.line:
            lines.append(f"    {f.line.strip()}")
        if f.filename.startswith("<library:"):
            lib.append(f.filename[len("<library:"):-1])
    lines.append(f"{type(exc).__name__}: {exc}")
    return "\n".join(lines), list(dict.fromkeys(lib))


class Namespace:
    """One task's namespace and the bookkeeping the arms need."""

    def __init__(self, ops: dict[str, Callable[..., Any]], extra: dict[str, Any] | None = None):
        self.ops = ops
        self.g: dict[str, Any] = {"__builtins__": safe_builtins(), "__name__": "__task__", "ApiError": ApiError}
        self.g.update(PRELOADED)
        self.g.update({name: self._logged(name, fn) for name, fn in ops.items()})
        self.g["submit"] = self._submit
        if extra:
            self.g.update(extra)
        self.cells = 0
        self.defined: dict[str, dict[str, str]] = {}  # cell filename -> {function name: source}
        self.ran: set[tuple[str, str]] = set()  # (filename, function name) of the model's functions that ran
        self.cell_library_calls: dict[int, set[str]] = {}  # cell number -> library functions that ran in it
        self.op_log: list[tuple[str, str, int]] = []  # (operation, the caller's filename, cell number)
        self.library_returns: list[tuple[str, Any]] = []  # (library function, value) for calls made from a cell

    @property
    def library_calls(self) -> set[str]:
        return {fn[len("<library:"):-1] for fn, _ in self.ran if fn.startswith("<library:")}

    def _logged(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        ns = self

        def op(*args: Any, **kwargs: Any) -> Any:
            ns.op_log.append((name, sys._getframe(1).f_code.co_filename, ns.cells))
            return fn(*args, **kwargs)

        op.__name__, op.__doc__ = name, fn.__doc__
        op.__signature__ = getattr(fn, "__signature__", None)  # type: ignore[attr-defined]
        return op

    @staticmethod
    def _submit(answer: Any = None) -> None:
        raise Submitted(answer)

    def exec_sources(self, sources: dict[str, str], kind: str = "library") -> None:
        """Define functions from source, each under its own filename, so their
        frames are traced and shown in tracebacks as "<kind:name>"."""
        for name, src in sources.items():
            fn = f"<{kind}:{name}>"
            register_source(fn, src)
            exec(compile(src, fn, "exec"), self.g)

    def load_library(self, sources: dict[str, str]) -> None:
        self.exec_sources(sources, "library")

    def run(self, code: str) -> CellResult:
        self.cells += 1
        cell = self.cells
        filename = f"<cell {cell}>"
        try:
            tree = ast.parse(code, filename=filename)
        except SyntaxError as e:
            return CellResult(output="", error=f"SyntaxError: {e.msg} (line {e.lineno})", error_type="SyntaxError")
        register_source(filename, code)
        buf = io.StringIO()
        lines = [0]
        ran = self.ran
        cell_lib = self.cell_library_calls.setdefault(cell, set())
        returns = self.library_returns

        def local_trace(frame, event, arg):
            if event == "line":
                lines[0] += 1
                if lines[0] > LINE_LIMIT:
                    raise LineLimit(f"The code ran more than {LINE_LIMIT:,} lines; it was stopped.")
            return local_trace

        def library_trace(frame, event, arg):
            if event == "return" and frame.f_back is not None and frame.f_back.f_code.co_filename.startswith("<cell"):
                returns.append((frame.f_code.co_name, arg))
            return local_trace(frame, event, arg) and library_trace

        def global_trace(frame, event, arg):
            fn = frame.f_code.co_filename
            if fn.startswith(TRACKED):
                if frame.f_code.co_name != "<module>":
                    ran.add((fn, frame.f_code.co_name))
                    if fn.startswith("<library:"):
                        cell_lib.add(fn[len("<library:"):-1])
                        return library_trace
                return local_trace
            return None

        # As in a notebook, the value of a final bare expression is shown.
        shown = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            v = tree.body[-1].value
            if not (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("print", "submit", "save")):
                shown = ast.Expression(body=v)
                tree = ast.Module(body=tree.body[:-1], type_ignores=[])
        result = CellResult(output="")
        old = sys.gettrace()
        sys.settrace(global_trace)
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, filename, "exec"), self.g)
                if shown is not None:
                    value = eval(compile(shown, filename, "eval"), self.g)
                    if value is not None:
                        print(repr(value))
        except Submitted as s:
            result.submitted, result.answer = True, s.answer
        except Saved as s:
            result.saved = s
        except (LineLimit, OpLimit) as e:
            result.error, result.error_type = f"{type(e).__name__}: {e}", type(e).__name__
        except Exception as e:  # noqa: BLE001 - the model's code may raise anything
            result.error, result.library_in_traceback = format_traceback(e)
            result.error_type = type(e).__name__
        finally:
            sys.settrace(old)
        result.output = _truncate(buf.getvalue())
        self._record_defs(tree, code, filename)
        return result

    def _record_defs(self, tree: ast.Module, code: str, filename: str) -> None:
        """Remember the source of each top-level function this cell defined (and
        that the namespace now holds), for the library arm to store."""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                obj = self.g.get(node.name)
                if getattr(getattr(obj, "__code__", None), "co_filename", None) == filename:
                    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                    src = "\n".join(code.splitlines()[start - 1: node.end_lineno]) + "\n"
                    self.defined.setdefault(filename, {})[node.name] = src

    def task_functions(self) -> dict[str, str]:
        """Functions defined in this task that the namespace still holds, in their
        final versions, as source."""
        out = {}
        for filename, defs in self.defined.items():
            for name, src in defs.items():
                obj = self.g.get(name)
                if getattr(getattr(obj, "__code__", None), "co_filename", None) == filename:
                    out[name] = src
        return out

    def format(self, r: CellResult) -> str:
        """What the model sees after a cell."""
        parts = []
        if r.output:
            parts.append(r.output.rstrip("\n"))
        if r.error:
            parts.append(r.error)
        if r.submitted:
            parts.append("Submitted.")
        return "\n".join(parts) if parts else "(no output)"
