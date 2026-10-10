"""One session's Python process. Run as a script by `executor.py`; it imports
nothing from this package, so any interpreter (any environment) can run it.

Protocol: one JSON object per line on stdin, one JSON reply per line on the
original stdout. User code never sees either: its stdin is /dev/null and its
stdout and stderr go to a capture file whose new contents are returned with
each reply.

    {"op": "init", "cwd": str, "api": {...} | null}
    {"op": "exec", "code": str, "tag": str, "read_only": bool, "timeout": int}
    {"op": "inspect", "detail": "summary" | "full"}
    {"op": "env"}                       # which interpreter this is
    {"op": "exit"}

Semantics:
- A cell is compiled as a module and executed with the session's namespace as
  its globals and locals, so every name it binds, in any block (if, for,
  while, try, with) or at top level, is a name of the session. If the cell ends
  with an expression, its value is shown, as in a REPL.
- read_only forks this process: the child runs the cell against a
  copy-on-write image of everything (names, objects, module state,
  environment variables) in a copy of the working directory, replies, and
  exits, so nothing the cell does reaches the session.
- After each stateful cell, the worker records which request (tag) last
  changed each name, by identity or by content, so the harness can tell which
  request's state a later cell read. This is measurement, timed separately.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import pickle
import select
import shutil
import signal
import sys
import tempfile
import time
import traceback
import types

MAX_OUT = 200_000  # characters of output returned per cell (the harness cuts further)
API_NAMES: set[str] = set()


class CellTimeout(BaseException):
    """Raised inside user code at its time limit; a BaseException, so an
    `except Exception` in the cell does not swallow it."""


def _alarm(signum, frame):
    raise CellTimeout("the cell ran past its time limit")


# ─────────────────────────── the environment's API ───────────────────────────
def make_api(cfg: dict, events: list) -> dict:
    """Functions every interpreter gets in the experiment: a slow export and a
    slow risk model, per client. Each call is recorded as an event."""
    import pandas as pd

    data_dir = cfg["data_dir"]
    load_s = float(cfg.get("load_s", 0))
    score_s = float(cfg.get("score_s", 0))
    known = set(cfg.get("clients", []))

    def _check(client_id):
        if not isinstance(client_id, str) or client_id not in known:
            raise ValueError(f"unknown client id {client_id!r}")

    def load_export(client_id):
        """Load a client's payments export (about 20 seconds). Every column is text."""
        _check(client_id)
        t0 = time.time()
        time.sleep(load_s)
        df = pd.read_csv(os.path.join(data_dir, f"{client_id}.csv"), dtype=str, keep_default_na=False)
        events.append({"fn": "load_export", "client": client_id, "s": round(time.time() - t0, 3)})
        return df

    def score_customers(client_id):
        """Run a client's churn-risk model (about 20 seconds): customer_id, risk."""
        _check(client_id)
        t0 = time.time()
        time.sleep(score_s)
        df = pd.read_csv(os.path.join(data_dir, f"{client_id}_risk.csv"), dtype={"customer_id": str, "risk": float})
        events.append({"fn": "score_customers", "client": client_id, "s": round(time.time() - t0, 3)})
        return df

    return {"load_export": load_export, "score_customers": score_customers}


# ─────────────────────────────── running a cell ──────────────────────────────
def names_read(tree: ast.Module) -> list[str]:
    """Names a cell may read from the session: every name it loads, except
    those a simple top-level statement surely bound earlier in the cell. A name
    bound only inside a block (if, for, try, ...) may not have been bound, so a
    later load of it still counts. Measurement only; it over-approximates."""
    stored: set[str] = set()
    out: list[str] = []
    for stmt in tree.body:
        for node in ast.walk(stmt):
            name = None
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                name = node.id
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                name = node.target.id  # x += 1 reads x
            if name and name not in stored and name not in out:
                out.append(name)
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        stored.add(n.id)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                stored.add((a.asname or a.name).split(".")[0])
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stored.add(stmt.name)
    return out


def show(v) -> str:
    try:
        return repr(v)
    except Exception as e:  # a broken __repr__ should not lose the cell
        return f"<unrepresentable {type(v).__name__}: {e}>"


def format_error(e: BaseException) -> str:
    """The traceback without this worker's own frames."""
    te = traceback.TracebackException.from_exception(e)
    te.stack = traceback.StackSummary.from_list([f for f in te.stack if f.filename != __file__])
    return "".join(te.format()).strip()


def run_cell(ns: dict, code: str, timeout: int) -> dict:
    sys.stdout.flush()
    sys.stderr.flush()
    start = os.lseek(1, 0, os.SEEK_END)
    t0 = time.perf_counter()
    value = error = None
    reads: list[str] = []
    try:
        tree = ast.parse(code, "<cell>", "exec")
        reads = names_read(tree)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        signal.alarm(max(1, int(timeout)))
        try:
            exec(compile(tree, "<cell>", "exec"), ns)
            if last is not None:
                v = eval(compile(last, "<cell>", "eval"), ns)
                if v is not None:
                    value = show(v)[:MAX_OUT]
        finally:
            signal.alarm(0)
    except BaseException as e:  # noqa: BLE001 - a sandbox reports everything, incl. SystemExit
        error = format_error(e)[:MAX_OUT]
    exec_s = time.perf_counter() - t0
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    end = os.lseek(1, 0, os.SEEK_END)
    out = os.pread(1, min(end - start, 4 * MAX_OUT), start).decode("utf-8", "replace")
    os.ftruncate(1, 0)
    os.lseek(1, 0, os.SEEK_SET)
    if len(out) > MAX_OUT:
        out = out[: MAX_OUT // 2] + f"\n...[{len(out) - MAX_OUT} characters cut]...\n" + out[-MAX_OUT // 2 :]
    return {"stdout": out, "value": value, "error": error, "exec_s": round(exec_s, 4), "reads_names": reads}


# ───────────────────────────── provenance (measurement) ──────────────────────
def _content(obj, depth=0):
    """A content digest for the kinds of objects analyses keep, or None."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # pragma: no cover - pandas is always there in the experiment
        np = pd = None
    if pd is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
        h = pd.util.hash_pandas_object(obj, index=True)
        cols = tuple(map(str, obj.columns)) if isinstance(obj, pd.DataFrame) else (str(obj.name),)
        dtypes = tuple(map(str, obj.dtypes)) if isinstance(obj, pd.DataFrame) else (str(obj.dtype),)
        return ("pd", obj.shape, cols, dtypes, int(h.sum()) if len(h) else 0)
    if np is not None and isinstance(obj, np.ndarray):
        if obj.nbytes > 50_000_000 or obj.dtype == object:
            return None
        return ("np", obj.shape, str(obj.dtype), hashlib.blake2b(obj.tobytes(), digest_size=16).hexdigest())
    if isinstance(obj, types.FunctionType):
        co = obj.__code__
        return ("fn", hashlib.blake2b(co.co_code + repr((co.co_consts, co.co_names, obj.__defaults__)).encode(),
                                      digest_size=16).hexdigest())
    if isinstance(obj, (str, bytes, int, float, complex, bool, type(None))):
        return ("v", hashlib.blake2b(pickle.dumps(obj), digest_size=16).hexdigest())
    if isinstance(obj, (dict, list, tuple, set, frozenset)) and depth < 3 and len(obj) <= 100_000:
        items = sorted(obj.items(), key=lambda kv: repr(kv[0])) if isinstance(obj, dict) else (
            sorted(obj, key=repr) if isinstance(obj, (set, frozenset)) else list(obj))
        parts = []
        for it in items:
            c = _content(it, depth + 1) if not isinstance(obj, dict) else (
                _content(it[0], depth + 1), _content(it[1], depth + 1))
            parts.append(c if c is not None else ("id", id(it)))
        return ("c", type(obj).__name__, hashlib.blake2b(repr(parts).encode(), digest_size=16).hexdigest())
    return None


def fingerprint(obj):
    try:
        return (id(obj), _content(obj))
    except Exception:
        return (id(obj), None)


def tracked(ns: dict):
    for k, v in ns.items():
        if k.startswith("__") or isinstance(v, types.ModuleType):
            continue
        if k in API_NAMES and getattr(v, "_api", False):
            continue
        yield k, v


# ─────────────────────────────── inspection ──────────────────────────────────
def describe(name, v, detail: str) -> dict:
    t = type(v).__name__
    d = {"name": name, "type": t}
    try:
        import pandas as pd

        if isinstance(v, pd.DataFrame):
            d["shape"] = list(v.shape)
            d["columns"] = {str(c): str(v[c].dtype) for c in list(v.columns)[:40]}
            if detail == "full":
                d["head"] = v.head(3).to_string()[:1500]
            return d
        if isinstance(v, pd.Series):
            d["length"] = len(v)
            d["dtype"] = str(v.dtype)
            if detail == "full":
                d["head"] = v.head(5).to_string()[:800]
            return d
    except Exception:
        pass
    if isinstance(v, (types.FunctionType, types.BuiltinFunctionType)):
        try:
            import inspect as _inspect

            d["signature"] = f"{name}{_inspect.signature(v)}"
        except Exception:
            pass
        return d
    if isinstance(v, type):
        return d
    if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
        try:
            d["length"] = len(v)
        except Exception:
            pass
    r = show(v)
    d["repr"] = r[:200] + ("..." if len(r) > 200 else "") if detail == "full" or len(r) <= 80 else r[:80] + "..."
    return d


# ─────────────────────────────────── main ────────────────────────────────────
def main() -> None:
    # Keep the protocol on private descriptors; user code gets /dev/null as
    # stdin and a capture file as stdout and stderr.
    proto_in = os.fdopen(os.dup(0), "r", encoding="utf-8")
    proto_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    cap = tempfile.TemporaryFile()
    os.dup2(cap.fileno(), 1)
    os.dup2(cap.fileno(), 2)
    sys.stdin = open(0, "r", closefd=False)
    sys.stdout = open(1, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
    sys.stderr = open(2, "w", buffering=1, closefd=False, encoding="utf-8", errors="replace")
    signal.signal(signal.SIGALRM, _alarm)

    ns: dict = {"__name__": "__main__", "__builtins__": __builtins__}
    events: list = []
    prov: dict[str, str] = {}
    fps: dict[str, tuple] = {}
    cwd = os.getcwd()

    def reply(obj) -> None:
        proto_out.write(json.dumps(obj, default=str) + "\n")
        proto_out.flush()

    def update_provenance(tag: str) -> float:
        t0 = time.perf_counter()
        seen = set()
        for k, v in list(tracked(ns)):
            seen.add(k)
            fp = fingerprint(v)
            old = fps.get(k)
            if old != fp:
                fps[k] = fp
                # a name rebound to an equal value keeps the request that made the value
                if old is None or fp[1] is None or old[1] != fp[1]:
                    prov[k] = tag
        for k in list(fps):
            if k not in seen:
                fps.pop(k, None)
                prov.pop(k, None)
        return time.perf_counter() - t0

    def read_only(code: str, timeout: int) -> dict:
        sys.stdout.flush()
        sys.stderr.flush()
        r, w = os.pipe()
        pid = os.fork()
        if pid == 0:  # the child: a copy-on-write image of the session
            snap = None
            try:
                os.close(r)
                own = tempfile.TemporaryFile()
                os.dup2(own.fileno(), 1)
                os.dup2(own.fileno(), 2)
                # The session's own directory is copied, and the child works in
                # the copy (and its temporary directory points into the copy).
                snap = tempfile.mkdtemp(prefix="readonly-", dir=os.path.dirname(cwd))
                shutil.copytree(cwd, snap, dirs_exist_ok=True, symlinks=True)
                here = os.getcwd()
                rel = os.path.relpath(here, cwd)
                if not rel.startswith(".."):
                    os.chdir(os.path.join(snap, rel))
                tmp = os.path.join(snap, ".tmp")
                os.makedirs(tmp, exist_ok=True)
                os.environ["TMPDIR"] = tmp
                tempfile.tempdir = tmp
                res = run_cell(ns, code, timeout)
                res["events"] = list(events)
                data = json.dumps(res, default=str).encode()
            except BaseException as e:  # noqa: BLE001
                data = json.dumps({"stdout": "", "value": None, "error": f"read-only call failed: {e!r}",
                                   "exec_s": 0, "reads_names": []}).encode()
            try:
                view = memoryview(data)
                while view:
                    n = os.write(w, view)
                    view = view[n:]
            finally:
                if snap:
                    shutil.rmtree(snap, ignore_errors=True)
                os._exit(0)
        os.close(w)
        chunks = []
        deadline = time.time() + timeout + 15
        while True:
            left = deadline - time.time()
            if left <= 0:
                os.kill(pid, signal.SIGKILL)
                break
            ready, _, _ = select.select([r], [], [], left)
            if not ready:
                continue
            b = os.read(r, 1 << 20)
            if not b:
                break
            chunks.append(b)
        os.close(r)
        _, status = os.waitpid(pid, 0)
        if chunks:
            try:
                return json.loads(b"".join(chunks))
            except json.JSONDecodeError:
                pass
        return {"stdout": "", "value": None, "exec_s": 0, "reads_names": [], "events": [],
                "error": f"the read-only process ended without a result (wait status {status}); the session is unchanged"}

    for line in proto_in:
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            reply({"error": "bad request"})
            continue
        op = msg.get("op")
        if op == "init":
            cwd = msg["cwd"]
            os.makedirs(cwd, exist_ok=True)
            os.chdir(cwd)
            try:
                import numpy  # noqa: F401  (imported once, so every arm pays the same start-up)
                import pandas as pd

                pd.set_option("display.max_rows", 60)
                pd.set_option("display.max_columns", 30)
                pd.set_option("display.width", 160)
            except Exception:
                pass
            if msg.get("api"):
                api = make_api(msg["api"], events)
                for k, f in api.items():
                    f._api = True
                    ns[k] = f
                    API_NAMES.add(k)
            reply({"ok": True, "pid": os.getpid()})
        elif op == "exec":
            tag = msg.get("tag") or "?"
            timeout = int(msg.get("timeout") or 300)
            events.clear()
            before = dict(prov)  # which request last changed each name, before this cell
            if msg.get("read_only"):
                res = read_only(msg["code"], timeout)
                measure = 0.0
            else:
                res = run_cell(ns, msg["code"], timeout)
                res["events"] = list(events)
                measure = update_provenance(tag)
            names = res.pop("reads_names", [])
            res["reads"] = {k: before[k] for k in names if k in before}
            res["measure_s"] = round(measure, 4)
            events.clear()
            reply(res)
        elif op == "inspect":
            detail = msg.get("detail", "summary")
            items = [describe(k, v, detail) for k, v in tracked(ns)]
            reply({"names": items})
        elif op == "env":
            reply({"executable": sys.executable, "prefix": sys.prefix, "version": sys.version.split()[0]})
        elif op == "exit":
            reply({"ok": True})
            break
        else:
            reply({"error": f"unknown op {op!r}"})


if __name__ == "__main__":
    main()
