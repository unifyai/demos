"""The idea's mechanism: one executor whose every call is a point in a small
space of coordinates.

    mode     "stateful"  run in the named session, which is kept
             "stateless" run in a fresh interpreter, discarded afterwards
             "read_only" run against a snapshot of the named session (a fork of
                         its process); nothing the code does reaches the session
    session  a name; a stateful call to a new name starts a new session
    env      which interpreter (virtual environment) the session runs in

A session is identified by (env, name) and is one worker process
(`worker.py`). Every arm in the experiment runs its code through this class;
the baselines fix the coordinates instead of letting the model choose them.
Invalid combinations come back as structured errors with a suggestion, never
as exceptions.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

WORKER = Path(__file__).with_name("worker.py")
MODES = ("stateful", "stateless", "read_only")
NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


def invalid(error: str, suggestion: str) -> dict[str, Any]:
    return {"error": error, "error_type": "validation", "suggestion": suggestion}


class WorkerDied(RuntimeError):
    pass


class Worker:
    """One Python process speaking the worker protocol."""

    def __init__(self, python: str, cwd: Path, api: dict | None, log: Path | None = None):
        self.python, self.cwd, self.api, self.log = python, cwd, api, log
        self.proc: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        self.cwd.mkdir(parents=True, exist_ok=True)
        (self.cwd / ".tmp").mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k not in ("OPENROUTER_API_KEY",)}
        env["TMPDIR"] = str(self.cwd / ".tmp")
        env["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        env["PYTHONHASHSEED"] = "0"
        stderr = open(self.log, "ab") if self.log else asyncio.subprocess.DEVNULL
        self.proc = await asyncio.create_subprocess_exec(
            self.python, "-I", str(WORKER),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=stderr,
            cwd=str(self.cwd), env=env, limit=1 << 26,
        )
        if self.log:
            stderr.close()
        await self.call({"op": "init", "cwd": str(self.cwd), "api": self.api}, timeout=120)

    async def call(self, msg: dict, timeout: float) -> dict:
        async with self.lock:
            if self.proc is None or self.proc.returncode is not None:
                raise WorkerDied("the session's process is not running")
            assert self.proc.stdin and self.proc.stdout
            self.proc.stdin.write((json.dumps(msg) + "\n").encode())
            await self.proc.stdin.drain()
            try:
                line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
            except asyncio.TimeoutError:
                self.kill()
                raise WorkerDied(f"no reply within {timeout:.0f} s; the process was killed") from None
            if not line:
                raise WorkerDied(f"the process exited (code {self.proc.returncode})")
            return json.loads(line)

    def kill(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                await self.call({"op": "exit"}, timeout=5)
            except Exception:
                pass
            try:
                await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.kill()
                await self.proc.wait()


@dataclass
class Session:
    name: str
    env: str
    worker: Worker
    order: int
    calls: int = 0
    last_used: int = 0
    created_by: str | None = None
    meta: dict = field(default_factory=dict)


class Executor:
    """Runs code at a chosen (mode, session, env).

    envs maps an environment's name to its Python executable; "default" is
    this interpreter. api configures the functions every interpreter gets
    (see worker.make_api), or None for none.
    """

    def __init__(self, root: str | os.PathLike, api: dict | None = None, envs: dict[str, str] | None = None,
                 timeout_s: int = 300, max_sessions: int = 24):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.api = api
        self.envs = {"default": sys.executable, **(envs or {})}
        self.timeout_s = timeout_s
        self.max_sessions = max_sessions
        self.sessions: dict[tuple[str, str], Session] = {}
        self._ids = itertools.count(1)
        self._clock = itertools.count(1)
        self._fresh: set[Worker] = set()

    # ───────────────────────────── running code ─────────────────────────────
    async def run(self, code: str, mode: str, session: str | None = None, env: str = "default",
                  tag: str | None = None) -> dict[str, Any]:
        """Run code; returns stdout, value, error, timings, the API calls it
        made (events) and which request last changed each name it read."""
        if mode not in MODES:
            return invalid(f"Unknown mode {mode!r}.", "Use mode='stateful', 'stateless' or 'read_only'.")
        if not isinstance(code, str) or not code.strip():
            return invalid("No code given.", "Pass the Python code to run in `code`.")
        if env not in self.envs:
            return invalid(f"Unknown environment {env!r}.", f"Use one of: {', '.join(sorted(self.envs))}.")
        if mode == "stateless":
            if session:
                return invalid("A stateless call runs in a fresh interpreter and cannot name a session.",
                               "Leave out `session`, or use mode='stateful' to run in that session "
                               "or mode='read_only' to read it without changing it.")
            return await self._run_fresh(code, env, tag)
        if not session:
            return invalid(f"mode={mode!r} needs a session name.",
                           "Give `session` a name (a stateful call to a new name starts a new session), "
                           "or use mode='stateless' for a one-off run.")
        if not NAME_RE.match(session):
            return invalid(f"Invalid session name {session!r}.",
                           "Use 1 to 64 letters, digits, '_', '-' or '.'.")
        key = (env, session)
        s = self.sessions.get(key)
        if mode == "read_only" and s is None:
            have = ", ".join(n for (e, n) in self.sessions if e == env) or "none"
            return invalid(f"read_only needs an existing session, and {session!r} does not exist.",
                           f"Sessions that exist: {have}. Use mode='stateful' to create {session!r}, "
                           "or list_sessions() to see what each session holds.")
        created = False
        t0 = time.perf_counter()
        if s is None:
            if len(self.sessions) >= self.max_sessions:
                return invalid(f"There are already {self.max_sessions} sessions.",
                               "Run in one of the existing sessions (list_sessions() shows them).")
            w = Worker(self.envs[env], self.root / f"s{next(self._ids)}-{_slug(session)}", self.api,
                       self.root / "workers.log")
            await w.start()
            s = Session(session, env, w, order=next(self._clock), created_by=tag)
            self.sessions[key] = s
            created = True
        try:
            res = await s.worker.call({"op": "exec", "code": code, "tag": tag or "?",
                                       "read_only": mode == "read_only", "timeout": self.timeout_s},
                                      timeout=self.timeout_s + 60)
        except WorkerDied as e:
            self.sessions.pop(key, None)
            return {"error": f"The session's process died ({e}); session {session!r} and its state are gone.",
                    "error_type": "session_died", "mode": mode, "session": session, "env": env,
                    "seconds": round(time.perf_counter() - t0, 3), "events": [], "reads": {}, "measure_s": 0.0}
        s.calls += 1
        s.last_used = next(self._clock)
        res.update(mode=mode, session=session, env=env, created=created,
                   seconds=round(time.perf_counter() - t0, 3))
        return res

    async def _run_fresh(self, code: str, env: str, tag: str | None) -> dict[str, Any]:
        t0 = time.perf_counter()
        w = Worker(self.envs[env], self.root / f"fresh{next(self._ids)}", self.api, self.root / "workers.log")
        self._fresh.add(w)
        try:
            await w.start()
            res = await w.call({"op": "exec", "code": code, "tag": tag or "?", "read_only": False,
                                "timeout": self.timeout_s}, timeout=self.timeout_s + 60)
        except WorkerDied as e:
            res = {"error": f"The interpreter died ({e}).", "error_type": "process_died", "events": [], "reads": {},
                   "measure_s": 0.0}
        finally:
            await w.close()
            self._fresh.discard(w)
            shutil.rmtree(w.cwd, ignore_errors=True)
        res.update(mode="stateless", session=None, env=env, created=False,
                   seconds=round(time.perf_counter() - t0, 3))
        res["reads"] = {}  # nothing to read in a fresh interpreter
        return res

    # ───────────────────────────── looking around ───────────────────────────
    async def list_sessions(self) -> dict[str, Any]:
        out = []
        for (env, name), s in sorted(self.sessions.items(), key=lambda kv: kv[1].order):
            try:
                names = (await s.worker.call({"op": "inspect", "detail": "summary"}, timeout=60))["names"]
            except WorkerDied:
                names = None
            row: dict[str, Any] = {"session": name, "calls": s.calls,
                                   "names": [_brief(d) for d in names] if names is not None else "unavailable"}
            if len(self.envs) > 1:
                row["env"] = env
            out.append(row)
        return {"sessions": out}

    async def inspect(self, session: str, env: str = "default") -> dict[str, Any]:
        if not session:
            return invalid("No session given.", "Pass the name of a session; list_sessions() shows them.")
        s = self.sessions.get((env, session))
        if s is None:
            have = ", ".join(n for (e, n) in self.sessions if e == env) or "none"
            return invalid(f"Session {session!r} does not exist.", f"Sessions that exist: {have}.")
        try:
            names = (await s.worker.call({"op": "inspect", "detail": "full"}, timeout=60))["names"]
        except WorkerDied as e:
            return {"error": f"The session's process is gone ({e}).", "error_type": "session_died"}
        return {"session": session, "calls": s.calls, "names": names}

    async def environment(self, session: str | None = None, env: str = "default") -> dict[str, Any]:
        """Which interpreter a session (or a fresh one in env) runs in. For tests."""
        if session:
            return await self.sessions[(env, session)].worker.call({"op": "env"}, timeout=30)
        w = Worker(self.envs[env], self.root / f"probe{next(self._ids)}", None)
        await w.start()
        try:
            return await w.call({"op": "env"}, timeout=30)
        finally:
            await w.close()
            shutil.rmtree(w.cwd, ignore_errors=True)

    async def close(self) -> None:
        for s in list(self.sessions.values()):
            await s.worker.close()
        for w in list(self._fresh):
            await w.close()
        self.sessions.clear()


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)[:40]


def _brief(d: dict) -> str:
    if "shape" in d:
        return f"{d['name']}: {d['type']} {d['shape'][0]}x{d['shape'][1]}"
    if "length" in d:
        return f"{d['name']}: {d['type']}({d['length']})"
    if "signature" in d:
        return f"{d['signature']}"
    if "repr" in d and len(d["repr"]) <= 40:
        return f"{d['name']}: {d['type']} = {d['repr']}"
    return f"{d['name']}: {d['type']}"


def make_tmp_root(prefix: str = "execcoord-") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
