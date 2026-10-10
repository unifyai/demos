"""The mechanism: modes, named sessions, environments, inspection, errors."""

import asyncio
import json
import os
import subprocess
import sys

import pytest

from execution_coordinates.executor import Executor


def run(coro):
    return asyncio.run(coro)


def make_api(tmp_path, load_s=0.0):
    d = tmp_path / "data"
    d.mkdir(exist_ok=True)
    (d / "acme.csv").write_text("txn_id,amount\nT1,1.5\nT2,2.5\n")
    (d / "acme_risk.csv").write_text("customer_id,risk\nC1,0.9\nC2,0.1\n")
    return {"data_dir": str(d), "load_s": load_s, "score_s": 0.0, "clients": ["acme"]}


async def with_ex(tmp_path, fn, **kw):
    ex = Executor(tmp_path / "root", **kw)
    try:
        return await fn(ex)
    finally:
        await ex.close()


def val(res):
    assert res.get("error") is None, res.get("error")
    return res.get("value")


# ───────────────────────── names from every block persist ─────────────────────
BLOCKS = '''
import math as m
from collections import Counter as C
if True:
    in_if = 1
for i in range(3):
    in_for = i
while True:
    in_while = 2
    break
try:
    in_try = 3
    raise ValueError
except ValueError as e:
    in_except = 4
finally:
    in_finally = 5
with open("f.txt", "w") as fh:
    in_with = 6
def fn():
    global via_global
    via_global = 7
    local_only = 8
    return local_only
fn()
class K:
    pass
if (walrus := 9):
    pass
squares = [j * j for j in range(3)]
'''


def test_names_bound_in_every_block_persist(tmp_path):
    async def go(ex):
        assert (await ex.run(BLOCKS, "stateful", "s"))["error"] is None
        return val(await ex.run(
            "(m.pi > 3, C('aab')['a'], in_if, in_for, i, in_while, in_try, in_except, in_finally, in_with,"
            " via_global, fn(), K.__name__, walrus, squares, 'j' in globals(), 'local_only' in globals(),"
            " 'e' in globals())", "stateful", "s"))

    out = run(with_ex(tmp_path, go))
    # e is unbound at the end of an except block, as in Python; j stays local to the comprehension
    assert out == "(True, 2, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 'K', 9, [0, 1, 4], False, False, False)"


def test_repl_value_print_and_errors(tmp_path):
    async def go(ex):
        a = await ex.run("x = 41\nprint('hello')\nx + 1", "stateful", "s")
        b = await ex.run("x = 5", "stateful", "s")
        c = await ex.run("None", "stateful", "s")
        d = await ex.run("def f():\n    return 1 / 0\nf()", "stateful", "s")
        e = await ex.run("def broken(:\n", "stateful", "s")
        f = await ex.run("x", "stateful", "s")
        g = await ex.run("globals()['x'], sorted(k for k in dir() if not k.startswith('_'))[:3]", "stateful", "s")
        return a, b, c, d, e, f, g

    a, b, c, d, e, f, g = run(with_ex(tmp_path, go))
    assert a["value"] == "42" and a["stdout"] == "hello\n" and a["error"] is None
    assert b["value"] is None and c["value"] is None
    assert "ZeroDivisionError" in d["error"] and 'File "<cell>", line 2, in f' in d["error"]
    assert "worker.py" not in d["error"]
    assert "SyntaxError" in e["error"]
    assert f["value"] == "5"  # a failing cell does not undo earlier work
    assert g["value"].startswith("(5, [")  # globals() and dir() see the session


def test_subprocess_output_and_stdin_do_not_touch_the_protocol(tmp_path):
    async def go(ex):
        a = await ex.run("import os\nos.system('echo from-shell')\nprint('after')", "stateful", "s")
        b = await ex.run("try:\n    input()\nexcept EOFError:\n    print('eof')", "stateful", "s")
        c = await ex.run("1 + 1", "stateful", "s")
        return a, b, c

    a, b, c = run(with_ex(tmp_path, go))
    assert "from-shell" in a["stdout"] and "after" in a["stdout"]
    assert b["stdout"] == "eof\n" and c["value"] == "2"


# ─────────────────────────────── read-only ────────────────────────────────────
SETUP = '''
import os, random, math
import pandas as pd
df = pd.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
rates = {"USD": 0.9}
items = [1, 2]
keep = "original"
random.seed(1)
os.environ["MODE"] = "base"
with open("data.txt", "w") as f:
    f.write("base")
pd.set_option("display.max_rows", 60)
'''

MUTATE = '''
df["new"] = 1
df.loc[df["a"] == 1, "s"] = "CHANGED"
df.drop(index=2, inplace=True)
df.sort_values("a", ascending=False, inplace=True)
rates["USD"] = 99.0
items.append(3)
keep = "rebound"
added = "new name"
del math
random.random()
os.environ["MODE"] = "changed"
pd.set_option("display.max_rows", 5)
os.chdir("/")
print("mutated", len(df))
(len(df), rates["USD"])
'''

MUTATE_FILES = '''
import os, tempfile
with open("data.txt", "w") as f:
    f.write("changed")
with open("created.txt", "w") as f:
    f.write("new")
with open(os.path.join(tempfile.gettempdir(), "tmpfile.txt"), "w") as f:
    f.write("tmp")
os.remove("data.txt") if False else None
'''

CHECK = '''
import os as _os, tempfile as _tf
(df.shape, list(df.columns), df["s"].tolist(), rates, items, keep, "added" in globals(), "math" in globals(),
 _os.environ["MODE"], open("data.txt").read(), _os.path.exists("created.txt"),
 _os.path.exists(_os.path.join(_tf.gettempdir(), "tmpfile.txt")), pd.get_option("display.max_rows"))
'''


def test_read_only_changes_nothing_in_the_session(tmp_path):
    async def go(ex):
        assert (await ex.run(SETUP, "stateful", "s"))["error"] is None
        before = val(await ex.run(CHECK, "stateful", "s"))
        cwd_before = val(await ex.run("_os.getcwd()", "stateful", "s"))
        ro = await ex.run(MUTATE, "read_only", "s")
        ro_files = await ex.run(MUTATE_FILES, "read_only", "s")
        after = val(await ex.run(CHECK, "stateful", "s"))
        cwd_after = val(await ex.run("_os.getcwd()", "stateful", "s"))
        # random state: the next draw in the session is the one after seed(1), as if read-only never ran
        draw = val(await ex.run("random.random()", "stateful", "s"))
        return before, ro, ro_files, after, cwd_before, cwd_after, draw

    before, ro, ro_files, after, cwd_before, cwd_after, draw = run(with_ex(tmp_path, go))
    assert ro["error"] is None and ro["value"] == "(2, 99.0)" and "mutated 2" in ro["stdout"]
    assert ro_files["error"] is None
    assert after == before
    assert "'CHANGED'" not in after and "'new'" not in after and "99" not in after
    assert cwd_after == cwd_before
    import random

    random.seed(1)
    assert draw == repr(random.random())


def test_read_only_sees_the_session_and_its_files(tmp_path):
    async def go(ex):
        await ex.run(SETUP, "stateful", "s")
        return await ex.run("print(open('data.txt').read()); float(df['a'].sum() * rates['USD'])", "read_only", "s")

    res = run(with_ex(tmp_path, go))
    assert res["error"] is None and res["stdout"] == "base\n" and res["value"] == "5.4"


def test_read_only_crash_or_timeout_leaves_session_intact(tmp_path):
    async def go(ex):
        await ex.run("x = [1, 2, 3]", "stateful", "s")
        crash = await ex.run("import os\nos._exit(3)", "read_only", "s")
        kill = await ex.run("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)", "read_only", "s")
        slow = await ex.run("while True:\n    try:\n        pass\n    except Exception:\n        pass", "read_only", "s")
        still = val(await ex.run("x", "stateful", "s"))
        return crash, kill, slow, still

    crash, kill, slow, still = run(with_ex(tmp_path, go, timeout_s=2))
    assert "without a result" in crash["error"] and "without a result" in kill["error"]
    assert "CellTimeout" in slow["error"]
    assert still == "[1, 2, 3]"


def test_stateful_timeout_interrupts_the_cell_and_keeps_the_session(tmp_path):
    async def go(ex):
        await ex.run("x = 1", "stateful", "s")
        slow = await ex.run("y = 0\nwhile True:\n    try:\n        y += 1\n    except Exception:\n        pass",
                            "stateful", "s")
        return slow, val(await ex.run("x, y > 0", "stateful", "s"))

    slow, out = run(with_ex(tmp_path, go, timeout_s=1))
    assert "CellTimeout" in slow["error"] and out == "(1, True)"


def test_system_exit_does_not_end_the_session(tmp_path):
    async def go(ex):
        await ex.run("x = 1", "stateful", "s")
        a = await ex.run("import sys\nsys.exit(2)", "stateful", "s")
        return a, val(await ex.run("x", "stateful", "s"))

    a, x = run(with_ex(tmp_path, go))
    assert "SystemExit" in a["error"] and x == "1"


# ───────────────────────── stateless and named sessions ──────────────────────
def test_stateless_is_fresh_every_time(tmp_path):
    async def go(ex):
        await ex.run("x = 1\nopen('f.txt', 'w').write('s')", "stateful", "s")
        a = await ex.run("y = 2\nopen('g.txt', 'w').write('t')\n'x' in globals()", "stateless")
        b = await ex.run("import os\n('y' in globals(), os.path.exists('g.txt'), os.path.exists('f.txt'))",
                         "stateless")
        c = await ex.run("'y' in globals()", "stateful", "s")
        return a, b, c

    a, b, c = run(with_ex(tmp_path, go))
    assert a["value"] == "False" and a["session"] is None
    assert b["value"] == "(False, False, False)"
    assert c["value"] == "False"


def test_named_sessions_are_isolated(tmp_path):
    async def go(ex):
        await ex.run("df = 'acme data'\ndef helper():\n    return 'acme'", "stateful", "acme")
        await ex.run("df = 'globex data'\ndef helper():\n    return 'globex'", "stateful", "globex")
        a = val(await ex.run("df, helper()", "stateful", "acme"))
        b = val(await ex.run("df, helper()", "stateful", "globex"))
        return a, b

    a, b = run(with_ex(tmp_path, go))
    assert a == "('acme data', 'acme')" and b == "('globex data', 'globex')"


def test_invalid_coordinates_explain_themselves(tmp_path):
    async def go(ex):
        await ex.run("x = 1", "stateful", "base")
        return [
            await ex.run("1", "read_only", "missing"),
            await ex.run("1", "read_only"),
            await ex.run("1", "stateless", "base"),
            await ex.run("1", "stateful"),
            await ex.run("1", "sideways", "base"),
            await ex.run("1", "stateful", "bad name!"),
            await ex.run("", "stateful", "base"),
            await ex.run("1", "stateful", "base", env="nope"),
        ]

    errs = run(with_ex(tmp_path, go))
    for e in errs:
        assert e["error_type"] == "validation" and e["error"] and e["suggestion"], e
    assert "base" in errs[0]["suggestion"]  # names the sessions that exist


def test_session_cap(tmp_path):
    async def go(ex):
        await ex.run("1", "stateful", "a")
        await ex.run("1", "stateful", "b")
        return await ex.run("1", "stateful", "c"), await ex.run("2", "stateful", "a")

    capped, ok = run(with_ex(tmp_path, go, max_sessions=2))
    assert capped["error_type"] == "validation" and ok["value"] == "2"


# ─────────────────────────────── inspection ───────────────────────────────────
def test_list_and_inspect_see_every_session_and_every_name(tmp_path):
    async def go(ex):
        await ex.run("import pandas as pd\nif True:\n    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})\n"
                     "rates = {'USD': 0.9}\ndef helper(x, y=1):\n    return x", "stateful", "acme")
        await ex.run("for i in range(2):\n    total = i", "stateful", "globex")
        listing = await ex.list_sessions()
        acme = await ex.inspect("acme")
        globex = await ex.inspect("globex")
        missing = await ex.inspect("nope")
        return listing, acme, globex, missing

    listing, acme, globex, missing = run(with_ex(tmp_path, go))
    assert [s["session"] for s in listing["sessions"]] == ["acme", "globex"]
    assert "df: DataFrame 2x2" in listing["sessions"][0]["names"]
    assert "rates: dict(1)" in listing["sessions"][0]["names"]
    assert "helper(x, y=1)" in listing["sessions"][0]["names"]
    assert "total: int = 1" in listing["sessions"][1]["names"] and "i: int = 1" in listing["sessions"][1]["names"]
    names = {d["name"]: d for d in acme["names"]}
    assert names["df"]["shape"] == [2, 2] and set(names["df"]["columns"]) == {"a", "b"} and "head" in names["df"]
    assert "pd" not in names  # modules are not state worth listing
    assert {d["name"] for d in globex["names"]} == {"i", "total"}
    assert missing["error_type"] == "validation" and "acme" in missing["suggestion"]


# ─────────────────────────── the environment axis ────────────────────────────
def test_sessions_run_in_the_environment_they_name(tmp_path):
    venv = tmp_path / "other-env"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    other = str(venv / "bin" / "python")

    async def go(ex):
        await ex.run("x = 'default'", "stateful", "s")
        await ex.run("import sys\nx = sys.prefix", "stateful", "s", env="other")
        a = await ex.environment("s")
        b = await ex.environment("s", env="other")
        c = val(await ex.run("x", "stateful", "s"))
        d = val(await ex.run("x", "stateful", "s", env="other"))
        e = val(await ex.run("import sys; sys.prefix", "stateless", env="other"))
        return a, b, c, d, e

    a, b, c, d, e = run(with_ex(tmp_path, go, envs={"other": other}))
    assert a["prefix"] != b["prefix"] and b["prefix"] == str(venv)
    assert c == "'default'" and d == repr(str(venv)) and e == repr(str(venv))


# ──────────────────────── the API and provenance record ──────────────────────
def test_api_calls_are_recorded_in_every_mode(tmp_path):
    api = make_api(tmp_path)

    async def go(ex):
        a = await ex.run("df = load_export('acme')\nlen(df)", "stateful", "s")
        b = await ex.run("load_export('acme')['amount'].tolist()", "read_only", "s")
        c = await ex.run("float(score_customers('acme')['risk'].max())", "stateless")
        d = await ex.run("load_export('other')", "stateful", "s")
        return a, b, c, d

    a, b, c, d = run(with_ex(tmp_path, go, api=api))
    assert a["value"] == "2" and [e["fn"] for e in a["events"]] == ["load_export"]
    assert b["value"] == "['1.5', '2.5']" and [e["fn"] for e in b["events"]] == ["load_export"]
    assert c["value"] == "0.9" and [e["fn"] for e in c["events"]] == ["score_customers"]
    assert "unknown client id" in d["error"] and d["events"] == []


def test_reads_report_which_request_last_changed_each_name(tmp_path):
    async def go(ex):
        await ex.run("import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})\nrates = {'x': 1}", "stateful", "s", tag="A1")
        await ex.run("df['b'] = 1", "stateful", "s", tag="B1")  # in place
        await ex.run("rates2 = rates", "stateful", "s", tag="A2")
        await ex.run("rates['x'] = 2", "read_only", "s", tag="B2")  # read-only: no change
        r = await ex.run("total = df['a'].sum() * rates['x']\ntotal", "stateful", "s", tag="A3")
        r2 = await ex.run("total + 1", "stateful", "s", tag="A4")
        return r, r2

    r, r2 = run(with_ex(tmp_path, go))
    assert r["reads"] == {"df": "B1", "rates": "A1"} and r["value"] == "np.int64(3)"
    assert r2["reads"] == {"total": "A3"}


def test_rebinding_to_an_equal_value_keeps_its_provenance(tmp_path):
    async def go(ex):
        await ex.run("rates = {'x': 1}\ndef helper(v):\n    return v * 2", "stateful", "s", tag="A1")
        await ex.run("rates = {'x': 1}\ndef helper(v):\n    return v * 2", "stateful", "s", tag="B1")  # same values
        same = await ex.run("helper(rates['x'])", "stateful", "s", tag="A2")
        await ex.run("def helper(v):\n    return v * 3", "stateful", "s", tag="B2")  # a different helper
        diff = await ex.run("helper(rates['x'])", "stateful", "s", tag="A3")
        return same, diff

    same, diff = run(with_ex(tmp_path, go))
    assert same["reads"] == {"helper": "A1", "rates": "A1"}
    assert diff["reads"] == {"helper": "B2", "rates": "A1"} and diff["value"] == "3"
