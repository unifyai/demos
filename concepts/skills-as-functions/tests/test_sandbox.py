from skills.api import Api
from skills.sandbox import LINE_LIMIT, Namespace
from skills.world import generate


def ns(size="realistic", version=1, **kw):
    return Namespace(Api(generate(1, size), version, size, **kw).functions())


def test_submit_ends_the_cell_and_cannot_be_swallowed():
    n = ns()
    r = n.run("try:\n    submit(42)\nexcept Exception:\n    print('swallowed')\nprint('after')")
    assert r.submitted and r.answer == 42 and r.output == "" and r.error is None
    r = n.run("try:\n    submit('x')\nexcept BaseException:\n    pass\nprint('not reached?')")
    # a bare BaseException handler can catch it; the loop treats the cell as not submitted
    assert not r.submitted


def test_output_is_captured_and_variables_persist():
    n = ns()
    assert n.run("x = today()\nprint(x)").output == "2026-09-15\n"
    r = n.run("print(x[:4])")
    assert r.output == "2026\n" and n.format(r) == "2026"
    assert n.format(n.run("y = 1")) == "(no output)"


def test_imports_are_limited():
    n = ns()
    assert n.run("import os").error_type == "ImportError"
    assert n.run("import re as r2\nprint(r2.sub('a', 'b', 'aa'))").output == "bb\n"
    assert n.run("from datetime import date\nprint(date(2026, 1, 2).isoformat())").output == "2026-01-02\n"
    assert n.run("open('/etc/passwd')").error_type == "NameError"


def test_runaway_loops_are_stopped():
    n = ns()
    r = n.run("while True:\n    pass")
    assert r.error_type == "LineLimit" and f"{LINE_LIMIT:,}" in r.error
    n2 = ns(call_limit=50)
    r = n2.run("while True:\n    try:\n        today()\n    except Exception:\n        pass")
    assert r.error_type == "OpLimit"


def test_traceback_shows_only_the_models_code_and_the_library():
    n = ns(version=2)
    n.load_library({"find_id": "def find_id(name):\n    \"\"\"Find a contact id.\"\"\"\n    return search_contacts(name)['results'][0]['id']\n"})
    r = n.run("def task():\n    return find_id('x')\nprint(task())")
    assert r.error_type == "ApiError" and "410 Gone" in r.error
    assert '"<cell 1>"' in r.error and '"<library:find_id>"' in r.error and "api.py" not in r.error
    assert "return search_contacts(name)['results'][0]['id']" in r.error
    assert r.library_in_traceback == ["find_id"]
    assert n.library_calls == {"find_id"}


def test_task_functions_keeps_final_versions_only():
    n = ns()
    n.run("def a():\n    \"\"\"one\"\"\"\n    return 1\ndef b():\n    return 2")
    n.run("def a():\n    \"\"\"two\"\"\"\n    return 3")
    n.run("raise ValueError('x')\ndef c():\n    return 4")  # c never defined
    n.run("@staticmethod\ndef d():\n    return 5") if False else None
    fs = n.task_functions()
    assert set(fs) == {"a", "b"} and '"""two"""' in fs["a"]


def test_syntax_errors_are_reported():
    r = ns().run("def f(:\n  pass")
    assert r.error_type == "SyntaxError" and "line 1" in r.error


def test_long_output_is_cut():
    r = ns().run("print('x' * 10000)")
    assert "characters cut" in r.output and len(r.output) < 4200


def test_a_final_bare_expression_is_shown_as_in_a_notebook():
    n = ns()
    assert n.run("x = 2\nx + 3").output == "5\n"
    assert n.run("'a' + 'b'").output == "'ab'\n"
    assert n.run("None").output == ""
    assert n.run("print('p')").output == "p\n"  # print is not shown twice
    r = n.run("y = 1\n1/0")
    assert r.error_type == "ZeroDivisionError" and '"<cell 5>", line 2' in r.error
    assert n.run("today()").output == "'2026-09-15'\n"
    r = n.run("submit(7)")
    assert r.submitted and r.answer == 7
    n.run("def f():\n    return 9")
    assert n.task_functions()["f"].startswith("def f")  # defs in a cell with no final expression still recorded
