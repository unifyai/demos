"""The scope-aware read analysis used to confirm the worker's record."""

from execution_coordinates.reads import refined_reads, session_reads


def test_loop_and_comprehension_variables_are_not_session_reads():
    assert session_reads("for c in cols:\n    print(df[c])") == {"cols", "print", "df"}
    assert session_reads("[x * k for x in d if x]") == {"k", "d"}
    assert session_reads("{a: b for a, b in items.items()}") == {"items"}


def test_names_bound_on_every_branch_are_not_read_later():
    assert "x" not in session_reads("if a:\n    x = 1\nelse:\n    x = 2\nprint(x)")
    assert "x" in session_reads("if a:\n    x = 1\nprint(x)")
    assert "df" in session_reads("if 'df' not in globals():\n    df = load()\nuse(df)")


def test_functions_withs_and_assignments():
    assert session_reads("def g(y, w=v):\n    return y + z\ng(1)") == {"v", "z"}
    assert session_reads("with open(f) as fh:\n    data = fh.read()\nprint(data)") == {"open", "f", "print"}
    assert session_reads("df['a'] = 1") == {"df"}
    assert session_reads("x += 1") == {"x"}
    assert session_reads("x = 1\nx += 1") == set()
    assert session_reads("lambda q: q + r") == {"r"}
    assert session_reads("d = df.copy()\nd['b'] = d.a") == {"df"}
    assert session_reads("try:\n    v = 1\nexcept ValueError as e:\n    print(e)\nprint(v)") == {"ValueError", "print", "v"}
    assert session_reads("import pandas as pd\npd.read_csv('x')") == set()
    assert session_reads("def broken(:") == set()


def test_refined_reads_intersects_the_record():
    calls = [{"name": "run_python", "code": "for c in cols:\n    print(df[c])",
              "reads": {"c": "b#1", "cols": "a#1", "df": "b#1"}},
             {"name": "final_answer"}]
    assert refined_reads(calls) == [("cols", "a#1"), ("df", "b#1")]
