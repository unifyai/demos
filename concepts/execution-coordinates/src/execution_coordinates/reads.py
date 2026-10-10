"""Which names a cell reads from the session: a scope-aware static analysis.

The worker records, for every name a cell might read, which request last
changed its value; its own test of "might read" over-approximates (a loop
variable rebound before use counts). The analysis intersects that record with
this sharper test, which follows the cell's blocks: a name counts as read from
the session only where the cell has not surely bound it itself (loop and
comprehension variables inside their loop, names bound on every branch of an
if, function parameters, `with ... as` names, and so on).
"""

from __future__ import annotations

import ast
import warnings


def _targets(node) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _args(a: ast.arguments) -> set[str]:
    names = [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs]
    if a.vararg:
        names.append(a.vararg.arg)
    if a.kwarg:
        names.append(a.kwarg.arg)
    return set(names)


def session_reads(code: str) -> set[str]:
    try:
        with warnings.catch_warnings():  # model code with "\." in a plain string warns when parsed
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(code)
    except SyntaxError:
        return set()
    reads: set[str] = set()

    def expr(node, bound: set[str]) -> None:
        if node is None:
            return
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            local = set(bound)
            for gen in node.generators:
                expr(gen.iter, local)
                local |= _targets(gen.target)
                for cond in gen.ifs:
                    expr(cond, local)
            if isinstance(node, ast.DictComp):
                expr(node.key, local)
                expr(node.value, local)
            else:
                expr(node.elt, local)
            return
        if isinstance(node, ast.Lambda):
            for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
                expr(d, bound)
            expr(node.body, set(bound) | _args(node.args))
            return
        if isinstance(node, ast.NamedExpr):
            expr(node.value, bound)
            bound.add(node.target.id)
            return
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in bound:
                reads.add(node.id)
            return
        for child in ast.iter_child_nodes(node):
            expr(child, bound)

    def store_target(t, bound: set[str]) -> None:
        # x = ... binds x; df["a"] = ... and obj.attr = ... read df and obj
        if isinstance(t, ast.Name):
            bound.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                store_target(e, bound)
        elif isinstance(t, ast.Starred):
            store_target(t.value, bound)
        else:
            expr(t, bound)

    def block(stmts, bound: set[str]) -> None:
        for s in stmts:
            stmt(s, bound)

    def stmt(s, bound: set[str]) -> None:
        if isinstance(s, ast.Assign):
            expr(s.value, bound)
            for t in s.targets:
                store_target(t, bound)
        elif isinstance(s, ast.AugAssign):
            if isinstance(s.target, ast.Name):
                if s.target.id not in bound:
                    reads.add(s.target.id)
            else:
                expr(s.target, bound)
            expr(s.value, bound)
            store_target(s.target, bound)
        elif isinstance(s, ast.AnnAssign):
            expr(s.value, bound)
            store_target(s.target, bound)
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            for a in s.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in s.decorator_list + s.args.defaults + [d for d in s.args.kw_defaults if d is not None]:
                expr(d, bound)
            bound.add(s.name)
            block(s.body, set(bound) | _args(s.args))
        elif isinstance(s, ast.ClassDef):
            for d in s.decorator_list + s.bases:
                expr(d, bound)
            bound.add(s.name)
            block(s.body, set(bound))
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            expr(s.iter, bound)
            block(s.body, set(bound) | _targets(s.target))
            block(s.orelse, set(bound))
        elif isinstance(s, ast.While):
            expr(s.test, bound)
            block(s.body, set(bound))
            block(s.orelse, set(bound))
        elif isinstance(s, ast.If):
            expr(s.test, bound)
            b1, b2 = set(bound), set(bound)
            block(s.body, b1)
            block(s.orelse, b2)
            bound |= b1 & b2  # bound on every branch
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                expr(it.context_expr, bound)
                if it.optional_vars is not None:
                    store_target(it.optional_vars, bound)
            block(s.body, bound)
        elif isinstance(s, ast.Try) or (hasattr(ast, "TryStar") and isinstance(s, ast.TryStar)):
            block(s.body, set(bound))
            for h in s.handlers:
                expr(h.type, bound)
                block(h.body, set(bound) | ({h.name} if h.name else set()))
            block(s.orelse, set(bound))
            block(s.finalbody, bound)
        elif isinstance(s, (ast.Global, ast.Nonlocal, ast.Pass, ast.Break, ast.Continue)):
            pass
        elif isinstance(s, ast.Delete):
            for t in s.targets:
                if not isinstance(t, ast.Name):
                    expr(t, bound)
        else:
            for child in ast.iter_child_nodes(s):
                if isinstance(child, ast.stmt):
                    stmt(child, bound)
                else:
                    expr(child, bound)

    block(tree.body, set())
    return reads


def refined_reads(tool_calls: list[dict]) -> list[tuple[str, str]]:
    """(name, tag) pairs a request's cells read from their session, recorded
    by the worker and confirmed by the scope-aware analysis."""
    out = []
    for t in tool_calls:
        if t.get("name") != "run_python" or not t.get("reads"):
            continue
        keep = session_reads(t.get("code") or "")
        out.extend((n, tag) for n, tag in t["reads"].items() if n in keep)
    return out
