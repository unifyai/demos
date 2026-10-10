"""Hybrid retrieval: BM25 and embedding similarity, fused by reciprocal rank."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOP = set(
    "a an the and or of to for in on at by with from is was were be been are it its this that these those what which who whom "
    "when where how did do does done you your yours i me my we our us they them their he she his her as about any have has had "
    "not no so if than then there here into over under up down out can could would should will just all also more most very "
    "maya please".split()
)


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


class BM25:
    """Okapi BM25 over documents, or over documents already turned into token counts."""

    def __init__(self, docs: list[str] | None = None, k1: float = 1.2, b: float = 0.75, counts: list[Counter[str]] | None = None):
        self.k1, self.b = k1, b
        self.tf = counts if counts is not None else [Counter(tokens(d)) for d in docs or []]
        docs = self.tf
        self.lens = np.array([sum(t.values()) for t in self.tf], dtype=float)
        self.avg = float(self.lens.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = set(tokens(query))
        out = np.zeros(len(self.tf))
        if not len(self.tf):
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lens / max(self.avg, 1e-9))
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + norm[i])
            out[i] = s
        return out


def normalise(vecs: np.ndarray) -> np.ndarray:
    vecs = np.asarray(vecs, dtype=np.float32)
    n = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.maximum(n, 1e-9)


def rrf_order(score_lists: list[np.ndarray], k: int = 60) -> list[int]:
    """Document indices by reciprocal-rank fusion of several score arrays (higher is better)."""
    n = len(score_lists[0])
    fused = np.zeros(n)
    for s in score_lists:
        order = np.argsort(-s, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        fused += 1.0 / (k + ranks + 1)
    return list(np.argsort(-fused, kind="stable"))


def hybrid_order(query: str, docs: list[str], doc_vecs: np.ndarray, q_vec: np.ndarray) -> list[int]:
    bm = BM25(docs).scores(query)
    emb = normalise(doc_vecs) @ normalise(q_vec[None, :])[0]
    return rrf_order([bm, emb])


def fill(order: list[int], lengths: list[int], budget: int, sep: int = 2) -> list[int]:
    """Take items in `order` while they fit in `budget` characters (skipping
    any that would overflow), and return the chosen indices."""
    used, chosen = 0, []
    for i in order:
        need = lengths[i] + (sep if chosen else 0)
        if used + need <= budget:
            chosen.append(i)
            used += need
    return chosen
