"""Reranking, measured. Retrieve a wider candidate set, score each candidate
against the query with a second model, keep the top k. Off by default; the
eval says whether it earns its latency.

Two rerankers: a local cross-encoder (MS MARCO MiniLM, from
sentence-transformers, everything on the machine) and a lexical one that
needs no model, so the wiring can be tested and run anywhere."""

import re
import time

CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 20
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


class LexicalReranker:
    """Token overlap between the query and the chunk, weighted toward the
    query's rarer words. A baseline, not a contender."""

    name = "lexical"

    def score(self, query: str, texts: list[str]) -> list[float]:
        q = _tokens(query)
        if not q:
            return [0.0] * len(texts)
        docs = [_tokens(t) for t in texts]
        n = len(docs) or 1
        df = {w: sum(1 for d in docs if w in d) for w in q}
        return [sum((1.0 / (1 + df[w])) for w in q if w in d) / len(q) for d in docs]


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model: str = CROSS_ENCODER):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise RuntimeError("the cross-encoder reranker needs sentence-transformers: pip install sentence-transformers") from e
        self.model_name = model
        self.model = CrossEncoder(model)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


def make_reranker(name):
    if name in (None, "", "none"):
        return None
    if name == "lexical":
        return LexicalReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker {name!r}")


def rerank(query: str, chunks: list[dict], reranker, k: int) -> tuple[list[dict], int]:
    """Score the candidates, return the top k in score order and the time it took."""
    if not chunks:
        return [], 0
    t0 = time.time()
    scores = reranker.score(query, [c["text"] for c in chunks])
    ranked = sorted(zip(scores, range(len(chunks))), key=lambda p: (-p[0], p[1]))
    out = []
    for s, i in ranked[:k]:
        c = dict(chunks[i])
        c["dense_rank"] = i + 1
        c["rerank_score"] = round(float(s), 4)
        out.append(c)
    return out, int((time.time() - t0) * 1000)
