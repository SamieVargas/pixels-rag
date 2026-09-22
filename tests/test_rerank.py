"""Part 7: reranking, wired and measured offline.

    python tests/test_rerank.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import index as I  # noqa: E402
from core.days import load_days, latest_date  # noqa: E402
from core.pipeline import gather  # noqa: E402
from core.rerank import LexicalReranker, make_reranker, rerank  # noqa: E402
from core.router import resolve  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


class Reverse:
    """Scores candidates in reverse dense order, so the wiring is visible."""
    name = "reverse"

    def score(self, query, texts):
        return [float(i) for i in range(len(texts))]


def main():
    days = load_days(ROOT / "fixtures" / "days.json")
    coll, _ = I.build(I.make_client(), days, name="rerank_test", embedding_function=I.HashEmbedding())
    today = latest_date(days)

    print("the lexical reranker prefers the chunk that shares the query's words")
    lex = LexicalReranker()
    s = lex.score("migraine severe brain fog", ["Physical symptoms: Migraine\nBrain fog: Severe", "Steps: 9,000\nWeather: Sunny", ""])
    check("overlap scores highest, empty text scores zero", s[0] > s[1] and s[2] == 0.0)
    check("an empty query scores nothing", lex.score("", ["a"]) == [0.0])

    print("rerank keeps the top k in score order and records where each came from")
    chunks = [{"id": f"d{i}", "text": f"chunk {i}"} for i in range(6)]
    top, ms = rerank("q", chunks, Reverse(), 3)
    check("the reverse scorer reorders the dense list", [c["id"] for c in top] == ["d5", "d4", "d3"])
    check("dense rank and rerank score are kept", top[0]["dense_rank"] == 6 and top[0]["rerank_score"] == 5.0 and isinstance(ms, int))
    check("no candidates means no work", rerank("q", [], Reverse(), 3) == ([], 0))

    print("gather widens the candidate set and reranks to top_k")
    plan = resolve({"kind": "semantic", "date_range": {"start": None, "end": None}, "date_phrase": "in June", "filters": [],
                    "aggregate": {"metric": None, "stat": None, "group_by": None}, "rewritten_query": "migraine"}, today)
    plain = gather("migraine?", plan, days=days, collection=coll, top_k=3)
    rer = gather("migraine?", plan, days=days, collection=coll, top_k=3, reranker=Reverse(), candidates=10)
    check("plain retrieval records no rerank time", plain["rerank_ms"] is None and len(plain["retrieved"]) == 3)
    check("reranked retrieval returns top_k from a wider set with timing", len(rer["retrieved"]) == 3 and rer["rerank_ms"] is not None and rer["retrieved"][0]["dense_rank"] == 10)
    check("the evidence uses the reranked order", rer["evidence"].startswith(f"[{rer['retrieved'][0]['id']}]"))

    print("factory")
    check("none means no reranker", make_reranker("none") is None and make_reranker(None) is None)
    check("lexical builds", isinstance(make_reranker("lexical"), LexicalReranker))
    try:
        make_reranker("bm25-magic")
        check("unknown name raises", False)
    except ValueError:
        check("unknown name raises", True)
    try:
        make_reranker("cross-encoder")
        check("cross-encoder builds or explains what it needs", True)
    except RuntimeError as e:
        check("cross-encoder builds or explains what it needs", "sentence-transformers" in str(e))

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
