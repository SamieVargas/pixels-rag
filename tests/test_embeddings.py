"""Part 8: the embedding-model ablation runs the arms it can and names the
ones it cannot.

    python tests/test_embeddings.py
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from core import index as I  # noqa: E402
from core.days import load_days, latest_date  # noqa: E402
from core.embeddings import ARMS, make_embedding_function, run_arm  # noqa: E402
from core.router import resolve  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def main():
    import run
    days = load_days(ROOT / "fixtures" / "days.json")
    today = latest_date(days)
    golden = run.load_golden(ROOT / "evals" / "golden.jsonl")
    qs = [(g["question"], resolve(g["plan"], today), g["expected_dates"]) for g in golden if g["kind"] == "semantic"][:3]

    print("factory")
    ef, label = make_embedding_function("hash")
    check("the test arm builds", isinstance(ef, I.HashEmbedding) and label == "hash-test")
    check("minilm is Chroma's default (no function object)", make_embedding_function("minilm") == (None, "all-MiniLM-L6-v2"))
    try:
        make_embedding_function("word2vec")
        check("unknown arm raises", False)
    except ValueError:
        check("unknown arm raises", True)
    os.environ.pop("OPENAI_API_KEY", None)
    try:
        make_embedding_function("openai")
        check("the API arm refuses without a key and says what it sends", False)
    except RuntimeError as e:
        check("the API arm refuses without a key and says what it sends", "OPENAI_API_KEY" in str(e) and "sends every chunk" in str(e))

    print("arms")
    row = run_arm("hash", days=days, questions=qs, score=run.score_retrieval)
    check("a runnable arm reports recall, build time and latency", row["ran"] and row["recall5"] is not None and row["build_s"] >= 0 and row["query_ms"] is not None)
    row = run_arm("openai", days=days, questions=qs, score=run.score_retrieval)
    check("an arm that cannot run says why instead of failing", row["ran"] is False and "OPENAI_API_KEY" in row["reason"])

    print("the runner writes the table")
    with tempfile.TemporaryDirectory() as tmp:
        code = run.main(["--offline", "--embedding-ablation", "--arms", "hash,openai", "--out", tmp])
        files = os.listdir(tmp)
        md = Path(tmp, next(f for f in files if f.endswith("-embeddings.md"))).read_text(encoding="utf-8")
        check("exit 0 and both rows present", code == 0 and "| hash |" in md and "| openai |" in md and "not run" in md)
        check("the privacy line is in the table", "sends every chunk's text" in md)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


def test_all():
    """pytest entry point: the checks above, one test per file. A failing
    check makes main() exit 1, which pytest reports as this test failing."""
    main()


if __name__ == "__main__":
    main()
