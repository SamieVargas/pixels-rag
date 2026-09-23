"""Part 11: --explain rows, one per retrieved chunk.

    python tests/test_explain.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from core import index as I  # noqa: E402
from core.days import load_days  # noqa: E402
from core.explain import explain, render  # noqa: E402
from core.pipeline import ask  # noqa: E402
from stubs import FakeClient, decision, answer  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def main():
    days = load_days(ROOT / "fixtures" / "days.json")
    coll, _ = I.build(I.make_client(), days, name="explain_test", embedding_function=I.HashEmbedding())
    f = {"field": "rating", "op": ">=", "value": 4}
    # find the first retrieved day without a model, so the stub answer can cite it
    from core.days import latest_date
    from core.pipeline import gather
    from core.router import resolve
    plan = resolve(decision("semantic", phrase="the first two weeks of June", filters=[f], query="good days"), latest_date(days))
    first = gather("good days in early June?", plan, days=days, collection=coll)["retrieved"][0]["id"]
    fc = FakeClient([decision("semantic", phrase="the first two weeks of June", filters=[f], query="good days"), answer(f"On {first} it was good.", [first])])
    r = ask("good days in early June?", days=days, collection=coll, client=fc)
    exp = explain(r, days)
    check("one row per retrieved chunk", len(exp["rows"]) == len(r["retrieved"]))
    check("every retrieved day matched the filter and sat in the range", all("rating >= 4 ✓" in row["filters"] and row["in_range"] == "✓" for row in exp["rows"]))
    check("the cited chunk is marked", [row["cited"] for row in exp["rows"]].count(True) == 1 and exp["rows"][0]["cited"])
    check("route, query and range are reported", exp["route"] == "semantic" and exp["query"] == "good days" and exp["date_range"] == ("2026-06-01", "2026-06-14"))
    text = render(exp)
    check("the rendering carries the table and the validator line", "| chunk | score |" in text and "validator: passed" in text)

    fc = FakeClient([decision("aggregate", metric="sleep_score", stat="mean"), answer("Mean was 63.", [], claims=[])])
    r = ask("average sleep?", days=days, collection=coll, client=fc)
    exp = explain(r, days)
    check("an aggregate explain lists the covered days with no filters", exp["route"] == "aggregate" and all(row["filters"] == "no filters" for row in exp["rows"]))

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
