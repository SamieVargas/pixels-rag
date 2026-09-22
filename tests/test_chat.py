"""Part 10: follow-ups rewritten from the last three turns.

    python tests/test_chat.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "evals"))

from core import index as I  # noqa: E402
from core.chat import Session  # noqa: E402
from core.days import load_days, latest_date  # noqa: E402
from stubs import FakeClient, decision, answer  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def main():
    days = load_days(ROOT / "fixtures" / "days.json")
    coll, _ = I.build(I.make_client(), days, name="chat_test", embedding_function=I.HashEmbedding())
    hy = {"field": "exercise", "op": "contains", "value": "hot yoga"}

    print("a session hands the router the prior turns and uses the rewrite")
    fc = FakeClient([
        decision("aggregate", metric="rating", stat="mean", group_by="weekend"), answer("Weekends averaged 3.4 and weekdays 3.0.", [], claims=[]),
        decision("aggregate", filters=[hy], metric="rating", stat="mean", group_by="match",
                 query="What was my average rating on hot yoga days versus other days?"),
        answer("Hot yoga days averaged 3.3 against 3.1.", [], claims=[]),
    ])
    s = Session(days=days, collection=coll, client=fc)
    s.ask("What was my average rating on weekends versus weekdays?")
    r = s.ask("And on hot yoga days?")
    router_prompt = fc.requests[2]["messages"][0]["content"]
    check("the first question sends no history", "CONVERSATION SO FAR" not in fc.requests[0]["messages"][0]["content"])
    check("the follow-up sends the prior question and answer", "CONVERSATION SO FAR" in router_prompt and "weekends versus weekdays" in router_prompt and "Weekends averaged" in router_prompt)
    check("the follow-up took the router's plan", r["route"] == "aggregate" and r["plan"]["filters"] == [hy] and r["plan"]["query"].startswith("What was my average rating on hot yoga"))
    check("two turns are kept as four messages", len(s.history()) == 4)

    print("history is capped at the last three turns")
    replies = []
    for i in range(5):
        replies += [decision("unanswerable")]
    fc = FakeClient(replies)
    s = Session(days=days, collection=coll, client=fc, keep=3)
    for i in range(5):
        s.ask(f"question {i}")
    check("only three turns remain", len(s.history()) == 6 and s.history()[0]["text"] == "question 2")
    check("the router saw only those three", "question 0" not in fc.requests[4]["messages"][0]["content"] and "question 2" in fc.requests[4]["messages"][0]["content"])

    print("rewrite=False asks alone")
    fc = FakeClient([decision("unanswerable"), decision("unanswerable")])
    s = Session(days=days, collection=coll, client=fc)
    s.ask("first")
    s.ask("second", rewrite=False)
    check("no history block when rewriting is off", "CONVERSATION SO FAR" not in fc.requests[1]["messages"][0]["content"])

    print("the runner's follow-up table (stubbed router)")
    import run
    golden = run.load_golden(ROOT / "evals" / "golden.jsonl", follow_ups=True)
    check("two follow-ups carry history and plans", len(golden) == 2 and all(g["history"] and g["plan"] for g in golden))
    fc = FakeClient([
        golden[0]["plan"], answer("Hot yoga days averaged 3.3 against 3.1.", [], claims=[]),
        decision("filter", filters=[hy]), answer("Some days.", [], claims=[]),
        golden[1]["plan"], answer("Eight days in June.", golden[1]["expected_dates"][:1]),
        decision("semantic", query="only in June"), answer("Unclear.", [], claims=[], unanswerable=True),
    ])
    with tempfile.TemporaryDirectory() as tmp:
        code = run.main(["--followups", "--embedding", "hash", "--out", tmp], client=fc)
        files = os.listdir(tmp)
        md = Path(tmp, next(f for f in files if f.endswith("-followups.md"))).read_text(encoding="utf-8")
        check("exit 0 and four rows", code == 0 and md.count("| H0") == 4)
        check("the golden-plan arm reads as a match and the bare arm does not", "| H01 | with history | aggregate | ✓" in md and "| H01 | alone | filter" in md)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
