"""Part 9: the recovery finding computed from the records, deterministically.

    python tests/test_recovery.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from core.days import load_days  # noqa: E402
import recovery  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def main():
    days = load_days(ROOT / "fixtures" / "days.json")
    res = recovery.analyse(days)
    combo = [r for r in res["rows"] if r["group"] == "hot yoga + walking"]
    sleep = next(r for r in combo if r["metric"] == "sleep_score")["same_day"]
    print("the planted pattern is found by code, not by the model")
    check("hot yoga + walking days sleep better than the rest", sleep["diff"] > 5 and sleep["n"] == 9)
    check("the interval excludes zero", sleep["ci"] and sleep["ci"][0] > 0)
    check("n per group adds up to the corpus", sleep["n"] + sleep["n_rest"] == len(days))
    print("each habit alone is in the table")
    groups = {r["group"] for r in res["rows"]}
    check("hot yoga, walking and meditation appear on their own", {"hot yoga", "walking", "meditation"} <= groups)
    print("deterministic")
    a = recovery.render(recovery.analyse(days), when=__import__("datetime").date(2026, 9, 22))
    b = recovery.render(recovery.analyse(days), when=__import__("datetime").date(2026, 9, 22))
    check("two runs render the same table", a == b and "| hot yoga + walking | sleep score |" in a)
    print("bootstrap")
    check("too-small groups give no interval", recovery.bootstrap_ci([1.0], [2.0, 3.0]) is None)
    lo, hi = recovery.bootstrap_ci([70.0] * 20, [60.0] * 20)
    check("identical groups give a tight interval around the difference", lo == hi == 10.0)

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
