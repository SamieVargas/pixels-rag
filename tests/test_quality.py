"""Part 6: data quality at the door.

    python tests/test_quality.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import index as I  # noqa: E402
from core import store  # noqa: E402
from core.quality import IngestError, render_report, validate_rows  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


ROWS = json.loads((ROOT / "fixtures" / "days.json").read_text(encoding="utf-8"))["rows"]


def main():
    print("the fixture passes with only its five unlogged days as warnings")
    rep = validate_rows(ROWS)
    check("ok, no errors", rep["ok"] and not rep["errors"])
    check("five missing days, nothing else", rep["warning_counts"] == {"missing_day": 5})
    check("the report renders", "| missing day | 5 |" in render_report(rep))

    print("each check fires")
    dup = validate_rows(ROWS[:3] + [dict(ROWS[1])])
    check("a duplicate date is an error", not dup["ok"] and dup["errors"][0]["kind"] == "duplicate" and dup["errors"][0]["date"] == ROWS[1]["date"])
    bad = dict(ROWS[0]); bad["sleepScore"] = 140; bad["rhr"] = 20
    rng = validate_rows([bad, ROWS[1]])
    check("out-of-range values are warnings with the field named", {w["detail"].split("=")[0] for w in rng["warnings"] if w["kind"] == "out_of_range"} == {"sleepScore", "rhr"})
    empty = {"date": "2026-05-03", "ratingNum": 3}
    emp = validate_rows([ROWS[0], empty])
    check("a day with nothing beyond date and rating is flagged", any(w["kind"] == "empty_day" and w["date"] == "2026-05-03" for w in emp["warnings"]))
    check("the gap between the two rows is a missing day", any(w["kind"] == "missing_day" and w["date"] == "2026-05-02" for w in emp["warnings"]))
    unparse = validate_rows([{"date": "not a date", "ratingNum": 3}])
    check("an unparseable date is an error", not unparse["ok"] and unparse["errors"][0]["kind"] == "bad_date")

    print("ingest refuses duplicates and writes the report")
    with tempfile.TemporaryDirectory() as tmp:
        db, reports = Path(tmp) / "db", Path(tmp) / "reports"
        try:
            store.ingest(db, ROWS[:5] + [dict(ROWS[2])], embedding_function=I.HashEmbedding(), embedding_model="hash-test", report_dir=reports)
            check("duplicate batch raises IngestError", False)
        except IngestError as e:
            check("duplicate batch raises IngestError", "rows share this date" in str(e))
        check("nothing was indexed", not (db / "days.json").exists())
        check("the report was written beside the results", any(p.name.startswith("ingest-") for p in reports.iterdir()))
        rep = store.ingest(db, ROWS[:5], embedding_function=I.HashEmbedding(), embedding_model="hash-test", report_dir=reports)
        check("a clean batch ingests and carries its quality report", rep["upserted"] == 5 and rep["quality"]["ok"])

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
