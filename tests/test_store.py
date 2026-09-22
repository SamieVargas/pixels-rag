"""Part 5: the index stays current. Idempotent upserts, a version stamp, a
rebuild on mismatch, and a status that counts the days behind the source.

    python tests/test_store.py
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import index as I  # noqa: E402
from core import store  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


ROWS = json.loads((ROOT / "fixtures" / "days.json").read_text(encoding="utf-8"))["rows"]
EF = I.HashEmbedding()


def main():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        print("first ingest builds and stamps")
        rep = store.ingest(db, ROWS[:100], embedding_function=EF, embedding_model="hash-test", report_dir=None)
        coll = I.load(I.make_client(str(db)), embedding_function=EF)
        check("no index yet means a full build", rep["rebuilt"] and rep["upserted"] == 100 and coll.count() == 100)
        md = coll.metadata or {}
        check("the collection carries the template version, its hash and the model", md.get("chunk_template_version") == store.CHUNK_TEMPLATE_VERSION
              and md.get("chunk_template_hash") == store.chunk_template_hash() and md.get("embedding_model") == "hash-test")

        print("re-running the same rows changes nothing")
        rep = store.ingest(db, ROWS[:100], embedding_function=EF, embedding_model="hash-test", report_dir=None)
        coll = I.load(I.make_client(str(db)), embedding_function=EF)
        check("upsert is idempotent: same count, nothing rebuilt", not rep["rebuilt"] and coll.count() == 100 and rep["updated"] == 100 and rep["new"] == 0)
        check("days.json holds one row per date", len(store.read_rows(db)) == 100)

        print("new days are added by upsert")
        rep = store.ingest(db, ROWS[100:], embedding_function=EF, embedding_model="hash-test", report_dir=None)
        coll = I.load(I.make_client(str(db)), embedding_function=EF)
        check("the tail was appended", rep["new"] == len(ROWS) - 100 and coll.count() == len(ROWS))
        edited = dict(ROWS[0]); edited["mood"] = "Elated"
        store.ingest(db, [edited], embedding_function=EF, embedding_model="hash-test", report_dir=None)
        got = I.load(I.make_client(str(db)), embedding_function=EF).get(ids=[edited["date"]], include=["metadatas"])
        check("an edited day replaces its chunk in place", got["metadatas"][0]["mood"] == "Elated" and I.load(I.make_client(str(db)), embedding_function=EF).count() == len(ROWS))

        print("status")
        st = store.status(db, ROWS + [{**ROWS[-1], "date": "2026-09-01"}, {**ROWS[-1], "date": "2026-09-02"}], embedding_function=EF, embedding_model="hash-test")
        check("days behind counts source days the index lacks", st["days_behind"] == 2 and st["missing_dates"] == ["2026-09-01", "2026-09-02"])
        check("newest dates are reported", st["newest_indexed"] == ROWS[-1]["date"] and st["newest_source"] == "2026-09-02")
        check("a matching stamp reports no mismatch", st["mismatch"] == [])

        print("a template or model change forces a rebuild")
        # The code moves on to a new template version; the index still carries the old one.
        saved = store.CHUNK_TEMPLATE_VERSION
        store.CHUNK_TEMPLATE_VERSION = "day@v2-test"
        try:
            st = store.status(db, None, embedding_function=EF, embedding_model="hash-test")
            check("status names the mismatch", any("chunk_template_version" in m for m in st["mismatch"]))
            rep = store.ingest(db, ROWS[-3:], embedding_function=EF, embedding_model="hash-test", report_dir=None)
            coll = I.load(I.make_client(str(db)), embedding_function=EF)
            check("ingest rebuilds everything and says why", rep["rebuilt"] and any("chunk_template_version" in r for r in rep["reasons"]) and coll.count() == len(ROWS))
            check("the rebuilt collection carries the current stamp", (coll.metadata or {}).get("chunk_template_version") == "day@v2-test")
            check("a second ingest after the rebuild is quiet again", not store.ingest(db, ROWS[-1:], embedding_function=EF, embedding_model="hash-test", report_dir=None)["rebuilt"])
        finally:
            store.CHUNK_TEMPLATE_VERSION = saved
        st = store.status(db, None, embedding_function=EF, embedding_model="other-model")
        check("a different embedding model is a mismatch too", any("embedding_model" in m for m in st["mismatch"]))

        print("since")
        from datetime import date
        check("since_days reaches back to the date inclusive", store.since_days("2026-08-25", today=date(2026, 8, 31)) == 7)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
