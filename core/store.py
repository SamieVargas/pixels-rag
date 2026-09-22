"""An index that stays current. Ingest upserts by date so a re-run is
idempotent; the collection records which chunk template and which embedding
model built it, and a mismatch with the code triggers a full rebuild with a
warning rather than a silent mix of two formats in one index."""

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

from core import index as I
from core.days import normalize_rows

# Bump when chunk.py's template changes meaning; the hash below catches edits
# nobody bumped for.
CHUNK_TEMPLATE_VERSION = "day@v1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_template_hash() -> str:
    src = (Path(__file__).resolve().parent.parent / "chunk.py").read_bytes()
    return hashlib.sha256(src).hexdigest()[:12]


def stamp(embedding_model: str = EMBEDDING_MODEL) -> dict:
    return {"chunk_template_version": CHUNK_TEMPLATE_VERSION, "chunk_template_hash": chunk_template_hash(), "embedding_model": embedding_model}


def _days_path(db: Path) -> Path:
    return Path(db) / "days.json"


def read_rows(db: Path) -> list[dict]:
    p = _days_path(db)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))["rows"]


def write_rows(db: Path, rows: list[dict]) -> None:
    Path(db).mkdir(parents=True, exist_ok=True)
    _days_path(db).write_text(json.dumps({"rows": sorted(rows, key=lambda r: r["date"])}, indent=1), encoding="utf-8")


def version_mismatch(collection, embedding_model: str = EMBEDDING_MODEL) -> list[str]:
    """What differs between the collection's stamp and the running code."""
    md = collection.metadata or {}
    want = stamp(embedding_model)
    return [f"{k}: index has {md.get(k)!r}, code has {v!r}" for k, v in want.items() if md.get(k) != v]


def rebuild(db: Path, rows: list[dict], *, embedding_function=None, embedding_model: str = EMBEDDING_MODEL):
    """Full build from `rows`, stamped. Returns (collection, day records)."""
    days = normalize_rows(rows)
    write_rows(db, rows)
    client = I.make_client(str(db))
    coll, _ = I.build(client, days, embedding_function=embedding_function, extra_metadata=stamp(embedding_model))
    return coll, days


def ingest(db: Path, new_rows: list[dict], *, embedding_function=None, embedding_model: str = EMBEDDING_MODEL) -> dict:
    """Upsert `new_rows` into the index by date. Idempotent: the same rows
    twice leave the same index. A version mismatch rebuilds everything."""
    db = Path(db)
    existing = {r["date"]: r for r in read_rows(db)}
    client = I.make_client(str(db))
    try:
        coll = I.load(client, embedding_function=embedding_function)
    except Exception:
        coll = None
    report = {"upserted": 0, "new": 0, "updated": 0, "rebuilt": False, "reasons": []}
    if coll is not None:
        report["reasons"] = version_mismatch(coll, embedding_model)
    if coll is None or report["reasons"]:
        merged = {**existing, **{r["date"]: r for r in new_rows}}
        coll, _ = rebuild(db, list(merged.values()), embedding_function=embedding_function, embedding_model=embedding_model)
        report.update(rebuilt=True, upserted=len(merged), new=len([d for d in merged if d not in existing]), updated=len([d for d in new_rows if r_date(d) in existing]))
        return report
    days = normalize_rows(new_rows)
    if days:
        coll.upsert(ids=[d["date"] for d in days], documents=[d["text"] for d in days], metadatas=[I.metadata_for(d) for d in days])
    for r in new_rows:
        key = r_date(r)
        if key in existing:
            report["updated"] += 1
        else:
            report["new"] += 1
        existing[key] = r
    report["upserted"] = len(days)
    write_rows(db, list(existing.values()))
    return report


def r_date(row) -> str:
    return row["date"] if isinstance(row, str) else row["date"]


def status(db: Path, source_rows=None, *, embedding_function=None, embedding_model: str = EMBEDDING_MODEL) -> dict:
    """How current the index is: newest indexed day, newest source day, the
    number of logged days the index is behind, and the version stamp."""
    db = Path(db)
    rows = read_rows(db)
    out = {"indexed_days": len(rows), "newest_indexed": max((r["date"] for r in rows), default=None),
           "newest_source": None, "days_behind": None, "mismatch": [], "stamp": None}
    try:
        coll = I.load(I.make_client(str(db)), embedding_function=embedding_function)
        out["stamp"] = {k: (coll.metadata or {}).get(k) for k in ("chunk_template_version", "chunk_template_hash", "embedding_model")}
        out["mismatch"] = version_mismatch(coll, embedding_model)
        out["indexed_chunks"] = coll.count()
    except Exception:
        out["stamp"] = None
    if source_rows is not None:
        have = {r["date"] for r in rows}
        src = {r["date"] for r in source_rows}
        out["newest_source"] = max(src, default=None)
        out["days_behind"] = len(src - have)
        out["missing_dates"] = sorted(src - have)[:10]
    return out


def since_days(since: str, today=None) -> int:
    """The `days` argument the export endpoint needs to reach back to `since`."""
    today = today or date.today()
    return max((today - date.fromisoformat(since)).days + 1, 1)
