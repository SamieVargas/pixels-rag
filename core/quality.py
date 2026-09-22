"""Data quality at the door. Retrieval over bad rows produces confident wrong
answers, and ingest is the cheapest place to stop that. Duplicates fail the
ingest; missing days, out-of-range values and empty days are warnings, and
the report is written beside the eval results."""

from datetime import date, timedelta

from chunk import row_to_chunk
from core.days import parse_date

# Field -> (min, max). A value outside is a warning, never silently dropped.
RANGES = {
    "ratingNum": (1, 5), "sleepScore": (0, 100), "sleepTime": (0, 16), "bodyBattery": (0, 100), "bodyBatteryBOD": (0, 100),
    "overnightHRV": (0, 200), "rhr": (30, 120), "garminStress": (0, 100), "steps": (0, 60000), "water": (0, 300), "dayOfPeriod": (0, 45),
}
# A chunk with only these lines says nothing a question could use.
BARE_LINES = ("Date:", "Rating:")


def validate_rows(rows: list[dict], expected_range=None) -> dict:
    """Return a report: errors (duplicates), warnings (missing days,
    out-of-range values, empty days), counts, and ok (no errors)."""
    errors, warnings = [], []
    seen: dict[str, int] = {}
    dated = []
    for r in rows:
        try:
            iso = parse_date(r.get("date"))
        except (ValueError, TypeError):
            errors.append({"kind": "bad_date", "date": str(r.get("date")), "detail": "unparseable date"})
            continue
        seen[iso] = seen.get(iso, 0) + 1
        dated.append((iso, r))
    for iso, n in sorted(seen.items()):
        if n > 1:
            errors.append({"kind": "duplicate", "date": iso, "detail": f"{n} rows share this date"})
    if dated:
        lo, hi = (expected_range or (min(seen), max(seen)))
        cur, end = date.fromisoformat(lo), date.fromisoformat(hi)
        missing = []
        while cur <= end:
            if cur.isoformat() not in seen:
                missing.append(cur.isoformat())
            cur += timedelta(days=1)
        for m in missing:
            warnings.append({"kind": "missing_day", "date": m, "detail": "no row for this day"})
    for iso, r in dated:
        for field, (lo, hi) in RANGES.items():
            v = r.get(field)
            if v in (None, "", False):
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                warnings.append({"kind": "not_a_number", "date": iso, "detail": f"{field}={v!r}"})
                continue
            if x < lo or x > hi:
                warnings.append({"kind": "out_of_range", "date": iso, "detail": f"{field}={x:g}, expected {lo} to {hi}"})
        lines = [l for l in row_to_chunk({**r, "date": iso}).splitlines() if not l.startswith(BARE_LINES)]
        if not lines:
            warnings.append({"kind": "empty_day", "date": iso, "detail": "the chunk holds nothing beyond the date and the rating"})
    kinds = {}
    for w in warnings:
        kinds[w["kind"]] = kinds.get(w["kind"], 0) + 1
    return {"ok": not errors, "rows": len(rows), "errors": errors, "warnings": warnings, "warning_counts": kinds,
            "range": (min(seen), max(seen)) if seen else None}


def render_report(report: dict, *, when=None) -> str:
    lines = [f"# Ingest report · {(when or date.today()).isoformat()}", "",
             f"{report['rows']} rows" + (f", {report['range'][0]} to {report['range'][1]}" if report["range"] else "") + f" · {'OK' if report['ok'] else 'FAILED'}", ""]
    lines += ["| Check | Count |", "| --- | --- |", f"| Errors (duplicate or unparseable dates) | {len(report['errors'])} |"]
    for k in ("missing_day", "out_of_range", "not_a_number", "empty_day"):
        lines.append(f"| {k.replace('_', ' ')} | {report['warning_counts'].get(k, 0)} |")
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- {e['date']}: {e['detail']}" for e in report["errors"]]
    if report["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {w['date']}: {w['detail']}" for w in report["warnings"][:60]]
        if len(report["warnings"]) > 60:
            lines.append(f"- ({len(report['warnings']) - 60} more)")
    return "\n".join(lines) + "\n"


class IngestError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("; ".join(f"{e['date']}: {e['detail']}" for e in report["errors"]))
