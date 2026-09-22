"""Day records: the raw export row, normalized once into the field names the
router, the filters and the aggregator use. The chunk text is v1's, unchanged,
so the embedding input is exactly what v1 embedded."""

import json
from datetime import date, datetime
from pathlib import Path

from chunk import row_to_chunk

from core.contracts import FIELDS, NUMERIC_FIELDS, BOOLEAN_FIELDS

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# Wearable zeros mean "not logged", never a measured zero.
ZERO_IS_MISSING = ("sleep_score", "sleep_hours", "body_battery", "hrv", "rhr", "water", "garmin_stress")


def parse_date(value) -> str:
    """Accept 2026-06-14, 6/14/2026, 06/14/26 or an ISO datetime; return YYYY-MM-DD."""
    s = str(value).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date {value!r}")


def _num(v):
    if v in (None, "", False):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _text(v):
    return "" if v is None else str(v).strip()


def _flag(v):
    return bool(v) and str(v).lower() not in ("false", "0", "no", "")


def normalize_row(row: dict) -> dict:
    """One raw row -> one day record with typed fields plus the v1 chunk text."""
    iso = parse_date(row["date"])
    d = date.fromisoformat(iso)
    rec = {
        "date": iso,
        "weekday": WEEKDAYS[d.weekday()],
        "weekend": d.weekday() >= 5,
        "rating": _num(row.get("ratingNum")),
        "sleep_score": _num(row.get("sleepScore")),
        "sleep_hours": _num(row.get("sleepTime")),
        "body_battery": _num(row.get("bodyBattery")),
        "hrv": _num(row.get("overnightHRV")),
        "rhr": _num(row.get("rhr")),
        "steps": _num(row.get("steps")),
        "water": _num(row.get("water")),
        "garmin_stress": _num(row.get("garminStress")),
        "mood": _text(row.get("mood")),
        "energy": _text(row.get("energyLevel")),
        "anxiety": _text(row.get("anxiety")),
        "stress": _text(row.get("stress")),
        "brain_fog": _text(row.get("brainFog")),
        "sleep_quality": _text(row.get("sleepQuality")),
        "exercise": _text(row.get("exercise")),
        "activity": _text(row.get("activityLevel")),
        "symptoms": _text(row.get("physicalSymptoms")),
        "hashimotos": _text(row.get("hashimotosSymptoms")),
        "diet": _text(row.get("dietQuality")),
        "caffeine": _text(row.get("caffeine")),
        "social": _text(row.get("socialInteractions")),
        "weather": _text(row.get("weather")),
        "productivity": _text(row.get("productivity")),
        "yoga": _flag(row.get("yoga")),
        "meditation": _flag(row.get("meditation")),
        "journaling": _flag(row.get("journal")),
        "reading": _flag(row.get("reading")),
        "gratitude": _flag(row.get("gratitude")),
        "sunlight": _flag(row.get("sunlight")),
        "alcohol": _flag(row.get("alcohol")),
        "fast_food": _flag(row.get("fastFood")),
        "late_meals": _flag(row.get("lateMeals")),
        "positive_event": _flag(row.get("hasPositiveEvent")),
        "negative_event": _flag(row.get("hasNegativeEvent")),
    }
    for k in ZERO_IS_MISSING:
        if rec[k] == 0:
            rec[k] = None
    rec["text"] = row_to_chunk({**row, "date": iso})
    return rec


def normalize_rows(rows: list[dict]) -> list[dict]:
    days = [normalize_row(r) for r in rows if r.get("date")]
    days.sort(key=lambda r: r["date"])
    return days


def load_days(path) -> list[dict]:
    """Read an export file ({"rows": [...]} or a bare list) into day records."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) else data
    return normalize_rows(rows)


def metadata_for(day: dict) -> dict:
    """The scalar fields ChromaDB can filter on. None is dropped: Chroma
    metadata holds str, int, float and bool only."""
    md = {"date": day["date"], "level": "day"}
    for name in FIELDS:
        v = day.get(name)
        if v is None or v == "":
            continue
        md[name] = v
    return md


def latest_date(days: list[dict]) -> date:
    return date.fromisoformat(max(d["date"] for d in days)) if days else date.today()
