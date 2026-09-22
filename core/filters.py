"""Metadata filters over day records, in code. The router names a field, an
operator and a value; this decides which days match. Unknown fields raise, so
a hallucinated field is an error rather than an empty result."""

from core.contracts import FIELDS, OPS

TRUE_WORDS = ("true", "yes", "1", "on", "done")


def _bool(v) -> bool:
    return v if isinstance(v, bool) else str(v).strip().lower() in TRUE_WORDS


def matches(day: dict, f: dict) -> bool:
    field, op, value = f["field"], f["op"], f["value"]
    if field not in FIELDS:
        raise ValueError(f"unknown field {field!r}")
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}")
    typ = FIELDS[field][0]
    have = day.get(field)
    if typ == "number":
        if have is None:
            return False
        want = float(value)
        return {"<": have < want, "<=": have <= want, ">": have > want, ">=": have >= want,
                "==": have == want, "!=": have != want, "contains": False}[op]
    if typ == "boolean":
        want = _bool(value)
        return (have == want) if op in ("==", "contains") else (have != want) if op == "!=" else False
    have_s, want_s = str(have or "").lower(), str(value).lower()
    if op == "contains":
        return want_s in have_s
    if op == "==":
        return have_s == want_s
    if op == "!=":
        return have_s != want_s
    return False


def in_range(day: dict, date_range) -> bool:
    if not date_range:
        return True
    start, end = date_range if isinstance(date_range, (tuple, list)) else (date_range.get("start"), date_range.get("end"))
    return (not start or day["date"] >= start) and (not end or day["date"] <= end)


def apply(days: list[dict], filters=(), date_range=None) -> list[dict]:
    """Days that satisfy every filter and sit inside the range, sorted by date."""
    out = [d for d in days if in_range(d, date_range) and all(matches(d, f) for f in (filters or []))]
    return sorted(out, key=lambda d: d["date"])
