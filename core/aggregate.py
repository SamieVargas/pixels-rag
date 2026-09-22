"""The aggregate route. No retrieval: the statistic is computed from the day
records with pandas and handed to the model as a table to narrate. The
validator then checks every number in the answer against this table."""

import pandas as pd

from core import filters as F
from core.contracts import STATS, NUMERIC_FIELDS


def _frame(days):
    return pd.DataFrame(days).drop(columns=["text"], errors="ignore")


def _stat(series: pd.Series, stat: str):
    s = series.dropna()
    if stat == "count":
        return int(s.shape[0])
    if s.empty:
        return None
    return {"mean": float(s.mean()), "median": float(s.median()), "min": float(s.min()), "max": float(s.max())}[stat]


def aggregate(days: list[dict], *, metric, stat, group_by=None, filters=(), date_range=None) -> dict:
    """Compute `stat` of `metric` over the days, optionally split by `group_by`.

    group_by is a field name (a boolean or text field) or "match", which splits
    the days into those that satisfy the filters and those that do not. With a
    field group_by, the filters narrow the days first.
    """
    if stat not in STATS:
        raise ValueError(f"unknown stat {stat!r}")
    if metric is not None and metric not in NUMERIC_FIELDS:
        raise ValueError(f"metric {metric!r} is not numeric")
    if stat != "count" and metric is None:
        raise ValueError("a metric is needed for anything but count")
    in_range = [d for d in days if F.in_range(d, date_range)]
    groups = []
    if group_by == "match":
        hit = F.apply(in_range, filters)
        hit_dates = {d["date"] for d in hit}
        rest = [d for d in in_range if d["date"] not in hit_dates]
        label = " and ".join(f"{f['field']} {f['op']} {f['value']}" for f in (filters or [])) or "match"
        for name, rows in ((label, hit), ("all other days", rest)):
            groups.append({"name": name, "n": len(rows), "value": _stat(_frame(rows)[metric] if metric and rows else pd.Series(dtype=float), stat) if rows else (0 if stat == "count" else None), "dates": [d["date"] for d in rows]})
    else:
        narrowed = F.apply(in_range, filters)
        if group_by:
            df = _frame(narrowed) if narrowed else pd.DataFrame(columns=[group_by] + ([metric] if metric else []))
            for key, sub in df.groupby(group_by, dropna=False, sort=True):
                name = {True: f"{group_by}: yes", False: f"{group_by}: no"}.get(key, f"{group_by}: {key}") if isinstance(key, bool) else f"{group_by}: {key}"
                groups.append({"name": name, "n": int(sub.shape[0]), "value": _stat(sub[metric], stat) if metric else int(sub.shape[0]), "dates": sorted(sub["date"].tolist())})
        else:
            df = _frame(narrowed)
            value = _stat(df[metric], stat) if (metric and not df.empty) else (len(narrowed) if stat == "count" else None)
            groups.append({"name": "all matching days", "n": len(narrowed), "value": value, "dates": [d["date"] for d in narrowed]})
    return {"metric": metric, "stat": stat, "group_by": group_by, "filters": list(filters or []), "date_range": date_range, "groups": groups}


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v - round(v)) > 1e-9 else str(int(round(v)))
    return str(v)


def render_table(result: dict) -> str:
    head = f"{result['stat']} of {result['metric']}" if result["metric"] else "count of days"
    lines = [f"| group | days | {head} |", "| --- | --- | --- |"]
    for g in result["groups"]:
        lines.append(f"| {g['name']} | {g['n']} | {fmt(g['value'])} |")
    return "\n".join(lines)
