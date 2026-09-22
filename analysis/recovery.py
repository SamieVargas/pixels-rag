"""The headline finding, reproduced deterministically.

The claim that hot yoga plus walking beat everything else for sleep and
recovery came from asking the model. This computes it from the day records:
mean sleep score and body battery on the days after each habit against all
other days, n per group, and a seeded bootstrap interval on the difference.

    python analysis/recovery.py                         # the fixture
    python analysis/recovery.py --days chroma_db/days.json   # your export
"""

import argparse
import random
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.days import load_days  # noqa: E402

METRICS = ("sleep_score", "body_battery")
BOOTSTRAP = 2000
SEED = 11


def tags(day: dict) -> set[str]:
    out = {t.strip().lower() for t in day["exercise"].split(",") if t.strip()}
    for h in ("meditation", "journaling", "reading", "gratitude", "sunlight", "alcohol", "late_meals"):
        if day.get(h):
            out.add(h.replace("_", " "))
    return out


def habits_present(days: list[dict], min_n: int = 3) -> list[str]:
    counts = {}
    for d in days:
        for t in tags(d):
            counts[t] = counts.get(t, 0) + 1
    return [t for t, n in sorted(counts.items(), key=lambda p: (-p[1], p[0])) if n >= min_n]


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def bootstrap_ci(a: list[float], b: list[float], n: int = BOOTSTRAP, seed: int = SEED) -> tuple[float, float] | None:
    """95% interval on mean(a) - mean(b), resampling each group with replacement."""
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if len(a) < 2 or len(b) < 2:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    diffs.sort()
    return (round(diffs[int(0.025 * n)], 1), round(diffs[int(0.975 * n) - 1], 1))


def compare(days: list[dict], group_fn, metric: str) -> dict:
    """Same-day comparison: the metric on days where group_fn holds, against the rest."""
    hit = [d[metric] for d in days if group_fn(d)]
    rest = [d[metric] for d in days if not group_fn(d)]
    ma, mb = _mean(hit), _mean(rest)
    return {"metric": metric, "n": len([x for x in hit if x is not None]), "n_rest": len([x for x in rest if x is not None]),
            "mean": ma, "mean_rest": mb, "diff": (ma - mb) if ma is not None and mb is not None else None, "ci": bootstrap_ci(hit, rest)}


def next_day_metric(days: list[dict]) -> dict[str, dict]:
    """Recovery is measured the morning after: map each day to the next logged day's record."""
    by = {d["date"]: d for d in days}
    out = {}
    for d in days:
        nxt = date.fromisoformat(d["date"]).toordinal() + 1
        n = by.get(date.fromordinal(nxt).isoformat())
        if n:
            out[d["date"]] = n
    return out


def analyse(days: list[dict]) -> dict:
    """The combination first, then each habit alone; same-day and next-day."""
    combo = lambda d: {"hot yoga", "walking"} <= tags(d)
    rows = []
    groups = [("hot yoga + walking", combo)] + [(h, (lambda h: lambda d: h in tags(d))(h)) for h in habits_present(days)]
    nd = next_day_metric(days)
    for name, fn in groups:
        for metric in METRICS:
            same = compare(days, fn, metric)
            # next day: the metric on the day after a habit day, against the day after any other day
            pairs = [(d, nd[d["date"]]) for d in days if d["date"] in nd]
            hit = [n[metric] for d, n in pairs if fn(d)]
            rest = [n[metric] for d, n in pairs if not fn(d)]
            ma, mb = _mean(hit), _mean(rest)
            nxt = {"n": len([x for x in hit if x is not None]), "mean": ma, "mean_rest": mb,
                   "diff": (ma - mb) if ma is not None and mb is not None else None, "ci": bootstrap_ci(hit, rest)}
            rows.append({"group": name, "metric": metric, "same_day": same, "next_day": nxt})
    return {"days": len(days), "rows": rows}


def f1(x):
    return "n/a" if x is None else f"{x:.1f}"


def fdiff(x, ci):
    if x is None:
        return "n/a"
    s = f"{x:+.1f}"
    return s + (f" [{ci[0]:+.1f}, {ci[1]:+.1f}]" if ci else "")


def render(result: dict, *, when=None, source="fixture") -> str:
    lines = [f"# Recovery by habit · {(when or date.today()).isoformat()} · {result['days']} days · {source}", "",
             "Mean on the habit's days against all other days, with n and a 95% bootstrap interval on the difference. "
             "Next-day columns measure the morning after.", "",
             "| Habit | Metric | n | Same day | Others | Diff [95% CI] | Next day | Others | Diff [95% CI] |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in result["rows"]:
        s, n = r["same_day"], r["next_day"]
        lines.append(f"| {r['group']} | {r['metric'].replace('_', ' ')} | {s['n']} | {f1(s['mean'])} | {f1(s['mean_rest'])} | {fdiff(s['diff'], s['ci'])} | "
                     f"{f1(n['mean'])} | {f1(n['mean_rest'])} | {fdiff(n['diff'], n['ci'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="reproduce the recovery finding from the day records")
    p.add_argument("--days", default=str(ROOT / "fixtures" / "days.json"))
    p.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = p.parse_args(argv)
    days = load_days(args.days)
    source = "fixture" if Path(args.days).resolve() == (ROOT / "fixtures" / "days.json").resolve() else "export"
    table = render(analyse(days), source=source)
    print(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"recovery-{date.today().isoformat()}{'' if source == 'fixture' else '-export'}.md"
    path.write_text(table, encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
