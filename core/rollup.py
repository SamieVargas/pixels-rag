"""Weekly rollup chunks, computed deterministically from the day records, for
the chunk-granularity ablation (arm B indexes these beside the day chunks).
A week chunk covers the days it summarizes, so retrieving one counts as
retrieving those days when recall is scored."""

from collections import Counter
from datetime import date, timedelta

from core.aggregate import fmt


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def week_chunks(days: list[dict]) -> list[dict]:
    """One chunk per Monday-to-Sunday week that has at least one logged day."""
    weeks: dict[str, list[dict]] = {}
    for d in days:
        dt = date.fromisoformat(d["date"])
        monday = (dt - timedelta(days=dt.weekday())).isoformat()
        weeks.setdefault(monday, []).append(d)
    out = []
    for monday, rows in sorted(weeks.items()):
        end = (date.fromisoformat(monday) + timedelta(days=6)).isoformat()
        rows.sort(key=lambda r: r["date"])
        ex = Counter()
        for r in rows:
            for tag in (t.strip().lower() for t in r["exercise"].split(",") if t.strip()):
                ex[tag] += 1
        best = max(rows, key=lambda r: (r["rating"] or 0, r["sleep_score"] or 0))
        worst_sleep = min((r for r in rows if r["sleep_score"] is not None), key=lambda r: r["sleep_score"], default=None)
        parts = [
            f"Week of {monday} to {end}: {len(rows)} days logged.",
            f"Mean rating {fmt(_mean(r['rating'] for r in rows))}/5, mean sleep score {fmt(_mean(r['sleep_score'] for r in rows))}, "
            f"mean body battery {fmt(_mean(r['body_battery'] for r in rows))}, mean steps {fmt(_mean(r['steps'] for r in rows))}.",
        ]
        if ex:
            parts.append("Exercise: " + ", ".join(f"{k} on {n} day{'s' if n != 1 else ''}" for k, n in ex.most_common()) + ".")
        habits = {"meditation": sum(r["meditation"] for r in rows), "alcohol": sum(r["alcohol"] for r in rows), "late meals": sum(r["late_meals"] for r in rows)}
        parts.append("Habits: " + ", ".join(f"{k} on {n} days" for k, n in habits.items() if n) + "." if any(habits.values()) else "Habits: none logged.")
        parts.append(f"Best day: {best['date']} (rating {fmt(best['rating'])}/5, mood {best['mood'] or 'unlogged'}).")
        if worst_sleep:
            parts.append(f"Lowest sleep: {worst_sleep['date']} (score {fmt(worst_sleep['sleep_score'])}).")
        moods = Counter(r["mood"] for r in rows if r["mood"])
        if moods:
            parts.append("Moods: " + ", ".join(f"{m} x{n}" for m, n in moods.most_common(3)) + ".")
        symptoms = sorted({r["symptoms"] for r in rows if r["symptoms"] and r["symptoms"] != "None"})
        if symptoms:
            parts.append("Symptoms logged: " + ", ".join(symptoms) + ".")
        out.append({
            "id": f"week:{monday}",
            "text": "\n".join(parts),
            "metadata": {"date": monday, "week_end": end, "level": "week", "n_days": len(rows)},
            "days": [r["date"] for r in rows],
        })
    return out
