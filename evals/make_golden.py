"""The golden question set, written against the committed fixture.

The questions, their kinds and their plans are hand-written here, before any
retrieval run. For the filter and aggregate kinds the expected dates and facts
are computed from the fixture by the plain-Python code below, on purpose not
core/aggregate.py or core/filters.py, so the tests can check the pipeline's
answers against labels it did not produce. Regenerate with

    python evals/make_golden.py
"""

import json
import statistics
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = json.loads((ROOT / "fixtures" / "days.json").read_text(encoding="utf-8"))["rows"]
by = {r["date"]: r for r in rows}
DATES = sorted(by)
TODAY = date.fromisoformat(DATES[-1])


def month(m):
    return [d for d in DATES if d[5:7] == f"{m:02d}"]


def weekend(d):
    return date.fromisoformat(d).weekday() >= 5


def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}" if abs(v - round(v)) > 1e-9 else str(int(round(v)))
    return str(v)


def mean(vals):
    vals = [v for v in vals if v]
    return sum(vals) / len(vals)


def plan(kind, *, phrase=None, filters=(), metric=None, stat=None, group_by=None, query=None):
    return {"kind": kind, "date_range": {"start": None, "end": None}, "date_phrase": phrase, "filters": list(filters),
            "aggregate": {"metric": metric, "stat": stat, "group_by": group_by}, "rewritten_query": query}


def f(field, op, value):
    return {"field": field, "op": op, "value": value}


last_two_weeks = [d for d in DATES if d >= (TODAY - timedelta(days=13)).isoformat()]
hot_yoga = [d for d in DATES if "Hot yoga" in by[d]["exercise"]]
not_hot_yoga = [d for d in DATES if d not in hot_yoga]
wk = [d for d in DATES if weekend(d)]
wd = [d for d in DATES if not weekend(d)]

G = [
    # semantic: the answer needs the days read
    ("S01", "semantic", "What happened on the day I had a migraine in June?",
     plan("semantic", phrase="in June", query="migraine headache bad day"), ["2026-06-14"], ["migraine"],
     "one planted day; the chunk names the symptom"),
    ("S02", "semantic", "Describe my best day in July.",
     plan("semantic", phrase="in July", filters=[f("rating", "==", 5)], query="best day happy positive"), ["2026-07-04"], ["positive"],
     "hybrid: rating filter plus dense search; July has one five-star day"),
    ("S03", "semantic", "What did recovery look like the day after my trail run in August?",
     plan("semantic", phrase="in August", query="recovery after trail running sleep body battery"), ["2026-08-09", "2026-08-10"], ["88"],
     "the hard one: 'the day after' is adjacency, which similarity cannot do; the run day is findable, the morning after is not"),
    ("S04", "semantic", "What was going on the day my anxiety was worst in May?",
     plan("semantic", phrase="in May", query="severe anxiety high stress catastrophizing"), ["2026-05-27"], ["Severe"],
     "one planted day in May with severe anxiety"),
    ("S05", "semantic", "Tell me about the night in May when I drank and ate late.",
     plan("semantic", phrase="in May", filters=[f("alcohol", "==", True), f("late_meals", "==", True)], query="alcohol late meals poor sleep"),
     ["2026-05-20"], ["52"], "hybrid with two boolean filters; one May day matches both"),
    ("S06", "semantic", "What was the week of June 8 like?",
     plan("semantic", phrase="the week of June 8", query="how the week went overall"),
     [d for d in DATES if "2026-06-08" <= d <= "2026-06-14"], [], "a whole week; the chunking ablation is aimed at this shape"),
    ("S07", "semantic", "What did the days with a Hashimoto's flare look like?",
     plan("semantic", query="Hashimoto's symptoms fatigue joint pain"), ["2026-06-22", "2026-08-15"], ["joint pain"],
     "two planted days; the phrase is in the chunk text and in no metadata field the router would filter on"),
    # filter: the answer is a list of days
    ("F01", "filter", "Which days was my sleep score under 60?",
     plan("filter", filters=[f("sleep_score", "<", 60)]), [d for d in DATES if by[d]["sleepScore"] < 60], [], "threshold"),
    ("F02", "filter", "List the days I did hot yoga in July.",
     plan("filter", phrase="in July", filters=[f("exercise", "contains", "hot yoga")]), [d for d in month(7) if d in hot_yoga], [], "contains plus a month"),
    ("F03", "filter", "Which days in the first two weeks of June had a rating of 4 or higher?",
     plan("filter", phrase="the first two weeks of June", filters=[f("rating", ">=", 4)]),
     [d for d in DATES if "2026-06-01" <= d <= "2026-06-14" and by[d]["ratingNum"] >= 4], [], "relative range resolved in code"),
    ("F04", "filter", "On which days did I have alcohol?",
     plan("filter", filters=[f("alcohol", "==", True)]), [d for d in DATES if by[d]["alcohol"]], [], "boolean"),
    ("F05", "filter", "Which days had a body battery over 80?",
     plan("filter", filters=[f("body_battery", ">", 80)]), [d for d in DATES if by[d]["bodyBattery"] > 80], [], "one match"),
    ("F06", "filter", "What days did I walk more than 12,000 steps in August?",
     plan("filter", phrase="in August", filters=[f("steps", ">", 12000)]), [d for d in month(8) if by[d]["steps"] > 12000], [], "a number with a comma in the question"),
    ("F07", "filter", "Which weekends in May did I rate 5?",
     plan("filter", phrase="in May", filters=[f("weekend", "==", True), f("rating", "==", 5)]),
     [d for d in month(5) if weekend(d) and by[d]["ratingNum"] == 5], [], "two filters, one derived field"),
    # aggregate: no retrieval, code computes the number
    ("A01", "aggregate", "What was my average sleep score in June?",
     plan("aggregate", phrase="in June", metric="sleep_score", stat="mean"), month(6),
     [fmt(mean(by[d]["sleepScore"] for d in month(6)))], "mean over a month"),
    ("A02", "aggregate", "Did I sleep better on hot yoga days?",
     plan("aggregate", filters=[f("exercise", "contains", "hot yoga")], metric="sleep_score", stat="mean", group_by="match"), DATES,
     [fmt(mean(by[d]["sleepScore"] for d in hot_yoga)), fmt(mean(by[d]["sleepScore"] for d in not_hot_yoga))],
     "the comparison that is really an aggregation; both group means must appear"),
    ("A03", "aggregate", "How many days did I meditate in July?",
     plan("aggregate", phrase="in July", filters=[f("meditation", "==", True)], stat="count"),
     [d for d in month(7) if by[d]["meditation"]], [str(sum(1 for d in month(7) if by[d]["meditation"]))], "count"),
    ("A04", "aggregate", "What was my best body battery in May?",
     plan("aggregate", phrase="in May", metric="body_battery", stat="max"), month(5),
     [fmt(float(max(by[d]["bodyBattery"] for d in month(5))))], "max"),
    ("A05", "aggregate", "What was my average rating on weekends versus weekdays?",
     plan("aggregate", metric="rating", stat="mean", group_by="weekend"), DATES,
     [fmt(mean(by[d]["ratingNum"] for d in wk)), fmt(mean(by[d]["ratingNum"] for d in wd))], "group by a derived boolean"),
    ("A06", "aggregate", "How many days had a sleep score under 60 in August?",
     plan("aggregate", phrase="in August", filters=[f("sleep_score", "<", 60)], stat="count"),
     [d for d in month(8) if by[d]["sleepScore"] < 60], [str(sum(1 for d in month(8) if by[d]["sleepScore"] < 60))], "count with a threshold"),
    ("A07", "aggregate", "What was my median steps over the last two weeks?",
     plan("aggregate", phrase="the last two weeks", metric="steps", stat="median"), last_two_weeks,
     [fmt(float(statistics.median(by[d]["steps"] for d in last_two_weeks)))], "relative phrase resolved against the newest logged day"),
    # unanswerable: never logged
    ("U01", "unanswerable", "What was my blood pressure in June?", plan("unanswerable", phrase="in June"), [], [], "not a field"),
    ("U02", "unanswerable", "What did I eat for breakfast on June 14?", plan("unanswerable", phrase="June 14"), [], [], "meals are not logged"),
    ("U03", "unanswerable", "How many meetings did I have last week?", plan("unanswerable", phrase="last week"), [], [], "not logged"),
    ("U04", "unanswerable", "What was my weight at the end of August?", plan("unanswerable", phrase="end of August"), [], [], "not logged"),
    ("U05", "unanswerable", "Who did I talk to on my best day?", plan("unanswerable"), [], [], "the log has a social rating, never names"),
]

FOLLOW_UPS = [
    ("H01", "aggregate", "And on hot yoga days?",
     [{"role": "user", "text": "What was my average rating on weekends versus weekdays?"},
      {"role": "assistant", "text": "Weekends averaged higher than weekdays."}],
     plan("aggregate", filters=[f("exercise", "contains", "hot yoga")], metric="rating", stat="mean", group_by="match",
          query="What was my average rating on hot yoga days versus other days?"), DATES,
     [fmt(mean(by[d]["ratingNum"] for d in hot_yoga)), fmt(mean(by[d]["ratingNum"] for d in not_hot_yoga))],
     "a follow-up that only makes sense with the prior turn; scored in Part 10"),
    ("H02", "filter", "Only in June?",
     [{"role": "user", "text": "Which days was my sleep score under 60?"},
      {"role": "assistant", "text": "Forty-four days, listed."}],
     plan("filter", phrase="in June", filters=[f("sleep_score", "<", 60)], query="Which days in June was my sleep score under 60?"),
     [d for d in month(6) if by[d]["sleepScore"] < 60], [], "a follow-up that narrows the prior question; scored in Part 10"),
]


def main():
    out = ROOT / "evals" / "golden.jsonl"
    lines = []
    for gid, kind, q, p, dates, facts, notes in G:
        lines.append({"id": gid, "kind": kind, "question": q, "plan": p, "expected_dates": dates, "expected_facts": facts, "notes": notes})
    for gid, kind, q, hist, p, dates, facts, notes in FOLLOW_UPS:
        lines.append({"id": gid, "kind": kind, "question": q, "history": hist, "requires_history": True, "plan": p,
                      "expected_dates": dates, "expected_facts": facts, "notes": notes})
    out.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")
    kinds = {}
    for l in lines:
        kinds[l["kind"]] = kinds.get(l["kind"], 0) + 1
    print(f"wrote {len(lines)} questions to {out}: {kinds}")


if __name__ == "__main__":
    main()
