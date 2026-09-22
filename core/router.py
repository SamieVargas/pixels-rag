"""Query understanding: one small model call that says what kind of question
this is and extracts what code needs (fields, operators, values, the date
words). Relative dates are resolved here in code, never by the model."""

import json
import time

from core.contracts import FIELDS, KINDS, OPS, STATS, NUMERIC_FIELDS, NOT_LOGGED, MODEL, router_schema, field_lines
from core.dates import resolve_phrase
from core.parse import parse_json

SYSTEM = f"""You classify a question about a personal daily log (one record per day: mood, sleep, wearables, exercise, habits) and extract what the code needs to answer it. You never answer the question yourself.

FIELDS the log has (name, type, meaning):
{field_lines()}

Not logged at all: {NOT_LOGGED}. A question that needs one of those is unanswerable.

KINDS:
- semantic: the answer needs the days' text read and summarized ("what was that day like", "what happened when", "describe", "what did recovery look like"). Add filters and date words when the question names them; the search runs inside that set.
- filter: the answer is the list of days meeting explicit conditions ("which days", "list the days", "on which days").
- aggregate: the answer is a statistic (average, mean, median, count, how many, best, worst, highest, lowest, min, max) or a comparison between groups ("did I sleep better on hot yoga days", "weekends versus weekdays").
- unanswerable: asks for something never logged.

RULES:
- Never resolve relative dates. Copy the date words from the question into date_phrase exactly ("last two weeks", "in August", "the first two weeks of June", "the week of June 8"). Fill date_range.start and date_range.end only with dates written explicitly in the question, as YYYY-MM-DD; otherwise null.
- filters use only the listed fields. Text fields take "contains" or "=="; booleans take "==" with true or false; numbers take a comparison. "hot yoga days" is exercise contains "hot yoga". "weekends" is weekend == true.
- For "did I X better on Y days", "X on Y days versus other days": kind aggregate, the filters describe Y, and aggregate.group_by is "match". For "weekends versus weekdays": group_by "weekend". For "on hot yoga days" alone (no comparison): filters describe the days and group_by is null.
- aggregate.metric must be a numeric field; "sleep" means sleep_score, "energy" is not numeric. "how many days" is stat count with metric null. "best" and "highest" are max; "worst" and "lowest" are min.
- rewritten_query: for semantic kinds, a short standalone search phrase (the descriptive words, no date words). When conversation history is given and the question is a follow-up ("and on weekends?", "only in June?"), rewrite it as a full standalone question in rewritten_query and fill the other fields as if it had been asked that way.
- Return only the JSON object."""

CONTRACT_HINT = "\n\nRespond ONLY with a JSON object of this shape, no prose:\n" + json.dumps({
    "kind": "semantic | filter | aggregate | unanswerable",
    "date_range": {"start": "YYYY-MM-DD or null", "end": "YYYY-MM-DD or null"},
    "date_phrase": "the date words from the question, or null",
    "filters": [{"field": "field name", "op": "< | <= | > | >= | == | != | contains", "value": "number, string or boolean"}],
    "aggregate": {"metric": "numeric field or null", "stat": "mean | median | count | min | max | null", "group_by": "field name, match, or null"},
    "rewritten_query": "standalone query or null",
})


def _text_of(message) -> str:
    return next((b.text for b in message.content if getattr(b, "type", "") == "text"), "")


def classify(client, question: str, *, today, history=None, contract="native", model=MODEL) -> tuple[dict, dict]:
    """Ask the model for the routing decision. Returns (decision, meta)."""
    user = ""
    if history:
        user += "CONVERSATION SO FAR:\n" + "\n".join(f"{t['role']}: {t['text']}" for t in history[-6:]) + "\n\n"
    user += f"TODAY (the newest day in the log): {today.isoformat()}\nQUESTION: {question}"
    kwargs = {"model": model, "max_tokens": 400, "system": SYSTEM + ("" if contract == "native" else CONTRACT_HINT),
              "messages": [{"role": "user", "content": user}]}
    if contract == "native":
        kwargs["output_config"] = {"format": {"type": "json_schema", "schema": router_schema()}}
    t0 = time.time()
    msg = client.messages.create(**kwargs)
    ms = int((time.time() - t0) * 1000)
    parsed = parse_json(_text_of(msg), getattr(msg, "stop_reason", None))
    usage = getattr(msg, "usage", None)
    meta = {"call": "router", "parse_path": parsed["path"], "latency_ms": ms,
            "input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)}
    if not parsed["ok"]:
        meta["error"] = parsed["error"]
        return {"kind": "semantic", "date_range": {"start": None, "end": None}, "date_phrase": None, "filters": [],
                "aggregate": {"metric": None, "stat": None, "group_by": None}, "rewritten_query": None}, meta
    return parsed["value"], meta


def resolve(decision: dict, today) -> dict:
    """Turn a decision (from the model or a golden plan) into a plan the
    pipeline can run: dates resolved, filters checked, aggregate normalized.
    Anything the model got wrong is dropped and named in plan['notes']."""
    notes = []
    kind = decision.get("kind") if decision.get("kind") in KINDS else "semantic"
    if decision.get("kind") not in KINDS:
        notes.append(f"unknown kind {decision.get('kind')!r}, treated as semantic")
    dr = decision.get("date_range") or {}
    start, end = dr.get("start"), dr.get("end")
    date_range = None
    if start or end:
        date_range = (start or "0000-01-01", end or "9999-12-31")
    elif decision.get("date_phrase"):
        date_range = resolve_phrase(decision["date_phrase"], today)
        if date_range is None:
            notes.append(f"could not resolve date phrase {decision['date_phrase']!r}")
    filters = []
    for f in decision.get("filters") or []:
        field, op, value = f.get("field"), f.get("op"), f.get("value")
        if field not in FIELDS or op not in OPS:
            notes.append(f"dropped filter {f!r}")
            continue
        typ = FIELDS[field][0]
        if typ == "number":
            try:
                value = float(value)
            except (TypeError, ValueError):
                notes.append(f"dropped filter {f!r}: not a number")
                continue
            if op == "contains":
                op = "=="
        elif typ == "boolean":
            value = value if isinstance(value, bool) else str(value).strip().lower() in ("true", "yes", "1")
            op = "==" if op == "contains" else op
        filters.append({"field": field, "op": op, "value": value})
    agg = decision.get("aggregate") or {}
    metric, stat, group_by = agg.get("metric"), agg.get("stat"), agg.get("group_by")
    if kind == "aggregate":
        if metric not in NUMERIC_FIELDS:
            if metric:
                notes.append(f"metric {metric!r} is not numeric")
            metric = None
        if stat not in STATS:
            stat = "count" if metric is None else "mean"
        if stat != "count" and metric is None:
            notes.append("aggregate without a numeric metric, treated as a filter question")
            kind = "filter"
        if group_by and group_by != "match" and group_by not in FIELDS:
            notes.append(f"dropped group_by {group_by!r}")
            group_by = None
    else:
        metric = stat = group_by = None
    if kind == "filter" and not filters:
        notes.append("filter question without filters, treated as semantic")
        kind = "semantic"
    query = decision.get("rewritten_query") or None
    return {"kind": kind, "date_range": date_range, "date_phrase": decision.get("date_phrase"), "filters": filters,
            "aggregate": {"metric": metric, "stat": stat, "group_by": group_by} if kind == "aggregate" else None,
            "query": query, "notes": notes}
