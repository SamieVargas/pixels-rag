"""--explain: for each retrieved chunk, its score, which filters matched,
and which route ran, so any answer can be audited in ten seconds."""

from core import filters as F
from core.aggregate import fmt


def explain(result: dict, days: list[dict]) -> dict:
    by = {d["date"]: d for d in days}
    plan = result["plan"]
    cited = set(result["answer"].get("cited_dates") or [])
    rows = []
    for c in result["retrieved"]:
        day = by.get(c["id"])
        if day is None:
            matched = "week chunk" if c["id"].startswith("week:") else "not a day"
        elif plan["filters"]:
            matched = ", ".join(f"{f['field']} {f['op']} {fmt(f['value']) if isinstance(f['value'], float) else f['value']} {'✓' if F.matches(day, f) else '✗'}" for f in plan["filters"])
        else:
            matched = "no filters"
        in_range = "✓" if (day is None or F.in_range(day, plan["date_range"])) else "✗"
        rows.append({"id": c["id"], "score": c.get("score"), "dense_rank": c.get("dense_rank"), "rerank_score": c.get("rerank_score"),
                     "filters": matched, "in_range": in_range if plan["date_range"] else "no range", "cited": c["id"] in cited})
    return {"route": result["route"], "query": plan.get("query"), "date_range": plan["date_range"], "filters": plan["filters"],
            "notes": plan.get("notes", []), "rows": rows, "validation": result["validation"], "usage": result["usage"]}


def render(exp: dict) -> str:
    head = [f"route: {exp['route']}", f"date range: {exp['date_range'] or 'none'}", f"filters: {exp['filters'] or 'none'}"]
    if exp["query"]:
        head.append(f"searched as: {exp['query']}")
    lines = ["  ".join(head), "", "| chunk | score | dense rank | rerank | filters | in range | cited |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in exp["rows"]:
        lines.append(f"| {r['id']} | {r['score'] if r['score'] is not None else 'n/a'} | {r['dense_rank'] or ''} | {r['rerank_score'] if r['rerank_score'] is not None else ''} | {r['filters']} | {r['in_range']} | {'✓' if r['cited'] else ''} |")
    v = exp["validation"]
    lines.append(f"\nvalidator: {'passed' if v['ok'] else 'FAILED'} · retries {v['retries']} · parse {v['parse_path'] or 'n/a'} · tokens {exp['usage']['input_tokens']}+{exp['usage']['output_tokens']}")
    if v["first_violations"]:
        lines.append("first attempt violations: " + "; ".join(v["first_violations"]))
    return "\n".join(lines)
