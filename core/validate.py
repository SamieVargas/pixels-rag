"""Citations enforced in code. The prompt asks for them; this checks them.

A cited date has to be one the pipeline retrieved. A claim has to name at
least one date unless the answer is an admission. And every number in the
answer has to appear in a cited day's text (or, on the aggregate route, in
the computed table), because numbers are what the model is most tempted to
invent about a time series."""

import re

DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b")
# A trailing period ends a sentence, not a number: "73.5." still yields 73.5.
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?!\w|\.\d)")


def normalize_number(tok: str) -> str:
    t = tok.replace(",", "").lstrip("+")
    try:
        f = float(t)
    except ValueError:
        return t
    return str(int(f)) if f == int(f) else str(f).rstrip("0").rstrip(".")


def numbers_in(text: str) -> set[str]:
    """Numbers as normalized strings, with dates removed first so 2026-06-14
    does not contribute 2026, 6 and 14."""
    scrubbed = DATE_RE.sub(" ", text or "")
    return {normalize_number(m.group(0)) for m in NUM_RE.finditer(scrubbed)}


def validate_answer(answer: dict, *, retrieved_ids, allowed_texts, question="", route="semantic") -> list[str]:
    """Return the list of violations; empty means the answer stands."""
    v = []
    retrieved = set(retrieved_ids or [])
    cited = list(answer.get("cited_dates") or [])
    for d in cited:
        if d not in retrieved:
            v.append(f"cited date {d} was not among the retrieved days")
    for i, claim in enumerate(answer.get("claims") or []):
        dates = claim.get("dates") or []
        if not answer.get("unanswerable") and route != "aggregate" and not dates:
            v.append(f"claim {i + 1} names no date: {claim.get('text', '')[:60]!r}")
        for d in dates:
            if d not in retrieved:
                v.append(f"claim {i + 1} cites {d}, which was not retrieved")
            elif d not in cited:
                v.append(f"claim {i + 1} cites {d}, which is missing from cited_dates")
    allowed = numbers_in(question)
    for t in allowed_texts or []:
        allowed |= numbers_in(t)
    for n in sorted(numbers_in(answer.get("answer") or "")):
        if n not in allowed:
            where = "the computed table" if route == "aggregate" else "any cited day"
            v.append(f"the number {n} in the answer does not appear in {where}")
    return v
