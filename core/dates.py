"""Relative and absolute date phrases, resolved in code against a fixed
"today" (the newest day in the index, so a question means the same thing on
every run). Returns (start, end) as ISO strings, inclusive, or None when the
phrase is not a date at all."""

import re
from calendar import monthrange
from datetime import date, timedelta

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})
WORDS = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "couple of": 2, "few": 3}


def _n(tok: str) -> int:
    tok = tok.strip().lower()
    return int(tok) if tok.isdigit() else WORDS[tok]


def _month(name: str, today: date) -> tuple[int, int]:
    m = MONTHS[name.lower()]
    y = today.year if m <= today.month else today.year - 1
    return y, m


def _month_span(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, monthrange(y, m)[1])


def _parse_day(text: str, today: date):
    """'June 14', 'June 14, 2026', '6/14', '6/14/2026', '2026-06-14' -> date."""
    t = text.strip().strip(".,").lower()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return date(int(m[1]), int(m[2]), int(m[3]))
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?", t)
    if m:
        y = int(m[3]) if m[3] else today.year
        if y < 100:
            y += 2000
        return date(y, int(m[1]), int(m[2]))
    m = re.fullmatch(r"([a-z]+)\.? (\d{1,2})(?:st|nd|rd|th)?(?:,? (\d{4}))?", t)
    if m and m[1] in MONTHS:
        y = int(m[3]) if m[3] else _month(m[1], today)[0]
        return date(y, MONTHS[m[1]], int(m[2]))
    return None


NUM = r"(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|couple of|few)"


def resolve_phrase(phrase, today: date):
    if not phrase:
        return None
    p = re.sub(r"\s+", " ", str(phrase).strip().lower().rstrip("?.!"))
    p = re.sub(r"^(in|on|during|for|over|of|the) ", "", p)
    p = re.sub(r"^(in|on|during|for|over|of|the) ", "", p)
    iso = lambda a, b: (a.isoformat(), b.isoformat())

    if p in ("today",):
        return iso(today, today)
    if p == "yesterday":
        return iso(today - timedelta(days=1), today - timedelta(days=1))
    if p in ("this week",):
        start = today - timedelta(days=today.weekday())
        return iso(start, today)
    if p == "last week":
        end = today - timedelta(days=today.weekday() + 1)
        return iso(end - timedelta(days=6), end)
    if p == "this month":
        return iso(date(today.year, today.month, 1), today)
    if p == "last month":
        y, m = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
        return iso(*_month_span(y, m))

    m = re.fullmatch(rf"(?:last|past|previous) {NUM} (day|week|month)s?", p)
    if m:
        n, unit = _n(m[1]), m[2]
        days = n if unit == "day" else n * 7 if unit == "week" else n * 30
        return iso(today - timedelta(days=days - 1), today)

    m = re.fullmatch(r"(first|second|last) (half|week|two weeks|three weeks|\d+ weeks?|\d+ days) of ([a-z]+)(?: (\d{4}))?", p)
    if m and m[3] in MONTHS:
        y = int(m[4]) if m[4] else _month(m[3], today)[0]
        start, end = _month_span(y, MONTHS[m[3]])
        part, unit = m[1], m[2]
        if unit == "half":
            mid = start + timedelta(days=14)
            return iso(start, mid) if part == "first" else iso(mid + timedelta(days=1), end)
        length = 7 if unit == "week" else 14 if unit == "two weeks" else 21 if unit == "three weeks" else int(unit.split()[0]) * (7 if "week" in unit else 1)
        if part == "first":
            return iso(start, min(start + timedelta(days=length - 1), end))
        if part == "second":
            s = start + timedelta(days=length)
            return iso(s, min(s + timedelta(days=length - 1), end))
        return iso(max(end - timedelta(days=length - 1), start), end)

    m = re.fullmatch(r"week of (.+)", p)
    if m:
        d = _parse_day(m[1], today)
        if d:
            return iso(d, d + timedelta(days=6))

    m = re.fullmatch(r"(?:between|from) (.+?) (?:and|to|through|until|-) (.+)", p)
    if m:
        a, b = _parse_day(m[1], today), _parse_day(m[2], today)
        if a and b:
            return iso(min(a, b), max(a, b))

    m = re.fullmatch(r"([a-z]+)\.?(?: (\d{4}))?", p)
    if m and m[1] in MONTHS:
        y = int(m[2]) if m[2] else _month(m[1], today)[0]
        return iso(*_month_span(y, MONTHS[m[1]]))
    m = re.fullmatch(r"(end|start|beginning) of ([a-z]+)(?: (\d{4}))?", p)
    if m and m[2] in MONTHS:
        y = int(m[3]) if m[3] else _month(m[2], today)[0]
        start, end = _month_span(y, MONTHS[m[2]])
        return iso(end - timedelta(days=6), end) if m[1] == "end" else iso(start, start + timedelta(days=6))

    d = _parse_day(p, today)
    if d:
        return iso(d, d)
    return None
