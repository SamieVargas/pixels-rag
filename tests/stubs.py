"""A scripted Anthropic client for the offline tests: each call pops the next
reply, and every request is kept so a test can look at what was sent."""

import json
from types import SimpleNamespace


def reply(payload, stop_reason="end_turn"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=120, output_tokens=60))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.replies:
            raise AssertionError("the stub client was asked for more replies than it was given")
        r = self.replies.pop(0)
        return r if hasattr(r, "content") else reply(r)


def decision(kind, *, phrase=None, filters=(), metric=None, stat=None, group_by=None, query=None, start=None, end=None):
    return {"kind": kind, "date_range": {"start": start, "end": end}, "date_phrase": phrase, "filters": list(filters),
            "aggregate": {"metric": metric, "stat": stat, "group_by": group_by}, "rewritten_query": query}


def answer(text, dates, claims=None, unanswerable=False):
    return {"answer": text, "claims": claims if claims is not None else [{"text": text, "dates": list(dates)}],
            "cited_dates": list(dates), "unanswerable": unanswerable, "why_unanswerable": None}
