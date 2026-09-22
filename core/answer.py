"""The answer call, under the Part 2 contract, with citations enforced in code
and one reject-and-retry: a validation failure goes back to the model as the
next user turn with the violations named, once."""

import json
import time

from core.contracts import MODEL, answer_schema
from core.parse import parse_json
from core.validate import validate_answer

SYSTEM = """You answer questions about a personal daily log using only the evidence provided. Rules:
- Every claim cites the dates it draws on, written exactly as they appear in the evidence (YYYY-MM-DD). cited_dates lists every date you used anywhere in the answer.
- Use numbers only as they appear in the evidence. Do not compute, round, or estimate new numbers; quote the ones you were given.
- When the evidence is a computed table, narrate the table: its numbers are exact, and the answer must use them as written.
- If the evidence does not answer the question, set unanswerable to true, say why in why_unanswerable, and keep claims empty.
- Keep the answer under 150 words, plain sentences, no headings."""

CONTRACT_HINT = "\n\nRespond ONLY with a JSON object of this shape, no prose:\n" + json.dumps({
    "answer": "string", "claims": [{"text": "string", "dates": ["YYYY-MM-DD"]}], "cited_dates": ["YYYY-MM-DD"],
    "unanswerable": False, "why_unanswerable": "string or null"})

EMPTY = {"answer": "", "claims": [], "cited_dates": [], "unanswerable": True, "why_unanswerable": "the model did not return a usable answer"}


def _text_of(message) -> str:
    return next((b.text for b in message.content if getattr(b, "type", "") == "text"), "")


def _coerce(value) -> dict:
    out = {**EMPTY, **{k: v for k, v in (value or {}).items() if k in EMPTY}}
    out["claims"] = [{"text": str(c.get("text", "")), "dates": [str(d) for d in (c.get("dates") or [])]} for c in (out["claims"] or []) if isinstance(c, dict)]
    out["cited_dates"] = [str(d) for d in (out["cited_dates"] or [])]
    out["unanswerable"] = bool(out["unanswerable"])
    return out


def generate_answer(client, question: str, *, evidence: str, route: str, retrieved_ids, allowed_texts,
                    contract="native", model=MODEL, max_retries=1) -> tuple[dict, dict]:
    """Returns (answer, meta). meta.valid says whether the final answer passed
    the validator; meta.violations holds the ones that stood after retries."""
    user = f"EVIDENCE:\n{evidence}\n\nQUESTION: {question}"
    messages = [{"role": "user", "content": user}]
    meta = {"call": "answer", "retries": 0, "parse_path": None, "latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
            "violations": [], "first_violations": [], "valid": False}
    answer = dict(EMPTY)
    for attempt in range(max_retries + 1):
        kwargs = {"model": model, "max_tokens": 800, "system": SYSTEM + ("" if contract == "native" else CONTRACT_HINT), "messages": messages}
        if contract == "native":
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": answer_schema()}}
        t0 = time.time()
        msg = client.messages.create(**kwargs)
        meta["latency_ms"] += int((time.time() - t0) * 1000)
        usage = getattr(msg, "usage", None)
        meta["input_tokens"] += getattr(usage, "input_tokens", 0)
        meta["output_tokens"] += getattr(usage, "output_tokens", 0)
        raw = _text_of(msg)
        parsed = parse_json(raw, getattr(msg, "stop_reason", None))
        meta["parse_path"] = parsed["path"]
        if not parsed["ok"]:
            violations = [f"reply could not be parsed: {parsed['error']}"]
            answer = dict(EMPTY)
        else:
            answer = _coerce(parsed["value"])
            violations = validate_answer(answer, retrieved_ids=retrieved_ids, allowed_texts=allowed_texts, question=question, route=route)
        if attempt == 0:
            meta["first_violations"] = violations
        if not violations:
            meta["valid"] = True
            meta["violations"] = []
            return answer, meta
        meta["violations"] = violations
        if attempt < max_retries:
            meta["retries"] += 1
            messages = messages + [
                {"role": "assistant", "content": raw or "{}"},
                {"role": "user", "content": "Your answer failed validation:\n" + "\n".join(f"- {v}" for v in violations)
                 + "\nReturn a corrected answer under the same contract. Cite only retrieved dates and use only numbers that appear in the evidence."},
            ]
    return answer, meta
