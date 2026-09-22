"""The tolerant parser. Native structured output makes it a fallback, and the
pipeline records which path handled each reply: native, recovered or failed."""

import json


def parse_json(text: str, stop_reason=None) -> dict:
    raw = (text or "").strip()
    try:
        return {"ok": True, "path": "native", "value": json.loads(raw)}
    except json.JSONDecodeError:
        pass
    unfenced = raw
    if unfenced.startswith("```"):
        unfenced = unfenced.split("\n", 1)[1] if "\n" in unfenced else ""
        unfenced = unfenced.rsplit("```", 1)[0]
    try:
        return {"ok": True, "path": "recovered", "value": json.loads(unfenced.strip())}
    except json.JSONDecodeError:
        pass
    start, end = unfenced.find("{"), unfenced.rfind("}")
    if start >= 0 and end > start:
        try:
            return {"ok": True, "path": "recovered", "value": json.loads(unfenced[start:end + 1])}
        except json.JSONDecodeError:
            pass
    cut = stop_reason == "max_tokens" or (start >= 0 and end <= start)
    return {"ok": False, "path": "failed", "error": "reply was cut off before the JSON closed" if cut else "reply was not JSON"}
