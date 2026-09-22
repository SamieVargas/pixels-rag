"""Offline tests for pixels-rag v2. No API key and no model download: the
Anthropic client is scripted and the embedder is a hash. Run with

    python tests/test_core.py
"""

import json
import os
import statistics
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "evals"))

from core import filters as F  # noqa: E402
from core import index as I  # noqa: E402
from core.aggregate import aggregate, render_table  # noqa: E402
from core.dates import resolve_phrase  # noqa: E402
from core.days import load_days, latest_date  # noqa: E402
from core.parse import parse_json  # noqa: E402
from core.pipeline import ask, gather  # noqa: E402
from core.router import classify, resolve  # noqa: E402
from core.validate import validate_answer, numbers_in  # noqa: E402
from stubs import FakeClient, decision, answer, reply  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


DAYS = load_days(ROOT / "fixtures" / "days.json")
BY = {d["date"]: d for d in DAYS}
TODAY = latest_date(DAYS)
CLIENT = I.make_client()
COLL, _ = I.build(CLIENT, DAYS, name="test_day", embedding_function=I.HashEmbedding())
COLL_WEEK, _ = I.build(CLIENT, DAYS, name="test_week", level="day+week", embedding_function=I.HashEmbedding())


class NoQueryCollection:
    """A collection that must never be asked: the aggregate route does not retrieve."""

    def query(self, **kw):
        raise AssertionError("the aggregate route queried the index")

    def get(self, **kw):
        raise AssertionError("the aggregate route read the index")


def main():
    print("relative dates resolve in code")
    t = date(2026, 8, 31)
    cases = {
        "last two weeks": ("2026-08-18", "2026-08-31"), "in August": ("2026-08-01", "2026-08-31"),
        "the first two weeks of June": ("2026-06-01", "2026-06-14"), "June 14": ("2026-06-14", "2026-06-14"),
        "between June 1 and June 14": ("2026-06-01", "2026-06-14"), "the week of June 8": ("2026-06-08", "2026-06-14"),
        "last week": ("2026-08-24", "2026-08-30"), "second half of July": ("2026-07-16", "2026-07-31"),
        "end of August": ("2026-08-25", "2026-08-31"), "yesterday": ("2026-08-30", "2026-08-30"),
    }
    for phrase, want in cases.items():
        check(f"{phrase!r} -> {want[0]}..{want[1]}", resolve_phrase(phrase, t) == want)
    check("a non-date phrase resolves to nothing", resolve_phrase("hot yoga days", t) is None)
    check("a month past today's month falls in the previous year", resolve_phrase("in October", t) == ("2025-10-01", "2025-10-31"))

    print("filters")
    under = F.apply(DAYS, [{"field": "sleep_score", "op": "<", "value": 60}])
    check("numeric threshold matches the fixture", [d["date"] for d in under] == [d for d in sorted(BY) if BY[d]["sleep_score"] is not None and BY[d]["sleep_score"] < 60])
    hy = F.apply(DAYS, [{"field": "exercise", "op": "contains", "value": "hot yoga"}], ("2026-07-01", "2026-07-31"))
    check("contains is case-insensitive and the range applies", all("Hot yoga" in d["exercise"] and d["date"].startswith("2026-07") for d in hy) and len(hy) == 8)
    check("booleans accept true as a word", len(F.apply(DAYS, [{"field": "alcohol", "op": "==", "value": "true"}])) == 22)
    try:
        F.apply(DAYS, [{"field": "blood_pressure", "op": "<", "value": 120}])
        check("unknown field raises", False)
    except ValueError:
        check("unknown field raises", True)

    print("aggregate matches a direct computation")
    res = aggregate(DAYS, metric="sleep_score", stat="mean", group_by="match", filters=[{"field": "exercise", "op": "contains", "value": "hot yoga"}])
    hot = [d["sleep_score"] for d in DAYS if "hot yoga" in d["exercise"].lower()]
    rest = [d["sleep_score"] for d in DAYS if "hot yoga" not in d["exercise"].lower()]
    check("group means equal pandas-free means", abs(res["groups"][0]["value"] - statistics.mean(hot)) < 1e-9 and abs(res["groups"][1]["value"] - statistics.mean(rest)) < 1e-9)
    check("group sizes add up to the corpus", res["groups"][0]["n"] + res["groups"][1]["n"] == len(DAYS))
    cnt = aggregate(DAYS, metric=None, stat="count", filters=[{"field": "meditation", "op": "==", "value": True}], date_range=("2026-07-01", "2026-07-31"))
    check("count of days", cnt["groups"][0]["value"] == 13)
    med = aggregate(DAYS, metric="steps", stat="median", date_range=("2026-08-18", "2026-08-31"))
    check("median over a range", med["groups"][0]["value"] == statistics.median(d["steps"] for d in DAYS if d["date"] >= "2026-08-18"))
    wk = aggregate(DAYS, metric="rating", stat="mean", group_by="weekend")
    check("group by a boolean field yields two named rows", [g["name"] for g in wk["groups"]] == ["weekend: no", "weekend: yes"])
    check("the table renders the numbers it will be checked against", "| weekend: yes |" in render_table(wk))

    print("parser paths")
    check("clean JSON is native", parse_json('{"a": 1}')["path"] == "native")
    check("fenced JSON is recovered", parse_json('```json\n{"a": 1}\n```')["path"] == "recovered")
    check("prose-wrapped JSON is recovered", parse_json('Sure: {"a": 1} there')["path"] == "recovered")
    check("no JSON fails", parse_json("nothing here")["ok"] is False)
    check("a cut-off reply is named", "cut off" in parse_json("{\"a\": 1", stop_reason="max_tokens")["error"])

    print("validator")
    retrieved = ["2026-06-14", "2026-06-15"]
    texts = [BY["2026-06-14"]["text"], BY["2026-06-15"]["text"]]
    good = answer("Sleep score was 48 on 2026-06-14.", ["2026-06-14"])
    check("a grounded answer passes", validate_answer(good, retrieved_ids=retrieved, allowed_texts=texts) == [])
    bad_date = answer("Fine.", ["2026-06-16"])
    check("an uncited date is rejected", any("not among the retrieved" in v for v in validate_answer(bad_date, retrieved_ids=retrieved, allowed_texts=texts)))
    invented = answer("Sleep averaged 73.5.", ["2026-06-14"])
    check("an invented number is rejected", any("73.5" in v for v in validate_answer(invented, retrieved_ids=retrieved, allowed_texts=texts)))
    empty = {"answer": "Bad day.", "claims": [{"text": "Bad day.", "dates": []}], "cited_dates": [], "unanswerable": False, "why_unanswerable": None}
    check("a claim with no dates is rejected", any("names no date" in v for v in validate_answer(empty, retrieved_ids=retrieved, allowed_texts=texts)))
    check("numbers inside dates are not numbers", numbers_in("on 2026-06-14 and 6/14 it was 48") == {"48"})
    check("a number from the question is allowed", validate_answer(answer("Under 60: none.", ["2026-06-14"]), retrieved_ids=retrieved, allowed_texts=texts, question="under 60?") == [])

    print("schemas obey the API's structured-output rules")
    from core.contracts import router_schema, answer_schema

    def nodes(o):
        if isinstance(o, dict):
            yield o
            for v in o.values():
                yield from nodes(v)
        elif isinstance(o, list):
            for v in o:
                yield from nodes(v)

    for name, schema in (("router", router_schema()), ("answer", answer_schema())):
        all_nodes = list(nodes(schema))
        check(f"{name}: no enum on a list-typed field", not any(isinstance(n.get("type"), list) and "enum" in n for n in all_nodes))
        check(f"{name}: no null inside an enum", not any(None in n.get("enum", []) for n in all_nodes))
        check(f"{name}: every object closes additionalProperties", all(n.get("additionalProperties") is False for n in all_nodes if n.get("type") == "object"))
    metric = router_schema()["properties"]["aggregate"]["properties"]["metric"]
    check("metric is a string enum or null", [b.get("type") for b in metric["anyOf"]] == ["string", "null"] and "rating" in metric["anyOf"][0]["enum"])

    print("router: contract, resolution, fallbacks")
    fc = FakeClient([decision("filter", phrase="in July", filters=[{"field": "exercise", "op": "contains", "value": "hot yoga"}])])
    dec, meta = classify(fc, "List the days I did hot yoga in July.", today=TODAY)
    check("native contract sends output_config with the router schema", fc.requests[0].get("output_config", {}).get("format", {}).get("type") == "json_schema")
    check("today is in the prompt", TODAY.isoformat() in fc.requests[0]["messages"][0]["content"])
    plan = resolve(dec, TODAY)
    check("the phrase resolved to July", plan["date_range"] == ("2026-07-01", "2026-07-31"))
    fc = FakeClient([reply('```json\n' + json.dumps(decision("semantic", query="migraine")) + '\n```')])
    dec, meta = classify(fc, "migraine day?", today=TODAY, contract="prompt")
    check("prompt contract sends no output_config and recovers fenced JSON", "output_config" not in fc.requests[0] and meta["parse_path"] == "recovered")
    p = resolve(decision("filter", filters=[{"field": "blood_pressure", "op": "<", "value": 120}]), TODAY)
    check("a hallucinated field is dropped and the empty filter falls back to semantic", p["kind"] == "semantic" and any("dropped filter" in n for n in p["notes"]))
    p = resolve(decision("aggregate", metric="energy", stat="mean"), TODAY)
    check("a non-numeric metric is refused", p["kind"] != "aggregate" or p["aggregate"]["metric"] is None)
    p = resolve(decision("aggregate", filters=[{"field": "meditation", "op": "==", "value": "true"}], stat="count"), TODAY)
    check("count needs no metric and booleans coerce", p["kind"] == "aggregate" and p["filters"][0]["value"] is True)
    fc = FakeClient([reply("not json")])
    dec, meta = classify(fc, "?", today=TODAY)
    check("an unparseable router reply degrades to semantic with the error kept", dec["kind"] == "semantic" and "error" in meta)

    print("index")
    hits = I.retrieve(COLL, "sleep", 5, allowed_ids=["2026-06-14", "2026-06-15"])
    check("restricted retrieval returns only allowed ids", {h["id"] for h in hits} <= {"2026-06-14", "2026-06-15"} and hits)
    check("an empty allowed set asks nothing", I.retrieve(COLL, "sleep", 5, allowed_ids=[]) == [])
    week_hits = I.retrieve(COLL_WEEK, "week summary mean rating", 5)
    check("the week index holds week chunks that cover seven days", any(h["id"].startswith("week:") and len(h["days"]) == 7 for h in I.retrieve(COLL_WEEK, "week summary mean rating sleep score", 20)))

    print("routes through the pipeline (scripted client)")
    fc = FakeClient([decision("semantic", phrase="in June", query="migraine")])
    g = gather("What happened on the day I had a migraine in June?", resolve(fc.replies[0], TODAY), days=DAYS, collection=COLL)
    check("semantic with a date phrase only retrieves June days", g["retrieved"] and all(c["id"].startswith("2026-06") for c in g["retrieved"]))
    first = g["retrieved"][0]["id"]
    fc = FakeClient([decision("semantic", phrase="in June", query="migraine"), answer(f"On {first} things were rough.", [first])])
    r = ask("What happened on the day I had a migraine in June?", days=DAYS, collection=COLL, client=fc)
    check("semantic route answers with a valid citation", r["route"] == "semantic" and r["validation"]["ok"] and r["counts"]["cited"] == 1)
    check("retrieved-but-uncited is counted", r["counts"]["retrieved_uncited"] == r["counts"]["retrieved"] - 1)

    july_hy = [d["date"] for d in F.apply(DAYS, [{"field": "exercise", "op": "contains", "value": "hot yoga"}], ("2026-07-01", "2026-07-31"))]
    fc = FakeClient([decision("filter", phrase="in July", filters=[{"field": "exercise", "op": "contains", "value": "hot yoga"}]),
                     answer("Eight hot yoga days in July.", july_hy[:2])])
    r = ask("List the days I did hot yoga in July.", days=DAYS, collection=COLL, client=fc)
    check("filter route returns the matching days as the retrieved set", r["route"] == "filter" and [c["id"] for c in r["retrieved"]] == july_hy)
    check("the evidence sent lists every matching day", all(d in fc.requests[1]["messages"][0]["content"] for d in july_hy))

    res = aggregate(DAYS, metric="sleep_score", stat="mean", group_by="match", filters=[{"field": "exercise", "op": "contains", "value": "hot yoga"}])
    table = render_table(res)
    nums = sorted(numbers_in(table) - {str(g["n"]) for g in res["groups"]})
    fc = FakeClient([decision("aggregate", filters=[{"field": "exercise", "op": "contains", "value": "hot yoga"}], metric="sleep_score", stat="mean", group_by="match"),
                     answer(f"Hot yoga days averaged {nums[-1]} against {nums[0]} on other days.", [], claims=[{"text": "table", "dates": []}])])
    r = ask("Did I sleep better on hot yoga days?", days=DAYS, collection=NoQueryCollection(), client=fc)
    check("aggregate route never queries the index and passes when it quotes the table", r["route"] == "aggregate" and r["validation"]["ok"])
    check("the table went to the model as evidence", "COMPUTED TABLE" in fc.requests[1]["messages"][0]["content"])
    fc = FakeClient([decision("aggregate", filters=[{"field": "exercise", "op": "contains", "value": "hot yoga"}], metric="sleep_score", stat="mean", group_by="match"),
                     answer("Hot yoga days averaged 99.9.", [], claims=[]), answer("Hot yoga days averaged 99.9.", [], claims=[])])
    r = ask("Did I sleep better on hot yoga days?", days=DAYS, collection=NoQueryCollection(), client=fc)
    check("a number outside the table fails even after the retry", not r["validation"]["ok"] and r["validation"]["retries"] == 1 and any("99.9" in v for v in r["validation"]["violations"]))

    fc = FakeClient([decision("unanswerable")])
    r = ask("What was my blood pressure in June?", days=DAYS, collection=NoQueryCollection(), client=fc)
    check("unanswerable short-circuits after the router", r["route"] == "unanswerable" and r["answer"]["unanswerable"] and r["model_calls"] == 1)

    print("reject and retry")
    fc = FakeClient([decision("semantic", phrase="in June", query="migraine"),
                     answer("Sleep was 12345 on a day.", ["2026-01-01"]), answer(f"It was {first}.", [first])])
    r = ask("migraine?", days=DAYS, collection=COLL, client=fc)
    check("the retry carries the violations as the next user turn", len(fc.requests[2]["messages"]) == 3 and "failed validation" in fc.requests[2]["messages"][2]["content"])
    check("the corrected answer stands and the first violations are kept", r["validation"]["ok"] and r["validation"]["retries"] == 1 and r["validation"]["first_violations"])

    print("golden set integrity")
    golden = [json.loads(l) for l in (ROOT / "evals" / "golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    scored = [g for g in golden if not g.get("requires_history")]
    kinds = {k: sum(1 for g in scored if g["kind"] == k) for k in ("semantic", "filter", "aggregate", "unanswerable")}
    check(f"26 scored questions across four kinds {kinds}", len(scored) == 26 and min(kinds.values()) >= 5)
    check("two follow-ups carry history", sum(1 for g in golden if g.get("requires_history")) == 2 and all(g.get("history") for g in golden if g.get("requires_history")))
    ok_dates = all(set(g["expected_dates"]) <= set(BY) for g in golden)
    check("every expected date exists in the fixture", ok_dates)
    clean = all(not resolve(g["plan"], TODAY)["notes"] for g in golden)
    check("every plan resolves without notes", clean)
    fmatch = all([d["date"] for d in F.apply(DAYS, resolve(g["plan"], TODAY)["filters"], resolve(g["plan"], TODAY)["date_range"])] == g["expected_dates"] for g in golden if g["kind"] == "filter")
    check("filter labels, computed independently, equal the filter code's answer", fmatch)
    amatch = True
    for g in golden:
        if g["kind"] != "aggregate":
            continue
        p = resolve(g["plan"], TODAY)
        t = render_table(aggregate(DAYS, metric=p["aggregate"]["metric"], stat=p["aggregate"]["stat"], group_by=p["aggregate"]["group_by"], filters=p["filters"], date_range=p["date_range"]))
        amatch &= all(f in t for f in g["expected_facts"])
    check("aggregate facts, computed independently, appear in the aggregator's table", amatch)

    print("runner scores one question offline and writes results")
    import run  # noqa: E402
    with tempfile.TemporaryDirectory() as tmp:
        code = run.main(["--offline", "--only", "F02,A02,U01,S01", "--embedding", "hash", "--out", tmp])
        files = os.listdir(tmp)
        check("exit 0", code == 0)
        check("wrote .md and .json", any(f.endswith(".md") for f in files) and any(f.endswith(".json") for f in files))
        data = json.loads(Path(tmp, next(f for f in files if f.endswith(".json"))).read_text(encoding="utf-8"))
        f02 = next(r for r in data["records"] if r["id"] == "F02")
        check("a filter question scores full recall offline", f02["recall"]["10"] == 1.0)
        code = run.main(["--offline", "--ablation", "--ablation-runs", "1", "--only", "S06,S01", "--embedding", "hash", "--out", tmp])
        check("ablation offline writes an ablation table", code == 0 and any("ablation" in f for f in os.listdir(tmp)))

    print("runner writes the partial table when a keyed run is interrupted")

    class Interrupting(FakeClient):
        """Answers like the stub for `after` calls, then raises the way Ctrl+C does."""

        def __init__(self, replies, after):
            super().__init__(replies)
            self.after = after

        def create(self, **kwargs):
            if len(self.requests) >= self.after:
                raise KeyboardInterrupt
            return super().create(**kwargs)

    f02 = next(g for g in golden if g["id"] == "F02")
    with tempfile.TemporaryDirectory() as tmp:
        stub = Interrupting([f02["plan"], answer("Those are the hot yoga days.", f02["expected_dates"][:1])], after=2)
        code = run.main(["--only", "F02,F03", "--embedding", "hash", "--out", tmp], client=stub)
        files = os.listdir(tmp)
        check("exit 130", code == 130)
        check("wrote a -partial .md and .json, nothing else", sorted(files) == sorted([f"{date.today().isoformat()}-partial.md", f"{date.today().isoformat()}-partial.json"]))
        md = Path(tmp, f"{date.today().isoformat()}-partial.md").read_text(encoding="utf-8")
        check("the header says PARTIAL: 1 of 2 questions", "PARTIAL: 1 of 2 questions" in md.splitlines()[0])
        data = json.loads(Path(tmp, f"{date.today().isoformat()}-partial.json").read_text(encoding="utf-8"))
        check("the json records completed, planned, partial", (data["completed"], data["planned"], data["partial"]) == (1, 2, True))
        check("the finished question is the one recorded", [r["id"] for r in data["records"]] == ["F02"])

    print("runner writes the partial ablation when an arm is interrupted")
    real = run._ablation_question
    seen = {"n": 0}

    def three_then_interrupt(*a, **k):
        seen["n"] += 1
        if seen["n"] > 3:
            raise KeyboardInterrupt
        return real(*a, **k)

    with tempfile.TemporaryDirectory() as tmp:
        run._ablation_question = three_then_interrupt
        try:
            code = run.main(["--offline", "--ablation", "--ablation-runs", "2", "--only", "S06,S01", "--embedding", "hash", "--out", tmp])
        finally:
            run._ablation_question = real
        files = os.listdir(tmp)
        check("exit 130 and an -ablation-partial pair", code == 130 and any(f.endswith("-offline-ablation-partial.md") for f in files) and any(f.endswith("-offline-ablation-partial.json") for f in files))
        md = Path(tmp, next(f for f in files if f.endswith(".md"))).read_text(encoding="utf-8")
        check("the header says PARTIAL: 3 of 8 runs", "PARTIAL: 3 of 8 runs" in md.splitlines()[0])
        check("arm B, never reached, reads n/a", "| Recall@5 (expected days covered) |" in md and md.count("n/a") >= 1)
        data = json.loads(Path(tmp, next(f for f in files if f.endswith(".json"))).read_text(encoding="utf-8"))
        check("only arm A has rows", list(data["arms"]) == ["A"] and len(data["arms"]["A"]["rows"]) == 3 and data["partial"] is True)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
