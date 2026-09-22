"""Retrieval evals over the golden set.

    python evals/run.py --offline                # retrieval only: plans from the golden set, no key
    python evals/run.py                          # keyed: the router decides, the model answers
    python evals/run.py --contract prompt        # the prompt contract with the parser as the path
    python evals/run.py --ablation               # chunk granularity: day vs day+week, 20 runs per arm

Per question: the route chosen versus the golden kind, recall@k of the
expected dates in the retrieved set (k = 3, 5, 10), MRR, expected facts
present in the answer, citation validity, abstention on unanswerable
questions and on answerable ones, tokens and latency. Writes a markdown
table and the raw records to evals/results/<date>[-offline][-ablation].md/json.
"""

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import index as I  # noqa: E402
from core.contracts import MODEL  # noqa: E402
from core.embeddings import ARMS, DEFAULT as DEFAULT_ARM, run_arm  # noqa: E402
from core.days import load_days, latest_date  # noqa: E402
from core.pipeline import ask, gather  # noqa: E402
from core.rerank import make_reranker  # noqa: E402
from core.router import resolve  # noqa: E402

KS = (3, 5, 10)
ABLATION_RUNS = 20


def load_golden(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if not r.get("requires_history")]


def covered_in_order(retrieved):
    """The days each retrieved chunk covers, in rank order (a week chunk covers seven)."""
    return [set(c.get("days") or [c["id"]]) for c in retrieved]


def score_retrieval(expected, retrieved, ks=KS, kind="semantic"):
    exp = set(expected)
    ranks = covered_in_order(retrieved)
    out = {"recall": {}, "mrr": None}
    if not exp or kind not in ("semantic", "filter"):
        return out
    for k in ks:
        seen = set().union(*ranks[:k]) if ranks[:k] else set()
        out["recall"][str(k)] = round(len(exp & seen) / len(exp), 4)
    out["mrr"] = 0.0
    for i, days in enumerate(ranks, start=1):
        if days & exp:
            out["mrr"] = round(1 / i, 4)
            break
    return out


def exact_match(expected, retrieved, kind):
    """For filter questions the retrieved set is the answer: equal to the
    expected dates or not. Recall@k with k below the match count is by
    construction under 100%, so this is the number that matters there."""
    if kind != "filter":
        return None
    return sorted(set(expected)) == sorted({c["id"] for c in retrieved})


def facts_present(facts, text):
    if not facts:
        return None
    t = (text or "").lower()
    return round(sum(1 for f in facts if str(f).lower() in t) / len(facts), 4)


def run_question(g, *, days, collection, client, contract, top_k, offline, ks=KS, reranker=None):
    today = latest_date(days)
    t0 = time.time()
    if offline:
        plan = resolve(g["plan"], today)
        ev = gather(g["question"], plan, days=days, collection=collection, top_k=top_k, reranker=reranker)
        rec = {"id": g["id"], "kind": g["kind"], "question": g["question"], "route": ev["route"], "route_ok": None,
               "retrieved": [c["id"] for c in ev["retrieved"]], "facts": None, "citations_valid": None,
               "abstained": None, "should_abstain": g["kind"] == "unanswerable", "retries": None, "parse_path": None,
               "tokens": None, "latency_ms": int((time.time() - t0) * 1000), "hard_fail": False, "answer": None}
        rec.update(score_retrieval(g["expected_dates"], ev["retrieved"], ks, kind=g["kind"]))
        rec["exact"] = exact_match(g["expected_dates"], ev["retrieved"], g["kind"])
        return rec
    r = ask(g["question"], days=days, collection=collection, client=client, contract=contract, top_k=top_k, reranker=reranker)
    invalid_citation = any("not among the retrieved" in v or "which was not retrieved" in v for v in r["validation"]["violations"])
    rec = {"id": g["id"], "kind": g["kind"], "question": g["question"], "route": r["route"], "route_ok": r["route"] == g["kind"],
           "retrieved": [c["id"] for c in r["retrieved"]], "facts": facts_present(g["expected_facts"], r["answer"]["answer"]),
           "citations_valid": not invalid_citation, "abstained": bool(r["answer"]["unanswerable"]),
           "should_abstain": g["kind"] == "unanswerable", "retries": r["validation"]["retries"],
           "parse_path": r["validation"]["parse_path"], "tokens": r["usage"], "latency_ms": r["latency_ms"],
           "validation_ok": r["validation"]["ok"], "violations": r["validation"]["violations"],
           "hard_fail": invalid_citation, "answer": r["answer"]["answer"], "plan_notes": r["plan"]["notes"]}
    rec.update(score_retrieval(g["expected_dates"], r["retrieved"], ks, kind=g["kind"]))
    rec["exact"] = exact_match(g["expected_dates"], r["retrieved"], g["kind"])
    return rec


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def aggregate_scores(records, ks=KS):
    by_kind = lambda k: [r for r in records if r["kind"] == k]
    ret = [r for r in records if r["kind"] in ("semantic", "filter")]
    agg = {"n": len(records), "route_accuracy": _mean([r["route_ok"] for r in records if r["route_ok"] is not None])}
    for kind in ("semantic", "filter"):
        rows = by_kind(kind)
        agg[f"recall_{kind}"] = {str(k): _mean([r["recall"].get(str(k)) for r in rows]) for k in ks}
        agg[f"mrr_{kind}"] = _mean([r["mrr"] for r in rows])
    agg["recall_all"] = {str(k): _mean([r["recall"].get(str(k)) for r in ret]) for k in ks}
    agg["filter_exact"] = _mean([r.get("exact") for r in by_kind("filter")])
    agg["facts"] = _mean([r["facts"] for r in records])
    agg["citations_valid"] = _mean([r["citations_valid"] for r in records if r["citations_valid"] is not None])
    un = [r for r in records if r["should_abstain"] and r["abstained"] is not None]
    an = [r for r in records if not r["should_abstain"] and r["abstained"] is not None]
    agg["abstained_when_should"] = (sum(1 for r in un if r["abstained"]), len(un))
    agg["abstained_when_should_not"] = (sum(1 for r in an if r["abstained"]), len(an))
    agg["hard_fails"] = sum(1 for r in records if r["hard_fail"])
    agg["retries"] = sum(r["retries"] or 0 for r in records)
    paths = {}
    for r in records:
        if r["parse_path"]:
            paths[r["parse_path"]] = paths.get(r["parse_path"], 0) + 1
    agg["parse_paths"] = paths
    toks = [r["tokens"] for r in records if r["tokens"]]
    agg["mean_tokens"] = {"input": _mean([t["input_tokens"] for t in toks]), "output": _mean([t["output_tokens"] for t in toks])} if toks else None
    agg["mean_latency_ms"] = _mean([r["latency_ms"] for r in records])
    return agg


def pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def render(records, agg, *, offline, contract, embedding, ks=KS, rerank=None):
    head = f"# pixels-rag eval · {date.today().isoformat()} · {'offline, plans from the golden set' if offline else f'keyed, `{MODEL}`, contract `{contract}`'} · embedding `{embedding}`" + (f" · rerank `{rerank}`" if rerank and rerank != "none" else "")
    lines = [head, "", "| Measure | Result |", "| --- | --- |", f"| Questions scored | {agg['n']} |"]
    lines.append(f"| Route accuracy (router kind = golden kind) | {pct(agg['route_accuracy']) if not offline else 'n/a offline'} |")
    for kind in ("semantic", "filter"):
        r = agg[f"recall_{kind}"]
        extra = f" · matched set exact {pct(agg['filter_exact'])}" if kind == "filter" else ""
        lines.append(f"| Recall@3 / @5 / @10, {kind} | {pct(r['3'])} / {pct(r['5'])} / {pct(r['10'])} · MRR {agg[f'mrr_{kind}'] if agg[f'mrr_{kind}'] is not None else 'n/a'}{extra} |")
    lines.append(f"| Expected facts in the answer | {pct(agg['facts']) if not offline else 'n/a offline'} |")
    lines.append(f"| Citations valid (hard fail otherwise) | {pct(agg['citations_valid']) if not offline else 'n/a offline'} · hard fails {agg['hard_fails']} |")
    a, b = agg["abstained_when_should"], agg["abstained_when_should_not"]
    lines.append(f"| Abstained on unanswerable / on answerable | {a[0]}/{a[1]} · {b[0]}/{b[1]} |" if not offline else "| Abstention | n/a offline |")
    lines.append(f"| Validator retries · parse paths | {agg['retries']} · {agg['parse_paths'] or 'n/a'} |")
    lines.append(f"| Mean tokens in / out · mean latency | {agg['mean_tokens']['input'] if agg['mean_tokens'] else 'n/a'} / {agg['mean_tokens']['output'] if agg['mean_tokens'] else 'n/a'} · {agg['mean_latency_ms']} ms |")
    lines += ["", "| Q | Kind | Route | R@3 | R@5 | R@10 | MRR | Facts | Cited ok | Abstained | Retries | ms |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        rc = r["recall"]
        lines.append(f"| {r['id']} | {r['kind']} | {r['route']}{'' if r['route_ok'] in (None, True) else ' ✗'} | {pct(rc.get('3'))} | {pct(rc.get('5'))} | {pct(rc.get('10'))} | "
                     f"{r['mrr'] if r['mrr'] is not None else 'n/a'} | {pct(r['facts'])} | {'n/a' if r['citations_valid'] is None else ('✓' if r['citations_valid'] else '✗')} | "
                     f"{'n/a' if r['abstained'] is None else ('✓' if r['abstained'] else '·')} | {r['retries'] if r['retries'] is not None else 'n/a'} | {r['latency_ms']} |")
    return "\n".join(lines)


def run_ablation(golden, *, days, client, contract, top_k, offline, runs, embedding_function, embedding):
    """Arm A: day chunks. Arm B: day chunks plus weekly rollups. Same
    questions, same plans (the router is bypassed so only the index differs)."""
    sem = [g for g in golden if g["kind"] == "semantic"]
    today = latest_date(days)
    chroma = I.make_client()
    arms = {}
    for arm, level in (("A", "day"), ("B", "day+week")):
        coll, _ = I.build(chroma, days, name=f"ablation_{arm}", level=level, embedding_function=embedding_function)
        rows = []
        for run in range(1, runs + 1):
            for g in sem:
                plan = resolve(g["plan"], today)
                if offline:
                    ev = gather(g["question"], plan, days=days, collection=coll, top_k=top_k)
                    sc = score_retrieval(g["expected_dates"], ev["retrieved"])
                    rows.append({"id": g["id"], "run": run, "recall5": sc["recall"].get("5"), "facts": None, "weeks": sum(1 for c in ev["retrieved"] if c["id"].startswith("week:"))})
                else:
                    r = ask(g["question"], days=days, collection=coll, client=client, contract=contract, top_k=top_k, plan=g["plan"])
                    sc = score_retrieval(g["expected_dates"], r["retrieved"])
                    rows.append({"id": g["id"], "run": run, "recall5": sc["recall"].get("5"), "facts": facts_present(g["expected_facts"], r["answer"]["answer"]),
                                 "weeks": sum(1 for c in r["retrieved"] if c["id"].startswith("week:"))})
                print(f"  arm {arm} run {run}/{runs} {g['id']} recall@5={sc['recall'].get('5')}", flush=True)
        arms[arm] = {"level": level, "rows": rows, "recall5": _mean([r["recall5"] for r in rows]), "facts": _mean([r["facts"] for r in rows]),
                     "week_chunks_retrieved": sum(r["weeks"] for r in rows)}
    lines = [f"# Chunk granularity ablation · {date.today().isoformat()} · {len(sem)} semantic questions × {runs} run(s) per arm · embedding `{embedding}`" + (" · offline" if offline else f" · `{MODEL}`"),
             "", "Arm A indexes one chunk per day. Arm B adds one deterministic rollup chunk per week, tagged `level: week`; a retrieved week counts as covering its seven days.", "",
             "| Measure | A · day chunks | B · day + week rollups |", "| --- | --- | --- |",
             f"| Recall@5 (expected days covered) | {pct(arms['A']['recall5'])} | {pct(arms['B']['recall5'])} |",
             f"| Expected facts in the answer | {pct(arms['A']['facts']) if not offline else 'n/a offline'} | {pct(arms['B']['facts']) if not offline else 'n/a offline'} |",
             f"| Week chunks retrieved (total) | {arms['A']['week_chunks_retrieved']} | {arms['B']['week_chunks_retrieved']} |",
             "", "| Q | A recall@5 | B recall@5 |", "| --- | --- | --- |"]
    for g in sem:
        a = _mean([r["recall5"] for r in arms["A"]["rows"] if r["id"] == g["id"]])
        b = _mean([r["recall5"] for r in arms["B"]["rows"] if r["id"] == g["id"]])
        lines.append(f"| {g['id']} | {pct(a)} | {pct(b)} |")
    return "\n".join(lines), arms


def run_rerank_compare(golden, *, days, collection, top_k, reranker_name):
    """Plain top-k against retrieve-20-then-rerank-to-k, on the semantic
    questions, retrieval only. Recall@5 and the milliseconds per question."""
    sem = [g for g in golden if g["kind"] == "semantic"]
    today = latest_date(days)
    rr = make_reranker(reranker_name)
    rows = []
    for g in sem:
        plan = resolve(g["plan"], today)
        t0 = time.time()
        plain = gather(g["question"], plan, days=days, collection=collection, top_k=top_k)
        plain_ms = int((time.time() - t0) * 1000)
        t0 = time.time()
        rer = gather(g["question"], plan, days=days, collection=collection, top_k=top_k, reranker=rr)
        rer_ms = int((time.time() - t0) * 1000)
        rows.append({"id": g["id"], "plain": score_retrieval(g["expected_dates"], plain["retrieved"])["recall"].get("5"),
                     "reranked": score_retrieval(g["expected_dates"], rer["retrieved"])["recall"].get("5"),
                     "plain_ms": plain_ms, "reranked_ms": rer_ms, "rerank_only_ms": rer["rerank_ms"]})
    a, b = _mean([r["plain"] for r in rows]), _mean([r["reranked"] for r in rows])
    lines = [f"# Reranking · {date.today().isoformat()} · reranker `{reranker_name}` · {len(sem)} semantic questions · retrieval only",
             "", "Plain: dense top-k. Reranked: dense top-20, then the reranker keeps the top k.", "",
             "| Measure | Plain top-k | Retrieve 20, rerank to k |", "| --- | --- | --- |",
             f"| Recall@5 | {pct(a)} | {pct(b)} |",
             f"| Mean ms per question (retrieval, plus reranking) | {_mean([r['plain_ms'] for r in rows])} | {_mean([r['reranked_ms'] for r in rows])} (reranker alone {_mean([r['rerank_only_ms'] for r in rows])}) |",
             "", "| Q | Plain | Reranked |", "| --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['id']} | {pct(r['plain'])} | {pct(r['reranked'])} |")
    gain = None if a is None or b is None else round((b - a) * 100, 1)
    lines += ["", f"Gain: {gain:+.1f} points of recall@5." if gain is not None else "Gain: n/a"]
    return "\n".join(lines), rows


def run_embedding_ablation(golden, *, days, arms, top_k):
    """One index per embedding model, the semantic questions through each,
    retrieval only. Arms that cannot run here say why."""
    sem = [g for g in golden if g["kind"] == "semantic"]
    today = latest_date(days)
    qs = [(g["question"], resolve(g["plan"], today), g["expected_dates"]) for g in sem]
    rows = [run_arm(a, days=days, questions=qs, score=score_retrieval, top_k=top_k) for a in arms]
    lines = [f"# Embedding models · {date.today().isoformat()} · {len(sem)} semantic questions · retrieval only", "",
             "The default stays local whatever the numbers say. The `openai` arm sends every chunk's text to OpenAI and is opt-in.", "",
             "| Arm | Model | Recall@5 | Index build | Query latency |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        if r["ran"]:
            lines.append(f"| {r['arm']}{' (default)' if r['arm'] == DEFAULT_ARM else ''} | `{r['model']}` | {pct(r['recall5'])} | {r['build_s']} s | {r['query_ms']} ms |")
        else:
            lines.append(f"| {r['arm']} | `{r['model']}` | not run | | {r['reason']} |")
    return "\n".join(lines), rows


def main(argv=None, client=None):
    p = argparse.ArgumentParser(description="pixels-rag retrieval evals")
    p.add_argument("--golden", default=str(ROOT / "evals" / "golden.jsonl"))
    p.add_argument("--days", default=str(ROOT / "fixtures" / "days.json"), help="export file to index (default: the committed fixture)")
    p.add_argument("--offline", action="store_true", help="plans from the golden set, retrieval metrics only, no key")
    p.add_argument("--contract", choices=("native", "prompt"), default="native")
    p.add_argument("--embedding", choices=("default", "hash"), default="default", help="hash is the offline test embedder")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--only", default=None, help="comma-separated question ids")
    p.add_argument("--ablation", action="store_true", help="chunk granularity ablation on the semantic questions")
    p.add_argument("--ablation-runs", type=int, default=ABLATION_RUNS)
    p.add_argument("--rerank", choices=("none", "lexical", "cross-encoder"), default="none", help="rerank the top 20 to top-k before answering")
    p.add_argument("--rerank-compare", action="store_true", help="plain top-k against reranked, semantic questions, retrieval only")
    p.add_argument("--embedding-ablation", action="store_true", help="one index per embedding model, semantic questions, retrieval only")
    p.add_argument("--arms", default="minilm,bge-small,e5-small,openai", help="comma-separated arms for --embedding-ablation")
    p.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = p.parse_args(argv)

    golden = load_golden(args.golden)
    if args.only:
        keep = set(args.only.split(","))
        golden = [g for g in golden if g["id"] in keep]
    days = load_days(args.days)
    ef = I.HashEmbedding() if args.embedding == "hash" else None
    if not args.offline and client is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY is not set. Use --offline for the retrieval-only run.", file=sys.stderr)
            return 2
        import anthropic
        client = anthropic.Anthropic()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat() + ("-offline" if args.offline else "")

    if args.embedding_ablation:
        arms = [a.strip() for a in args.arms.split(",") if a.strip()]
        table, rows = run_embedding_ablation(golden, days=days, arms=arms, top_k=args.top_k)
        print("\n" + table)
        (out / f"{stamp}-embeddings.md").write_text(table + "\n", encoding="utf-8")
        (out / f"{stamp}-embeddings.json").write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": rows}, indent=1), encoding="utf-8")
        print(f"wrote {out / (stamp + '-embeddings.md')} and .json")
        return 0

    if args.ablation:
        table, arms = run_ablation(golden, days=days, client=client, contract=args.contract, top_k=args.top_k, offline=args.offline,
                                   runs=args.ablation_runs, embedding_function=ef, embedding=args.embedding)
        print("\n" + table)
        (out / f"{stamp}-ablation.md").write_text(table + "\n", encoding="utf-8")
        (out / f"{stamp}-ablation.json").write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "runs": args.ablation_runs,
                                                                "offline": args.offline, "embedding": args.embedding, "arms": arms}, indent=1), encoding="utf-8")
        print(f"wrote {out / (stamp + '-ablation.md')} and .json")
        return 0

    chroma = I.make_client()
    coll, _ = I.build(chroma, days, name="eval", embedding_function=ef)
    if args.rerank_compare:
        table, rows = run_rerank_compare(golden, days=days, collection=coll, top_k=args.top_k, reranker_name=args.rerank if args.rerank != "none" else "lexical")
        print("\n" + table)
        name = args.rerank if args.rerank != "none" else "lexical"
        (out / f"{stamp}-rerank-{name}.md").write_text(table + "\n", encoding="utf-8")
        (out / f"{stamp}-rerank-{name}.json").write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "reranker": name,
                                                                    "embedding": args.embedding, "rows": rows}, indent=1), encoding="utf-8")
        return 0
    reranker = make_reranker(args.rerank)
    records = []
    for g in golden:
        r = run_question(g, days=days, collection=coll, client=client, contract=args.contract, top_k=args.top_k, offline=args.offline, reranker=reranker)
        records.append(r)
        print(f"  {g['id']:4} {g['kind']:12} route={r['route']:12} recall@5={r['recall'].get('5')} facts={r['facts']} {'HARD FAIL' if r['hard_fail'] else ''}", flush=True)
    agg = aggregate_scores(records)
    table = render(records, agg, offline=args.offline, contract=args.contract, embedding=args.embedding, rerank=args.rerank)
    stamp += "" if args.rerank == "none" else f"-rerank-{args.rerank}"
    print("\n" + table)
    (out / f"{stamp}.md").write_text(table + "\n", encoding="utf-8")
    (out / f"{stamp}.json").write_text(json.dumps({"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "offline": args.offline,
                                                   "contract": args.contract, "embedding": args.embedding, "model": MODEL, "aggregate": agg, "records": records},
                                                  indent=1, default=str), encoding="utf-8")
    print(f"wrote {out / (stamp + '.md')} and .json")
    return 1 if agg["hard_fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
