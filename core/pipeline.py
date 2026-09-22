"""One question, end to end: route, gather evidence, answer under the
contract, validate, report. `plan` lets the evals bypass the router with the
golden plan so retrieval can be scored without a key."""

import time

from core import filters as F
from core import index as I
from core.aggregate import aggregate, render_table
from core.answer import generate_answer
from core.contracts import MODEL
from core.days import latest_date
from core.rerank import CANDIDATES, rerank
from core.router import classify, resolve

EVIDENCE_CAP = 30


def _evidence(chunks: list[dict]) -> str:
    shown = chunks[:EVIDENCE_CAP]
    text = "\n\n---\n\n".join(f"[{c['id']}]\n{c['text']}" for c in shown)
    if len(chunks) > EVIDENCE_CAP:
        text += f"\n\n---\n\n({len(chunks) - EVIDENCE_CAP} more matching days not shown)"
    return text


def gather(question: str, plan: dict, *, days, collection, top_k: int = 8, reranker=None, candidates: int = CANDIDATES) -> dict:
    """The evidence for a resolved plan, with no model call: the retrieved
    chunks (semantic), the matching days (filter), the computed table
    (aggregate) or nothing (unanswerable). With a reranker, the dense search
    returns `candidates` and the reranker keeps `top_k`."""
    route = plan["kind"]
    out = {"route": route, "retrieved": [], "table": None, "evidence": None, "allowed_texts": [], "covered_ids": [], "rerank_ms": None}
    if route == "unanswerable":
        return out
    if route == "aggregate":
        a = plan["aggregate"]
        table = aggregate(days, metric=a["metric"], stat=a["stat"], group_by=a["group_by"], filters=plan["filters"], date_range=plan["date_range"])
        md = render_table(table)
        covered = sorted({d for g in table["groups"] for d in g["dates"]})
        out.update(table=table, retrieved=[{"id": d, "score": None, "days": [d]} for d in covered], covered_ids=covered,
                   evidence=f"COMPUTED TABLE (exact; narrate it, compute nothing else):\n{md}"
                   + (f"\nDate range: {plan['date_range'][0]} to {plan['date_range'][1]}" if plan["date_range"] else ""),
                   allowed_texts=[md])
        return out
    if route == "filter":
        matched = F.apply(days, plan["filters"], plan["date_range"])
        chunks = [{"id": d["date"], "text": d["text"], "score": None, "days": [d["date"]]} for d in matched]
    else:
        allowed = None
        if plan["filters"] or plan["date_range"]:
            allowed = [d["date"] for d in F.apply(days, plan["filters"], plan["date_range"])]
        q = plan["query"] or question
        if reranker is not None:
            chunks, ms = rerank(q, I.retrieve(collection, q, max(candidates, top_k), allowed_ids=allowed), reranker, top_k)
            out["rerank_ms"] = ms
        else:
            chunks = I.retrieve(collection, q, top_k, allowed_ids=allowed)
    out.update(retrieved=chunks, covered_ids=[c["id"] for c in chunks], evidence=_evidence(chunks) if chunks else None,
               allowed_texts=[c["text"] for c in chunks[:EVIDENCE_CAP]])
    return out


NO_MATCH = {"filter": "No logged day matches those conditions.", "semantic": "Nothing in the log matches that question."}


def ask(question: str, *, days, collection, client, today=None, top_k: int = 8, contract="native",
        history=None, plan=None, model=MODEL, reranker=None) -> dict:
    t0 = time.time()
    today = today or latest_date(days)
    calls = []
    if plan is None:
        decision, meta = classify(client, question, today=today, history=history, contract=contract, model=model)
        calls.append(meta)
        plan = resolve(decision, today)
    else:
        plan = resolve(plan, today)
    g = gather(question, plan, days=days, collection=collection, top_k=top_k, reranker=reranker)
    route, retrieved = g["route"], g["retrieved"]
    ameta = None
    if route == "unanswerable":
        answer = {"answer": "The log does not track that, so there is nothing to answer from.", "claims": [],
                  "cited_dates": [], "unanswerable": True, "why_unanswerable": "not logged"}
    elif g["evidence"] is None:
        answer = {"answer": NO_MATCH.get(route, "No evidence."), "claims": [], "cited_dates": [],
                  "unanswerable": True, "why_unanswerable": "no matching days"}
    else:
        answer, ameta = generate_answer(client, question, evidence=g["evidence"], route=route, retrieved_ids=g["covered_ids"],
                                        allowed_texts=g["allowed_texts"], contract=contract, model=model)
        calls.append(ameta)
    retrieved_ids = [c["id"] for c in retrieved]
    cited = set(answer.get("cited_dates") or [])
    covered_dates = {d for c in retrieved for d in c["days"]}
    counts = {"retrieved": len(retrieved), "cited": len(cited),
              "retrieved_uncited": len([i for i in retrieved_ids if i not in cited]),
              "cited_unretrieved": len([d for d in cited if d not in covered_dates and d not in retrieved_ids])}
    validation = {"ok": ameta["valid"] if ameta else True, "violations": ameta["violations"] if ameta else [],
                  "first_violations": ameta["first_violations"] if ameta else [], "retries": ameta["retries"] if ameta else 0,
                  "parse_path": ameta["parse_path"] if ameta else None}
    usage = {"input_tokens": sum(c.get("input_tokens", 0) for c in calls), "output_tokens": sum(c.get("output_tokens", 0) for c in calls)}
    return {"question": question, "route": route, "plan": plan, "retrieved": [{k: v for k, v in c.items() if k != "text"} for c in retrieved],
            "table": g["table"], "answer": answer, "validation": validation, "counts": counts, "usage": usage, "rerank_ms": g["rerank_ms"],
            "model_calls": len(calls), "calls": calls, "latency_ms": int((time.time() - t0) * 1000)}
