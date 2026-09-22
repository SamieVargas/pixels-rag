"""Embedding models, decided with data and with a privacy line.

Three local arms and one API arm. The default stays local whatever the
numbers say, because the API arm sends every chunk's text to a third party;
it is opt-in, and it says so."""

import os
import time

from core import index as I

ARMS = {
    "minilm": {"kind": "local", "model": "all-MiniLM-L6-v2", "needs": "nothing (Chroma's default ONNX model)"},
    "bge-small": {"kind": "local", "model": "BAAI/bge-small-en-v1.5", "needs": "pip install sentence-transformers, and the model download"},
    "e5-small": {"kind": "local", "model": "intfloat/e5-small-v2", "needs": "pip install sentence-transformers, and the model download"},
    "openai": {"kind": "api", "model": "text-embedding-3-small", "needs": "pip install openai and OPENAI_API_KEY; sends every chunk's text to OpenAI"},
    "hash": {"kind": "test", "model": "hash-test", "needs": "nothing; the offline test embedder, not a contender"},
}
DEFAULT = "minilm"


def make_embedding_function(name: str):
    """Return (embedding_function, label). Raises RuntimeError with the reason
    when an arm cannot run here, so the table can say 'not run' and why."""
    if name not in ARMS:
        raise ValueError(f"unknown embedding arm {name!r}; choose from {', '.join(ARMS)}")
    arm = ARMS[name]
    if name == "hash":
        return I.HashEmbedding(), arm["model"]
    if name == "minilm":
        return None, arm["model"]  # Chroma's default
    if name in ("bge-small", "e5-small"):
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            return SentenceTransformerEmbeddingFunction(model_name=arm["model"]), arm["model"]
        except Exception as e:
            raise RuntimeError(f"{name} needs {arm['needs']} ({type(e).__name__}: {str(e)[:80]})") from e
    if name == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(f"openai needs {arm['needs']}")
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            return OpenAIEmbeddingFunction(api_key=os.environ["OPENAI_API_KEY"], model_name=arm["model"]), arm["model"]
        except Exception as e:
            raise RuntimeError(f"openai needs {arm['needs']} ({type(e).__name__})") from e
    raise ValueError(name)


def run_arm(name: str, *, days, questions, score, top_k: int = 8):
    """Build an index with the arm's embedder, time it, run the semantic
    questions, and return recall@5 with the mean query latency. `score` is
    the runner's retrieval scorer, `questions` are (question, plan, expected)."""
    try:
        ef, label = make_embedding_function(name)
    except RuntimeError as e:
        return {"arm": name, "model": ARMS[name]["model"], "ran": False, "reason": str(e)}
    client = I.make_client()
    t0 = time.time()
    coll, _ = I.build(client, days, name=f"emb_{name.replace('-', '_')}", embedding_function=ef)
    build_s = round(time.time() - t0, 2)
    from core.pipeline import gather
    recalls, ms = [], []
    for question, plan, expected in questions:
        t0 = time.time()
        ev = gather(question, plan, days=days, collection=coll, top_k=top_k)
        ms.append(int((time.time() - t0) * 1000))
        recalls.append(score(expected, ev["retrieved"])["recall"].get("5"))
    recalls = [r for r in recalls if r is not None]
    return {"arm": name, "model": label, "ran": True, "build_s": build_s, "recall5": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "query_ms": round(sum(ms) / len(ms), 1) if ms else None, "n": len(questions)}
