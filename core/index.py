"""The ChromaDB index. Day chunks are v1's text with typed metadata; the
day+week level adds the rollups from core/rollup.py for the ablation. The
embedding function is a parameter so the tests can pass a deterministic one
and never download a model."""

import chromadb
from chromadb.api.types import EmbeddingFunction

from core.days import metadata_for
from core.rollup import week_chunks

COLLECTION = "pixels"
LEVELS = ("day", "day+week")


class HashEmbedding(EmbeddingFunction):
    """A bag-of-words hash embedding: deterministic, offline, good enough to
    tell 'migraine' from 'trail running'. For tests only."""

    DIMS = 1024

    def __init__(self):
        pass

    def __call__(self, input):
        out = []
        for text in input:
            vec = [0.0] * self.DIMS
            for tok in str(text).lower().replace(",", " ").replace(":", " ").split():
                h = 0
                for ch in tok:
                    h = (h * 31 + ord(ch)) & 0xFFFFFFFF
                vec[h % self.DIMS] += 1.0
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    @staticmethod
    def name():
        return "hash-test"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()

    @staticmethod
    def validate_config(config):
        return None

    @staticmethod
    def validate_config_update(old_config, new_config):
        return None


def make_client(path=None):
    return chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()


def chunks_for(days: list[dict], level: str = "day") -> list[dict]:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    chunks = [{"id": d["date"], "text": d["text"], "metadata": metadata_for(d), "days": [d["date"]]} for d in days]
    if level == "day+week":
        chunks += week_chunks(days)
    return chunks


def build(client, days: list[dict], *, name: str = COLLECTION, level: str = "day", embedding_function=None):
    """Build the collection from scratch. Returns (collection, chunks)."""
    try:
        client.delete_collection(name)
    except Exception:
        pass
    kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
    coll = client.create_collection(name=name, metadata={"hnsw:space": "cosine", "level": level}, **kwargs)
    chunks = chunks_for(days, level)
    coll.add(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks], metadatas=[c["metadata"] for c in chunks])
    return coll, chunks


def load(client, *, name: str = COLLECTION, embedding_function=None):
    kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
    return client.get_collection(name, **kwargs)


def retrieve(collection, query_text: str, k: int, allowed_ids=None) -> list[dict]:
    """Dense top-k, restricted to `allowed_ids` when the router narrowed the
    days. An empty allowed list means nothing can match, so nothing is asked.
    Week chunks are allowed when any of their days is."""
    kwargs = {}
    if allowed_ids is not None:
        allowed = set(allowed_ids)
        if not allowed:
            return []
        ids = list(allowed)
        for wk in _week_ids(collection):
            if any(d in allowed for d in wk["days"]):
                ids.append(wk["id"])
        kwargs["ids"] = ids
        k = min(k, len(ids))
    if k <= 0:
        return []
    res = collection.query(query_texts=[query_text], n_results=k, include=["documents", "metadatas", "distances"], **kwargs)
    out = []
    for i, cid in enumerate(res["ids"][0]):
        md = res["metadatas"][0][i] or {}
        out.append({"id": cid, "text": res["documents"][0][i], "metadata": md,
                    "score": round(1 - float(res["distances"][0][i]), 4),
                    "days": _days_of(cid, md)})
    return out


def _days_of(cid: str, md: dict) -> list[str]:
    if md.get("level") == "week":
        from datetime import date, timedelta
        start = date.fromisoformat(md["date"])
        return [(start + timedelta(days=i)).isoformat() for i in range(7)]
    return [cid]


def _week_ids(collection):
    res = collection.get(where={"level": "week"}, include=["metadatas"])
    return [{"id": cid, "days": _days_of(cid, md)} for cid, md in zip(res["ids"], res["metadatas"])]
