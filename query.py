"""Retrieval + generation: natural language question -> top-k day chunks -> Claude."""

import os

import anthropic

# Haiku 4.5 — fast and cheap, well-suited to synthesizing over a small retrieved set.
MODEL = "claude-haiku-4-5-20251001"


def query(question: str, collection, top_k: int = 8) -> dict:
    """
    Retrieve the top-k relevant day chunks and pass them to Claude for synthesis.

    Returns a dict with the answer text and the list of source date ids, so the
    caller can render them however it likes.
    """
    # Retrieve
    results = collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    retrieved_chunks = results["documents"][0]   # list of chunk texts
    retrieved_ids = results["ids"][0]            # list of date strings

    if not retrieved_chunks:
        return {"answer": "No data is indexed yet. Run `python main.py --rebuild` first.", "sources": []}

    # Build context block
    context = "\n\n---\n\n".join(
        f"[{retrieved_ids[i]}]\n{retrieved_chunks[i]}"
        for i in range(len(retrieved_chunks))
    )

    prompt = f"""You are analyzing personal behavioral and biometric data to answer a question.
You have access to {len(retrieved_chunks)} days of data retrieved as relevant to the question.

RETRIEVED DAYS:
{context}

QUESTION: {question}

Instructions:
- Answer directly based only on the retrieved data
- Cite specific dates when making claims (e.g. "on 5/29/2026...")
- Note patterns across multiple days when they exist
- Be honest about limitations — if the retrieved days don't fully answer the question, say so
- Keep the answer concise and grounded in the actual data
- Do not speculate beyond what the data shows"""

    client = anthropic.Anthropic()

    message = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = next((b.text for b in message.content if b.type == "text"), "")

    return {"answer": answer, "sources": retrieved_ids}
