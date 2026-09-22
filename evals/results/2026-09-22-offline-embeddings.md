# Embedding models · 2026-09-22 · 7 semantic questions · retrieval only

The default stays local whatever the numbers say. The `openai` arm sends every chunk's text to OpenAI and is opt-in.

| Arm | Model | Recall@5 | Index build | Query latency |
| --- | --- | --- | --- | --- |
| minilm (default) | `all-MiniLM-L6-v2` | 89% | 3.85 s | 210.4 ms |
| bge-small | `BAAI/bge-small-en-v1.5` | not run | | bge-small needs pip install sentence-transformers, and the model download (ValueError: The sentence_transformers python package is not installed. Please install it wit) |
| e5-small | `intfloat/e5-small-v2` | not run | | e5-small needs pip install sentence-transformers, and the model download (ValueError: The sentence_transformers python package is not installed. Please install it wit) |
| openai | `text-embedding-3-small` | not run | | openai needs pip install openai and OPENAI_API_KEY; sends every chunk's text to OpenAI |
