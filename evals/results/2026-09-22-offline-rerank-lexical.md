# Reranking · 2026-09-22 · reranker `lexical` · 7 semantic questions · retrieval only

Plain: dense top-k. Reranked: dense top-20, then the reranker keeps the top k.

| Measure | Plain top-k | Retrieve 20, rerank to k |
| --- | --- | --- |
| Recall@5 | 89% | 89% |
| Mean ms per question (retrieval, plus reranking) | 226.5714 | 224.7143 (reranker alone 0.0) |

| Q | Plain | Reranked |
| --- | --- | --- |
| S01 | 100% | 100% |
| S02 | 100% | 100% |
| S03 | 50% | 50% |
| S04 | 100% | 100% |
| S05 | 100% | 100% |
| S06 | 71% | 71% |
| S07 | 100% | 100% |

Gain: +0.0 points of recall@5.
