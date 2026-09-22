# Chunk granularity ablation · 2026-09-22 · 7 semantic questions × 1 run(s) per arm · embedding `default` · offline

Arm A indexes one chunk per day. Arm B adds one deterministic rollup chunk per week, tagged `level: week`; a retrieved week counts as covering its seven days.

| Measure | A · day chunks | B · day + week rollups |
| --- | --- | --- |
| Recall@5 (expected days covered) | 89% | 93% |
| Expected facts in the answer | n/a offline | n/a offline |
| Week chunks retrieved (total) | 0 | 14 |

| Q | A recall@5 | B recall@5 |
| --- | --- | --- |
| S01 | 100% | 100% |
| S02 | 100% | 100% |
| S03 | 50% | 50% |
| S04 | 100% | 100% |
| S05 | 100% | 100% |
| S06 | 71% | 100% |
| S07 | 100% | 100% |
