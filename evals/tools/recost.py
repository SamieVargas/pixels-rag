"""Re-price the eval results already on disk, without re-running anything.

    python evals/tools/recost.py                  # every evals/results/*.json that has per-question records
    python evals/tools/recost.py --dir <folder>   # another folder (the tests run it on a temporary copy)

For each main-run results file (the ones with `records`), the cost of every
question is computed from its recorded token counts at the list prices in
core/contracts.py, `cost_usd` is written into each record and `cost` into the
aggregate, and the matching .md gets the two cost rows of its summary table
and the USD column of its question table. A run whose records carry no token
counts is an offline one and is priced at $0 with the reason. Every other line
of both files is left byte for byte as it was, and running the tool twice
changes nothing the second time. Files without per-question records (the
ablation, follow-up, reranking and embedding tables) carry no token counts
and are skipped by name.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from run import QUESTION_HEADER, QUESTION_SEPARATOR, cost_rows, cost_summary, usage_cost, usd  # noqa: E402

RESULTS = ROOT / "evals" / "results"
COST_ROWS = ("| Cost per question (mean) |", "| Cost for the whole run |")


def price(data: dict) -> dict:
    """Write cost_usd into every record and cost into the aggregate, in place."""
    for r in data["records"]:
        r["cost_usd"] = usage_cost(r.get("tokens")) if r.get("tokens") else 0.0
    data["aggregate"]["cost"] = cost_summary(data["records"])
    return data


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def patch_markdown(md: str, data: dict) -> str:
    """Replace or insert the cost rows after the tokens row, and set the USD
    column of the question table. Nothing else is touched."""
    lines = [l for l in md.split("\n") if not l.startswith(COST_ROWS)]
    tokens_at = next((i for i, l in enumerate(lines) if l.startswith("| Mean tokens in / out")), None)
    if tokens_at is None:
        raise ValueError("no tokens row to put the cost rows after")
    lines[tokens_at + 1:tokens_at + 1] = cost_rows(data["aggregate"]["cost"])
    head = next((i for i, l in enumerate(lines) if l.startswith("| Q | Kind | Route |")), None)
    if head is None:
        raise ValueError("no question table")
    lines[head], lines[head + 1] = QUESTION_HEADER, QUESTION_SEPARATOR
    n = QUESTION_HEADER.count("|") - 1
    by = {r["id"]: r for r in data["records"]}
    for j in range(head + 2, len(lines)):
        if not lines[j].startswith("| "):
            break
        cells = _cells(lines[j])
        cell = usd(by[cells[0]]["cost_usd"]) if cells[0] in by else "n/a"
        cells = cells[:n - 1] + [cell] if len(cells) >= n else cells + [cell]
        lines[j] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines)


def recost(folder=RESULTS) -> dict:
    """Re-price every results file in the folder. Returns what changed and what was skipped."""
    out = {"changed": [], "unchanged": [], "skipped": []}
    for jp in sorted(Path(folder).glob("*.json")):
        raw = jp.read_text(encoding="utf-8")
        data = json.loads(raw)
        if "records" not in data or "aggregate" not in data:
            out["skipped"].append(jp.name)
            continue
        new_json = json.dumps(price(data), indent=1, default=str)
        mp = jp.with_suffix(".md")
        new_md = patch_markdown(mp.read_text(encoding="utf-8"), data) if mp.exists() else None
        touched = []
        if new_json != raw:
            jp.write_text(new_json, encoding="utf-8")
            touched.append(jp.name)
        if new_md is not None and new_md != mp.read_text(encoding="utf-8"):
            mp.write_text(new_md, encoding="utf-8")
            touched.append(mp.name)
        (out["changed"] if touched else out["unchanged"]).append(jp.stem)
        c = data["aggregate"]["cost"]
        print(f"{jp.stem}: {usd(c['per_question'])} per question, {usd(c['total'])} for the run"
              + (f" ({c['note']})" if c.get("note") else "") + (" · rewrote " + ", ".join(touched) if touched else " · already priced"))
    for name in out["skipped"]:
        print(f"{name}: skipped, no per-question token counts")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="re-price the eval results on disk from their recorded tokens")
    p.add_argument("--dir", default=str(RESULTS), help="folder of results .json/.md pairs")
    args = p.parse_args(argv)
    recost(args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
