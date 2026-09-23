"""Cost per run: the price table, cost_usd, the runner's cost fields, and the
re-pricing tool over the results already on disk. No key, no model download.

    python tests/test_cost.py
"""

import json
import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "evals"))
sys.path.insert(0, str(ROOT / "evals" / "tools"))

from core.contracts import MODEL, PRICES, PRICES_READ_ON, cost_usd  # noqa: E402
from stubs import FakeClient, decision, answer  # noqa: E402
import recost  # noqa: E402
import run  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def read_pair(folder, suffix=".md"):
    md_name = next(f for f in os.listdir(folder) if f.endswith(suffix))
    md = Path(folder, md_name).read_text(encoding="utf-8")
    data = json.loads(Path(folder, md_name[:-3] + ".json").read_text(encoding="utf-8"))
    return md, data, md_name


def strip_costs(folder):
    """Turn a freshly written pair back into the pre-cost format: no cost fields
    in the JSON, no cost rows and no USD column in the markdown."""
    md, data, md_name = read_pair(folder)
    for r in data["records"]:
        del r["cost_usd"]
    del data["aggregate"]["cost"]
    Path(folder, md_name[:-3] + ".json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    out, in_table = [], False
    for line in md.split("\n"):
        if line.startswith(recost.COST_ROWS):
            continue
        if line.startswith("| Q | Kind |"):
            in_table = True
        if in_table and line.startswith("|"):
            line = "|".join(line.split("|")[:-2]) + "|"
        out.append(line)
    Path(folder, md_name).write_text("\n".join(out), encoding="utf-8")


def main():
    print("the price table")
    check("the model the pipeline calls has a price", MODEL in PRICES)
    check("the prices carry the date they were read", date.fromisoformat(PRICES_READ_ON) <= date.today())
    check("a million input tokens cost the input price", cost_usd(MODEL, 1_000_000, 0) == PRICES[MODEL]["input"])
    check("a million output tokens cost the output price", cost_usd(MODEL, 0, 1_000_000) == PRICES[MODEL]["output"])
    check("S01 of the 2026-09-22 run, 4297 in and 901 out, is $0.0088", abs(cost_usd(MODEL, 4297, 901) - 0.008802) < 1e-9)
    check("no tokens, no cost", cost_usd(MODEL, 0, 0) == 0.0 and cost_usd(MODEL, None, None) == 0.0)
    try:
        cost_usd("claude-not-a-model", 1, 1)
        check("an unpriced model raises and names the table", False)
    except KeyError as e:
        check("an unpriced model raises and names the table", "PRICES" in str(e))

    print("the offline runner prices every question at zero and says why")
    with tempfile.TemporaryDirectory() as tmp:
        code = run.main(["--offline", "--only", "S01,F05,U01", "--embedding", "hash", "--out", tmp])
        md, data, _ = read_pair(tmp)
        check("exit 0 and every record carries cost_usd 0.0", code == 0 and all(r["cost_usd"] == 0.0 for r in data["records"]))
        c = data["aggregate"]["cost"]
        check("the aggregate cost is zero with the reason", c["per_question"] == 0.0 and c["total"] == 0.0 and c["priced"] == 0 and "no model call" in c["note"])
        check("the markdown rows say $0.0000 and why", "| Cost per question (mean) | $0.0000 (offline" in md and "| Cost for the whole run | $0.0000 (offline" in md)
        check("the question table has a USD column at $0.0000", "| ms | USD |" in md and md.count("| $0.0000 |") == 3)

    print("the keyed runner prices each question from its recorded tokens")
    golden = {g["id"]: g for g in run.load_golden(ROOT / "evals" / "golden.jsonl")}
    fc = FakeClient([golden["F05"]["plan"], answer("One day.", golden["F05"]["expected_dates"]), decision("unanswerable")])
    with tempfile.TemporaryDirectory() as tmp:
        code = run.main(["--only", "F05,U01", "--embedding", "hash", "--out", tmp], client=fc)
        md, data, _ = read_pair(tmp)
        by = {r["id"]: r for r in data["records"]}
        # the stub reports 120 in and 60 out per call; F05 made two calls, U01 one
        check("F05, router plus answer, costs two calls", by["F05"]["tokens"] == {"input_tokens": 240, "output_tokens": 120} and by["F05"]["cost_usd"] == round(cost_usd(MODEL, 240, 120), 8))
        check("U01, router only, costs one call", by["U01"]["tokens"] == {"input_tokens": 120, "output_tokens": 60} and by["U01"]["cost_usd"] == round(cost_usd(MODEL, 120, 60), 8))
        c = data["aggregate"]["cost"]
        check("the aggregate carries mean, total, the model and the price date", c["per_question"] == round((by["F05"]["cost_usd"] + by["U01"]["cost_usd"]) / 2, 4)
              and c["total"] == round(by["F05"]["cost_usd"] + by["U01"]["cost_usd"], 4) and c["priced"] == 2 and c["model"] == MODEL and c["prices_read_on"] == PRICES_READ_ON and c["note"] is None)
        check("the markdown rows carry the two figures", f"| Cost per question (mean) | {run.usd(c['per_question'])} |" in md
              and f"| Cost for the whole run | {run.usd(c['total'])} (2 questions, `{MODEL}` list prices read {PRICES_READ_ON}) |" in md)
        check("each question row ends with its cost", any(l.startswith("| F05 |") and l.endswith(f"| {run.usd(by['F05']['cost_usd'])} |") for l in md.splitlines()))

        print("recost brings a pre-cost pair up to the runner's output and is idempotent")
        strip_costs(tmp)
        old_md, old_data, _ = read_pair(tmp)
        check("the fixture pair really lost its costs", "cost_usd" not in old_data["records"][0] and "| USD |" not in old_md and "| Cost" not in old_md)
        out = recost.recost(tmp)
        new_md, new_data, _ = read_pair(tmp)
        check("the json got its costs back", [r["cost_usd"] for r in new_data["records"]] == [by["F05"]["cost_usd"], by["U01"]["cost_usd"]] and new_data["aggregate"]["cost"] == c)
        check("the patched markdown equals what the runner wrote", new_md == md and out["changed"] and not out["skipped"])
        again = recost.recost(tmp)
        check("a second pass changes nothing", again["unchanged"] and not again["changed"] and read_pair(tmp)[0] == md)

    print("the committed results are already priced, and the tables without token counts are skipped")
    with tempfile.TemporaryDirectory() as tmp:
        for p in (ROOT / "evals" / "results").iterdir():
            shutil.copy(p, tmp)
        before = {p.name: p.read_bytes() for p in Path(tmp).iterdir()}
        out = recost.recost(tmp)
        after = {p.name: p.read_bytes() for p in Path(tmp).iterdir()}
        check("recost over a copy of evals/results changes no file", before == after and not out["changed"])
        check("the keyed and offline runs were priced, the rest skipped by name",
              "2026-09-22" in out["unchanged"] and "2026-09-22-offline" in out["unchanged"] and all(n in " ".join(out["skipped"]) for n in ("ablation", "followups", "embeddings", "rerank")))
    keyed = json.loads((ROOT / "evals" / "results" / "2026-09-22.json").read_text(encoding="utf-8"))
    check("the keyed run's cost is the sum of its recorded tokens at the list prices",
          keyed["aggregate"]["cost"]["total"] == round(sum(cost_usd(MODEL, r["tokens"]["input_tokens"], r["tokens"]["output_tokens"]) for r in keyed["records"]), 4))

    print("the ablation and follow-up rows carry tokens and cost")
    with tempfile.TemporaryDirectory() as tmp:
        fc = FakeClient([answer("Nothing to add.", [], claims=[])] * 2)
        code = run.main(["--ablation", "--ablation-runs", "1", "--only", "S01", "--embedding", "hash", "--out", tmp], client=fc)
        md, data, _ = read_pair(tmp)
        one = round(cost_usd(MODEL, 120, 60), 8)
        check("each keyed ablation row carries its tokens and cost", code == 0 and all(r["tokens"] == {"input_tokens": 120, "output_tokens": 60} and r["cost_usd"] == one for a in data["arms"].values() for r in a["rows"]))
        check("each arm sums its cost and the table shows it", data["arms"]["A"]["cost_usd"] == round(one, 4) and f"| Cost for the arm, all runs | {run.usd(one)} | {run.usd(one)} |" in md)
        code = run.main(["--offline", "--ablation", "--ablation-runs", "1", "--only", "S01", "--embedding", "hash", "--out", tmp])
        md = next(Path(tmp, f).read_text(encoding="utf-8") for f in os.listdir(tmp) if f.endswith("-offline-ablation.md"))
        check("the offline ablation says $0 and why", "| Cost for the arm, all runs | $0.0000 (offline: no model call) | $0.0000 (offline: no model call) |" in md)
    with tempfile.TemporaryDirectory() as tmp:
        fus = run.load_golden(ROOT / "evals" / "golden.jsonl", follow_ups=True)
        fc = FakeClient([fus[0]["plan"], answer("Hot yoga days averaged 3.3 against 3.1.", [], claims=[]), decision("unanswerable"),
                         fus[1]["plan"], answer("Eight days.", fus[1]["expected_dates"][:1]), decision("unanswerable")])
        code = run.main(["--followups", "--embedding", "hash", "--out", tmp], client=fc)
        md, data, _ = read_pair(tmp, "-followups.md")
        check("each follow-up row carries tokens and cost", code == 0 and all(r["tokens"] and r["cost_usd"] > 0 for r in data["rows"]))
        check("the table has a USD column and a total", "| Rewritten as | USD |" in md and "Cost for the table: $" in md)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: " + ", ".join(FAILS))
        sys.exit(1)
    print("ALL CHECKS PASSED")


def test_all():
    """pytest entry point: the checks above, one test per file. A failing
    check makes main() exit 1, which pytest reports as this test failing."""
    main()


if __name__ == "__main__":
    main()
