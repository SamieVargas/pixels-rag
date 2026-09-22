"""
Life in Pixels RAG — ask natural language questions about your behavioral data.

  python main.py "What happened on the day I had a migraine in June?"
  python main.py --rebuild                      # fetch the export and rebuild the index
  python main.py --source fixture --rebuild     # the committed synthetic days instead
  python main.py --dry-run                      # first 3 chunks, no API calls
  python main.py --stats                        # corpus stats, no API calls
  python main.py --since 2026-08-01             # upsert the days logged since a date, idempotent
  python main.py --status                       # how far behind the source the index is
  python main.py --v1 "question"                # the v1 path: top-k, prompt-only citations

v2 routes each question (semantic, filter, aggregate, unanswerable), answers
under a contract with citations enforced in code, and prints what was
retrieved against what was cited.
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from chunk import row_to_chunk
from core import index as I
from core import store
from core.aggregate import render_table
from core.quality import IngestError, render_report, validate_rows
from core.days import normalize_rows
from ingest import fetch_pixels_data

console = Console()
ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "days.json"


def load_rows(source: str, days: int) -> list[dict]:
    if source == "fixture":
        return json.loads(FIXTURE.read_text(encoding="utf-8"))["rows"]
    return fetch_pixels_data(days=days)


def show_dry_run(rows: list[dict], n: int = 3) -> None:
    console.print(f"\n[bold]Dry run — showing first {n} of {len(rows)} chunks[/bold]\n")
    for row in rows[:n]:
        console.print(Panel(row_to_chunk(row), title=f"[cyan]{row['date']}[/cyan]", border_style="cyan"))


def show_stats(rows: list[dict], days: int) -> None:
    """Corpus stats plus the validation report the ingest would apply."""
    ratings = [r.get("ratingNum", 0) for r in rows if r.get("ratingNum", 0) > 0]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    report = validate_rows(rows)
    table = Table(title="Life in Pixels — corpus stats", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("Days requested", str(days))
    table.add_row("Days logged", str(len(rows)))
    table.add_row("Days missing", str(max(days - len(rows), 0)))
    if rows:
        dates = sorted(r["date"] for r in rows)
        table.add_row("Date range", f"{dates[0]} → {dates[-1]}")
    table.add_row("Average rating", f"{avg_rating:.2f} / 5")
    table.add_row("Validation", "[green]OK[/green]" if report["ok"] else f"[red]{len(report['errors'])} error(s), ingest would refuse[/red]")
    for k in ("missing_day", "out_of_range", "not_a_number", "empty_day"):
        table.add_row(f"  {k.replace('_', ' ')}", str(report["warning_counts"].get(k, 0)))
    console.print(table)
    for e in report["errors"]:
        console.print(f"  [red]- {e['date']}: {e['detail']}[/red]")
    for w in report["warnings"][:15]:
        console.print(f"  [dim]- {w['date']}: {w['detail']}[/dim]")
    if len(report["warnings"]) > 15:
        console.print(f"  [dim]({len(report['warnings']) - 15} more in the ingest report)[/dim]")


def rebuild(db: Path, source: str, days: int) -> None:
    console.print(f"[bold]Loading {source} data...[/bold]")
    rows = load_rows(source, days)
    console.print("[bold]Building the index...[/bold]")
    try:
        _, recs = store.rebuild(db, rows)
    except IngestError as e:
        console.print(f"[red]Ingest refused: {e}[/red]")
        sys.exit(1)
    console.print(f"[green]Index built: {len(recs)} days indexed at {db}[/green]")


def ingest_since(db: Path, source: str, since: str) -> None:
    rows = [r for r in load_rows(source, store.since_days(since)) if str(r["date"]) >= since]
    try:
        report = store.ingest(db, rows)
    except IngestError as e:
        console.print(f"[red]Ingest refused: {e}[/red]")
        sys.exit(1)
    q = report["quality"]
    if q["warnings"]:
        console.print(f"[yellow]{len(q['warnings'])} warning(s) in the ingest report: {q['warning_counts']}[/yellow]")
    if report["rebuilt"]:
        console.print("[yellow]Index rebuilt from scratch:[/yellow] " + ("; ".join(report["reasons"]) or "no index existed"))
    console.print(f"[green]Ingested {report['upserted']} days since {since}: {report['new']} new, {report['updated']} updated[/green]")


def show_status(db: Path, source: str, days: int) -> None:
    try:
        rows = load_rows(source, days)
    except Exception as e:
        console.print(f"[yellow]Could not reach the source ({e}); showing the index alone.[/yellow]")
        rows = None
    st = store.status(db, rows)
    table = Table(title="Index status", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("Days indexed", str(st["indexed_days"]))
    table.add_row("Newest indexed day", st["newest_indexed"] or "none")
    table.add_row("Newest source day", st["newest_source"] or "unknown")
    table.add_row("Days behind the source", "unknown" if st["days_behind"] is None else str(st["days_behind"]))
    if st.get("missing_dates"):
        table.add_row("First missing dates", ", ".join(st["missing_dates"]))
    if st["stamp"]:
        table.add_row("Chunk template", f"{st['stamp']['chunk_template_version']} ({st['stamp']['chunk_template_hash']})")
        table.add_row("Embedding model", str(st["stamp"]["embedding_model"]))
    if st["mismatch"]:
        table.add_row("[red]Rebuild needed[/red]", "; ".join(st["mismatch"]))
    console.print(table)


def load_corpus(db: Path):
    rows = json.loads((db / "days.json").read_text(encoding="utf-8"))["rows"]
    return normalize_rows(rows), I.load(I.make_client(str(db)))


def render_result(r: dict) -> None:
    a = r["answer"]
    console.print(Panel(Markdown(a["answer"] or "_(no answer)_"), title=f"Answer · route: {r['route']}", border_style="green" if r["validation"]["ok"] else "yellow"))
    if a["unanswerable"]:
        console.print(f"[dim]Abstained: {a['why_unanswerable']}[/dim]")
    for c in a["claims"]:
        console.print(f"  • {c['text']}  [dim]({', '.join(c['dates']) or 'no dates'})[/dim]")
    if r["table"]:
        console.print(Markdown(render_table(r["table"])))
    k = r["counts"]
    console.print(f"\n[dim]retrieved {k['retrieved']} · cited {k['cited']} · retrieved-but-uncited {k['retrieved_uncited']} · "
                  f"cited-but-unretrieved {k['cited_unretrieved']} · validator retries {r['validation']['retries']} · parse {r['validation']['parse_path'] or 'n/a'}"
                  f" · {r['usage']['input_tokens']}+{r['usage']['output_tokens']} tokens · {r['latency_ms']} ms[/dim]")
    if r["plan"]["notes"]:
        console.print(f"[dim]router notes: {'; '.join(r['plan']['notes'])}[/dim]")
    if not r["validation"]["ok"]:
        console.print("[red]The answer did not pass validation after a retry:[/red]")
        for v in r["validation"]["violations"]:
            console.print(f"  [red]- {v}[/red]")


def main():
    parser = argparse.ArgumentParser(description="Life in Pixels RAG — ask questions about your behavioral data")
    parser.add_argument("question", nargs="?", help="Natural language question about your data")
    parser.add_argument("--rebuild", action="store_true", help="Re-fetch data and rebuild the index")
    parser.add_argument("--source", choices=("live", "fixture"), default="live", help="live export or the committed fixture")
    parser.add_argument("--db", default="./chroma_db", help="index directory (default: ./chroma_db)")
    parser.add_argument("--days", type=int, default=180, help="Days of history to fetch (default: 180)")
    parser.add_argument("--top-k", type=int, default=8, help="Days to retrieve per query (default: 8)")
    parser.add_argument("--contract", choices=("native", "prompt"), default="native", help="structured output, or the prompt and the parser")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--v1", action="store_true", help="the v1 path: top-k and a prompt-only citation request")
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="Upsert the days logged since this date (idempotent)")
    parser.add_argument("--status", action="store_true", help="How current the index is against the source, no model calls")
    parser.add_argument("--dry-run", action="store_true", help="Show the first 3 chunks without calling any API")
    parser.add_argument("--stats", action="store_true", help="Show corpus stats without calling any API")
    args = parser.parse_args()
    db = Path(args.db)

    if args.dry_run or args.stats:
        rows = load_rows(args.source, args.days)
        if args.dry_run:
            show_dry_run(rows)
        if args.stats:
            show_stats(rows, args.days)
        return

    if args.status:
        show_status(db, args.source, args.days)
        return

    if args.since:
        ingest_since(db, args.source, args.since)
        if not args.question:
            return

    if args.rebuild or (args.question is None and not args.since):
        rebuild(db, args.source, args.days)
        if not args.question:
            console.print('\nIndex ready. Run: [bold]python main.py "your question here"[/bold]')
            return

    console.print(f"\n[bold]Searching for:[/bold] {args.question}\n")
    try:
        days, collection = load_corpus(db)
    except Exception:
        console.print(f"[red]No index at {db}. Run `python main.py --rebuild` first.[/red]")
        return

    if args.v1:
        from query import query
        result = query(args.question, collection, top_k=args.top_k)
        console.print(Panel(Markdown(result["answer"]), title="Answer (v1)", border_style="green"))
        if result["sources"]:
            console.print(f"\n[dim]Retrieved from: {', '.join(result['sources'])}[/dim]")
        return

    import anthropic
    from core.pipeline import ask
    r = ask(args.question, days=days, collection=collection, client=anthropic.Anthropic(), top_k=args.top_k, contract=args.contract)
    if args.json:
        print(json.dumps(r, indent=1, default=str))
        return
    render_result(r)


if __name__ == "__main__":
    main()
