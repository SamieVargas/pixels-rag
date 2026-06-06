"""
Life in Pixels RAG — ask natural language questions about your behavioral data.

Usage:
  python main.py "What were my energy patterns on high sleep score days?"
  python main.py --rebuild
  python main.py --rebuild "What does recovery look like after trail running?"
  python main.py --dry-run         # show the first 3 chunks, no API calls
  python main.py --stats           # show corpus stats, no API calls
"""

import argparse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ingest import fetch_pixels_data
from chunk import row_to_chunk
from embed import build_index, load_index
from query import query

console = Console()


def rows_to_chunks(rows: list[dict]) -> list[dict]:
    """Turn raw row dicts into the {id, text, metadata} shape ChromaDB wants."""
    chunks = []
    for row in rows:
        chunks.append({
            "id": row["date"],
            "text": row_to_chunk(row),
            "metadata": {
                "date": row["date"],
                "rating": str(row.get("ratingNum", 0)),
                "mood": row.get("mood", ""),
                "regulation": row.get("regulationQuality", ""),
                "sleep_score": str(row.get("sleepScore", 0)),
                "body_battery": str(row.get("bodyBattery", 0)),
            },
        })
    return chunks


def show_dry_run(rows: list[dict], n: int = 3) -> None:
    """Print the first n chunks so you can eyeball chunking before burning credits."""
    console.print(f"\n[bold]Dry run — showing first {n} of {len(rows)} chunks[/bold]\n")
    for row in rows[:n]:
        text = row_to_chunk(row)
        console.print(Panel(text, title=f"[cyan]{row['date']}[/cyan]", border_style="cyan"))


def show_stats(rows: list[dict], days: int) -> None:
    """Print corpus stats: date range, avg rating, missing days count."""
    ratings = [r.get("ratingNum", 0) for r in rows if r.get("ratingNum", 0) > 0]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    logged = len(rows)
    missing = max(days - logged, 0)

    table = Table(title="Life in Pixels — corpus stats", show_header=False)
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value")
    table.add_row("Days requested", str(days))
    table.add_row("Days logged", str(logged))
    table.add_row("Days missing", str(missing))
    if rows:
        table.add_row("Date range", f"{rows[-1]['date']} → {rows[0]['date']}")
    table.add_row("Average rating", f"{avg_rating:.2f} / 5")
    console.print(table)


def rebuild(days: int) -> list[dict]:
    """Fetch fresh data, build chunks, and index them."""
    console.print(f"[bold]Fetching {days} days of data...[/bold]")
    rows = fetch_pixels_data(days=days)

    console.print("[bold]Building chunks...[/bold]")
    chunks = rows_to_chunks(rows)

    build_index(chunks)
    console.print(f"[green]Index built: {len(chunks)} days indexed[/green]")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Life in Pixels RAG — ask questions about your behavioral data"
    )
    parser.add_argument("question", nargs="?", help="Natural language question about your data")
    parser.add_argument("--rebuild", action="store_true", help="Re-fetch data and rebuild the index")
    parser.add_argument("--days", type=int, default=180, help="Days of history to fetch (default: 180)")
    parser.add_argument("--top-k", type=int, default=8, help="Days to retrieve per query (default: 8)")
    parser.add_argument("--dry-run", action="store_true", help="Show the first 3 chunks without calling any API")
    parser.add_argument("--stats", action="store_true", help="Show corpus stats without calling any API")
    args = parser.parse_args()

    # Modes that fetch raw data but never touch the Anthropic API.
    if args.dry_run or args.stats:
        rows = fetch_pixels_data(days=args.days)
        if args.dry_run:
            show_dry_run(rows)
        if args.stats:
            show_stats(rows, args.days)
        return

    # Rebuild the index (also runs when no question is given so a bare invocation
    # leaves you with a fresh index ready to query).
    if args.rebuild or args.question is None:
        rebuild(args.days)
        if not args.question:
            console.print('\nIndex ready. Run: [bold]python main.py "your question here"[/bold]')
            return

    # Query
    if args.question:
        console.print(f"\n[bold]Searching for:[/bold] {args.question}\n")
        try:
            collection = load_index()
        except Exception:
            console.print("[red]No index found. Run `python main.py --rebuild` first.[/red]")
            return

        result = query(args.question, collection, top_k=args.top_k)
        console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="green"))
        if result["sources"]:
            console.print(f"\n[dim]Retrieved from: {', '.join(result['sources'])}[/dim]")


if __name__ == "__main__":
    main()
