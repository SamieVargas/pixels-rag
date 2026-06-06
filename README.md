# Behavioral Data RAG

A RAG pipeline for asking natural language questions over structured
daily behavioral and biometric data — questions a spreadsheet can't answer.

Built to reason across multiple dimensions simultaneously:
*"On days with low body battery, what else was true?"*
*"What exercise types correlate with better sleep scores?"*

A spreadsheet filters one column. This retrieves semantically and
reasons across the intersection of many fields at once.

## How it works

```
Data source (sanitized JSON endpoint)
    ↓ ingest.py    fetch via requests + token auth, parse, validate
    ↓ chunk.py     each day's row → information-dense text chunk
    ↓ embed.py     embed chunks locally, store in ChromaDB (on disk)
    ↓ query.py     question → retrieve top-k chunks → Claude Haiku synthesizes
    ↓ main.py      clean CLI output: answer + the dates it drew from
```

- **Embeddings** run locally via ChromaDB's default model
  (`sentence-transformers/all-MiniLM-L6-v2`) — no API cost, good enough for
  a corpus of a few hundred rows.
- **Generation** uses Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) — fast and
  cheap for this use case.
- **Privacy** — the data source sanitizes records server-side before anything
  leaves it, so free-text fields never reach this client. The pipeline only
  ever sees the structured, sanitized export.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in PIXELS_URL, PIXELS_TOKEN, ANTHROPIC_API_KEY
```

## Usage

```bash
# Fetch data and build the index (run this first, and whenever data changes)
python main.py --rebuild

# Ask a question
python main.py "What exercise types correlate with higher sleep scores?"

# Rebuild and ask in one shot
python main.py --rebuild "What does recovery look like after trail running?"
```

### Inspecting before you spend credits

```bash
python main.py --dry-run   # show the first 3 chunks — verify chunking, no API calls
python main.py --stats     # date range, avg rating, missing-days count — no API calls
```

### Flags

| Flag        | Description                                          |
| ----------- | ---------------------------------------------------- |
| `--rebuild` | Re-fetch data and rebuild the index                  |
| `--days N`  | Days of history to fetch (default: 180)              |
| `--top-k N` | Days to retrieve per query (default: 8)              |
| `--dry-run` | Show the first 3 chunks, no API calls                |
| `--stats`   | Show corpus stats, no API calls                      |

## Example questions

```bash
python main.py "What exercise types correlate with higher sleep scores?"
python main.py "On low energy days, what else was typically true?"
python main.py "What patterns appear on my highest-rated days?"
python main.py "How does sleep quality relate to next-day productivity?"
python main.py "What does my sleep look like when body battery is below 20?"
python main.py "What's different about my 5-star days vs my 1-2 star days?"
```

## Files

| File              | Role                                                        |
| ----------------- | ----------------------------------------------------------- |
| `ingest.py`       | Fetch from the data endpoint, parse, validate               |
| `chunk.py`        | Row dict → readable text chunk (drives retrieval quality)   |
| `embed.py`        | Embed chunks, store/load ChromaDB                           |
| `query.py`        | Retrieve top-k → generate cited answer with Claude          |
| `main.py`         | CLI entry point                                             |
| `chroma_db/`      | Local vector store — git-ignored, rebuilt on demand         |
