# Life in Pixels — RAG

I've been logging 45 behavioral metrics daily since late 2025. This is a RAG
system that lets me ask natural language questions about the patterns in that
data — questions a spreadsheet can't answer.

A spreadsheet can filter one thing. This can reason about the intersection of
many: *"On days where regulation was Struggling or Dysregulated, what else was
true?"* or *"What does my body battery recovery look like after high-exercise
periods?"* Those require retrieving the right subset of days semantically, then
reasoning across multiple fields at once — which is exactly what RAG does.

## How it works

```
Google Apps Script (sanitized JSON endpoint)
    ↓ ingest.py    fetch via requests + token auth, parse, validate
    ↓ chunk.py     each day's row → information-dense text chunk
    ↓ embed.py     embed chunks locally, store in ChromaDB (on disk)
    ↓ query.py     question → retrieve top-k chunks → Claude Haiku synthesizes
    ↓ main.py      clean CLI output: answer + the dates it drew from
```

- **Embeddings** run locally via ChromaDB's default model
  (`sentence-transformers/all-MiniLM-L6-v2`) — no API cost, good enough for
  ~180 rows.
- **Generation** uses Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) — fast and
  cheap for this use case.
- **Privacy** — sensitive free-text fields (notes, therapist, trauma response,
  etc.) are stripped server-side by the Apps Script *before* data ever leaves
  the source. This client never sees them.

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
python main.py "What were my energy patterns on high sleep score days?"

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
python main.py "On days where I was Struggling or Dysregulated, what else was happening?"
python main.py "What does my sleep look like when body battery is below 20?"
python main.py "When do unhelpful thinking patterns cluster — what other signals appear?"
python main.py "What's different about my 5-star days vs my 1-2 star days?"
python main.py "How does sunlight affect my mood and energy ratings?"
python main.py "On high-anxiety days, what regulation tools did I use?"
```

## Files

| File              | Role                                                        |
| ----------------- | ----------------------------------------------------------- |
| `ingest.py`       | Fetch from Apps Script, parse, validate                     |
| `chunk.py`        | Row dict → readable text chunk (drives retrieval quality)   |
| `embed.py`        | Embed chunks, store/load ChromaDB                           |
| `query.py`        | Retrieve top-k → generate cited answer with Claude          |
| `main.py`         | CLI entry point                                             |
| `chroma_db/`      | Local vector store — git-ignored, rebuilt on demand         |
