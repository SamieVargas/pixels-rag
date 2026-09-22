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

## v2: retrieval you can measure, and a router that knows when retrieval is the wrong tool

Everything above still holds and still runs (`--v1`). What v1 could not say
is whether retrieval is any good, whether the citations are real, or what
happens to "what was my average sleep on hot yoga days", which no amount of
similarity search answers correctly. v2 adds those three things.

### The router

```
question
  └─ router · one Haiku call under a JSON Schema · kind, fields, operators, the date words
       ├─ semantic      dense search over day chunks, restricted to the days the filters and dates allow
       ├─ filter        the matching days, decided in code; the model summarizes them
       ├─ aggregate     pandas computes the statistic; the model narrates the table and nothing else
       └─ unanswerable  an admission, no model call
```

The model does the part that needs language and code does the part that
needs to be right. Relative dates never reach the model as a problem to
solve: the router copies the date words from the question ("the first two
weeks of June", "last two weeks") and `core/dates.py` resolves them against
the newest logged day, so a question means the same thing on every run.
Filters run in `core/filters.py` over the day records, and the dense search
is told which ids it may return. An aggregate question never touches the
index. Every query logs its route.

### The answer contract

```jsonc
{ "answer": "string",
  "claims": [ { "text": "string", "dates": ["YYYY-MM-DD"] } ],
  "cited_dates": ["YYYY-MM-DD"],
  "unanswerable": false,
  "why_unanswerable": "string or null" }
```

The prompt asks for citations; `core/validate.py` enforces them. A cited date
has to be one the pipeline retrieved. A claim has to name at least one date
unless the answer is an admission. Every number in the answer has to appear
in a cited day's text, in the question, or in the computed table, because
numbers are what the model is most tempted to invent about a time series.
A failure goes back to the model once, with the violations as the next user
turn, and the CLI prints the retrieved-but-uncited and cited-but-unretrieved
counts on every answer. Structured output (`output_config.format` with the
schema) is the default contract; `--contract prompt` runs the same shape
through the prompt and the fence-stripping parser, and the pipeline records
which path handled each reply.

### Evals

`evals/golden.jsonl` holds 28 questions across the four kinds, written before
any retrieval run: seven semantic, seven filter, seven aggregate, five
unanswerable, plus two follow-ups that wait for Part 10. They are written
against `fixtures/days.json`, a seeded synthetic export with the real
export's shape and a few planted days, since the real data stays off the
repo (see `docs/decisions.md`). `python evals/run.py --offline` scores
retrieval with the golden plans and no key; the keyed run lets the router
decide and scores the answers.

Offline, with the default local embedder (all-MiniLM-L6-v2), 2026-09-22:

| Measure | Result |
| --- | --- |
| Semantic recall@3 / @5 / @10 | 63% / 89% / 93% · MRR 0.71 |
| Filter: matched set equals the expected set | 100% of 7 |
| Route accuracy, facts in the answer, citation validity, abstention | pending a keyed run |

Three semantic questions carry the misses. S03 asks what recovery looked like
the day after the trail run: the run day is found and the morning after is
not, because "the day after" is adjacency, which similarity cannot express.
S04 finds the severe-anxiety day at rank five. S06 asks about a whole week
and five of its seven days make the top five, which is the case the chunking
ablation was built for.

Chunk granularity, offline, one run per arm (retrieval is deterministic
without a key, so one run is the whole story; the twenty-run arms measure
the answer's fact coverage and need a key):

| Measure | A · day chunks | B · day + week rollups |
| --- | --- | --- |
| Recall@5 | 89% | 93% |
| S06, the week question | 71% | 100% |
| Week chunks retrieved | 0 | 14 |

The rollup lifts the week question and changes nothing else. That is a
narrower claim than "chunk design is the lever", and it is the one the data
supports so far.

### Running v2

```bash
pip install -r requirements.txt
python main.py --source fixture --rebuild           # index the synthetic days
python main.py "Did I sleep better on hot yoga days?"
python main.py "Which days was my sleep score under 60?"
python main.py --rebuild                             # your export, via .env
python evals/run.py --offline                        # retrieval only, no key
python evals/run.py                                  # the router and the answers, needs ANTHROPIC_API_KEY
python evals/run.py --ablation                       # day vs day+week, 20 runs per arm
python tests/test_core.py                            # everything, no key, no model download
```

### An index that stays current

v1 was a one-shot export. `python main.py --since 2026-08-01` reads the days
logged since a date and upserts them by `id = date`, so running it twice
leaves the same index, and an edited day replaces its chunk in place. The
collection records the chunk template version, a hash of `chunk.py`, and the
embedding model that built it; when the code disagrees with any of the three,
the next ingest rebuilds the whole index and says why, rather than mixing two
chunk formats in one store. `python main.py --status` prints the newest
indexed day, the newest day at the source, how many logged days the index is
behind, and the version stamp. A living index is the difference between a
demo and a tool you open on a Tuesday.

### Data quality at the door

Every ingest validates the rows first and writes the report to
`evals/results/ingest-<date>.md`. A duplicate date, or a date that does not
parse, fails the ingest before anything is written. Missing days in the
range, values outside each field's range (a sleep score of 140, a resting
heart rate of 20), and days whose chunk holds nothing beyond the date and the
rating are warnings, counted in the report and listed by day. `--stats`
shows the same report without touching the index. On the fixture the report
is clean apart from its five deliberately unlogged days.

### Reranking, measured

`--rerank cross-encoder` retrieves the top 20 and lets a local cross-encoder
(MS MARCO MiniLM from sentence-transformers, nothing leaves the machine) keep
the top k. `--rerank lexical` is a token-overlap baseline that needs no model,
there so the wiring can be tested and run anywhere. Both are off by default,
and `python evals/run.py --offline --rerank-compare --rerank <name>` writes
the comparison against plain top-k on the semantic questions.

| Reranker | Recall@5 plain | Recall@5 reranked | Added ms per question |
| --- | --- | --- | --- |
| lexical (offline, 2026-09-22) | 89% | 89% | 0 |
| cross-encoder | pending: needs `pip install sentence-transformers` and the model download | | |

The lexical baseline changes nothing, which is a tie and is reported as one.
The three semantic misses are adjacency (S03) and a whole week (S06), which
reordering the candidates does not repair. The cross-encoder row is the one
that decides whether the flag earns a default; if its gain is under a few
points it stays off.

### What v2 does not do yet

Parts 8 to 12 of the plan: the embedding-model ablation, the headline finding reproduced deterministically, query
rewriting for follow-ups, a privacy page with `--explain`, and a local MCP
server. Each lands on its own PR with its own number.
