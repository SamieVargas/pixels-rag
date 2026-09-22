# Decisions

Where v2 departs from the spec, or from what a reader might expect, and why.
The repo wins where the two disagree; this is the record.

## The golden set is written against a committed synthetic fixture

The spec asks for golden questions before any retrieval run, labelled with
the dates a correct answer draws on. The real export is personal data and
does not belong in the repo, and a golden set nobody else can run is not an
eval. `fixtures/make_days.py` generates 118 days with the export's shape,
seeded so the file is byte-identical on every run, with a handful of planted
days (a migraine, one five-star day in July, a trail run and the morning
after, a night of alcohol and late meals, a severe-anxiety day, two
Hashimoto's flares) and one planted pattern (hot yoga plus walking sleeps
better). The questions were written first; `evals/make_golden.py` computes
the filter and aggregate labels with plain Python that does not import the
pipeline, and a test checks the pipeline's filters and aggregator against
those labels. Running the evals on the real export means writing a golden set
against it; the runner takes any export file with `--days`.

## Relative dates: the model copies the words, code resolves them

The spec's router contract carries `date_range` as absolute dates and says
relative dates are resolved in code, never by the model. The two cannot both
hold unless the contract also carries the words. It does: `date_phrase` is
the date text copied from the question, and `date_range` is filled only when
the question already contains explicit dates. `core/dates.py` resolves the
phrase against the newest logged day, so "the last two weeks" means the same
thing on every run and in every test.

## Filters run in code, and the index is told which ids are allowed

ChromaDB's `where` covers comparisons and equality, not "contains" on a
comma-separated exercise field, and its semantics differ between versions.
The corpus is a few hundred rows, so `core/filters.py` decides which days
match, and the dense search is restricted to those ids with the query's
`ids` parameter. One mechanism, owned by testable code; the vector store
only ranks.

## Every number in the answer has to appear in a cited chunk

The validator is strict on purpose: a number that does not appear in a cited
day's text, in the question, or in the computed table is a violation, and
the model gets one retry with the violations named. That rejects an honest
"3 days" count the model derived itself. On a time series, invented numbers
are the failure that matters most, and a count the model wants to state
belongs on the aggregate route where code computes it.

## `group_by: "match"` on the aggregate contract

"Did I sleep better on hot yoga days" is a comparison between the days that
satisfy a filter and the days that do not. Rather than ask the model to
express that as two queries, the contract lets `group_by` be `match`, and
the aggregator splits the corpus by whether the filters hold. A field name in
`group_by` splits by that field instead.

## Weekly rollups count as covering their days

In the chunking ablation, arm B indexes one rollup chunk per week beside the
day chunks. When recall is scored, a retrieved week chunk covers its seven
days, since that is the point of it: "what was the week of June 8 like" can
be answered from one chunk instead of seven. The table also reports how many
week chunks were retrieved, so a recall gain can be traced to them.

## Haiku, as in v1

v1 pinned `claude-haiku-4-5-20251001` and the router and the answer call keep
it. The router's job is a small classification with a schema; the answer
narrates a handful of chunks or one table. The evals measure that choice
rather than assume it, and the model id is one constant in
`core/contracts.py`.

## `main.py` keeps the v1 path

`--v1` runs the original top-k call with prompt-only citations, so the two
can be compared on the same index and the README's v1 description stays true.
