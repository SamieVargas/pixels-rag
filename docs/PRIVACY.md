# What leaves the machine

This tool reads a personal daily log. Here is everything that leaves the
computer it runs on, and everything that does not.

**The data stays local.** The export is written to `chroma_db/days.json`
and embedded into a ChromaDB store in the same folder, with the default
embedding model running on the machine (Chroma's ONNX build of
all-MiniLM-L6-v2, downloaded once). Filters and aggregates are computed in
code over that file. Nothing about the index is sent anywhere.

**A question sends only its evidence to the model provider.** For a semantic
or filter question, that is the top-k retrieved day chunks (eight by default,
thirty at most on the filter route) plus the question itself. For an
aggregate question, it is the computed table and the question; no day text
is sent. An unanswerable question sends the question to the router and
nothing after that. The router call sends the question and, in a
conversation, the last three turns. The provider is Anthropic, under the key
in `.env`.

**Logs carry no content.** The eval results and the CLI print dates, scores,
the route, the plan, validation outcomes and token counts. They never
include the day text or the answer beyond what you asked to see on screen.
The committed results in `evals/results/` are computed on the synthetic
fixture, not on anyone's export.

**The embedding API arm is off by default and says what it sends.** The
embedding ablation has one arm (`openai`) that would send every chunk's text
to a third party to be embedded. It runs only when asked for by name with a
key set, and the table it writes carries that sentence. The default stays
local whatever that arm scores.

**Reranking stays local.** The cross-encoder runs on the machine; the lexical
baseline is a few lines of Python.

**The MCP server exposes two read-only tools.** `ask_pixels` runs the same
router, validator and model call as the CLI, with the same evidence going to
the same provider; `list_days` sends nothing anywhere. Neither tool writes.
The server runs as a local process over stdio; it does not listen on a port.

**Auditing an answer.** `python main.py --explain "..."` prints, for each
retrieved chunk, its similarity score, which filters matched, whether it was
inside the date range, and whether the answer cited it, plus the route, the
resolved plan and the validator's outcome.
