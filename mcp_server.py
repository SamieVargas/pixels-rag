"""A local MCP server over the same router and validator the CLI uses.

Two read-only tools, nothing else:

  ask_pixels(question)     route, retrieve or compute, answer under the contract
  list_days(start, end)    the logged days in a range, a few fields each

The data never leaves the machine except as the evidence for one question,
exactly as with the CLI (see docs/PRIVACY.md). Runs over stdio:

  PIXELS_DB=./chroma_db python mcp_server.py

Claude Desktop config:

  {"mcpServers": {"pixels": {"command": "python", "args": ["/path/to/pixels-rag/mcp_server.py"],
                             "env": {"PIXELS_DB": "/path/to/pixels-rag/chroma_db", "ANTHROPIC_API_KEY": "..."}}}}
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core import index as I  # noqa: E402
from core.days import normalize_rows  # noqa: E402
from core.pipeline import ask  # noqa: E402

LIST_CAP = 100
LIST_FIELDS = ("date", "rating", "mood", "energy", "sleep_score", "body_battery", "steps", "exercise")


def load_deps(db: str | None = None) -> dict:
    """The index, the day records and the model client, from PIXELS_DB."""
    db = Path(db or os.environ.get("PIXELS_DB", "./chroma_db"))
    rows = json.loads((db / "days.json").read_text(encoding="utf-8"))["rows"]
    import anthropic
    return {"days": normalize_rows(rows), "collection": I.load(I.make_client(str(db))), "client": anthropic.Anthropic()}


def build_server(deps=None):
    """Build the server. `deps` can be injected (the tests do); otherwise the
    index is loaded on the first call, so startup never touches the model."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("pixels-rag", instructions="Read-only questions over a personal daily log (mood, sleep, wearables, exercise, habits). "
                                                  "ask_pixels answers a question with cited dates; list_days lists the logged days in a range. Nothing writes.")
    state = {"deps": deps}

    def deps_():
        if state["deps"] is None:
            state["deps"] = load_deps()
        return state["deps"]

    @server.tool()
    def ask_pixels(question: str) -> dict:
        """Ask a question about the daily log. Returns the answer, the route taken
        (semantic, filter, aggregate or unanswerable), the dates it cites, and
        whether the answer passed the citation validator."""
        d = deps_()
        r = ask(question, days=d["days"], collection=d["collection"], client=d["client"])
        a = r["answer"]
        return {"answer": a["answer"], "route": r["route"], "cited_dates": a["cited_dates"], "claims": a["claims"],
                "unanswerable": a["unanswerable"], "why_unanswerable": a["why_unanswerable"],
                "validated": r["validation"]["ok"], "retrieved": [c["id"] for c in r["retrieved"]], "counts": r["counts"]}

    @server.tool()
    def list_days(start: str, end: str) -> dict:
        """List the logged days between two dates (YYYY-MM-DD, inclusive), with
        the rating, mood, energy, sleep score, body battery, steps and
        exercise for each. At most 100 days; `count` says how many matched."""
        d = deps_()
        rows = [{k: day.get(k) for k in LIST_FIELDS} for day in d["days"] if start <= day["date"] <= end]
        return {"days": rows[:LIST_CAP], "count": len(rows), "capped": len(rows) > LIST_CAP}

    return server


if __name__ == "__main__":
    build_server().run("stdio")
