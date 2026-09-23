"""Part 12: the MCP server exposes two read-only tools and nothing else.

    python tests/test_mcp.py
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from core import index as I  # noqa: E402
from core.days import load_days  # noqa: E402
from stubs import FakeClient, decision, answer  # noqa: E402
import mcp_server  # noqa: E402

FAILS = []


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        FAILS.append(name)


def payload(res):
    """The tool's return value, from structured content or the text block."""
    if res.structured_content:
        return res.structured_content
    return json.loads(res.content[0].text)


def main():
    days = load_days(ROOT / "fixtures" / "days.json")
    coll, _ = I.build(I.make_client(), days, name="mcp_test", embedding_function=I.HashEmbedding())
    fc = FakeClient([decision("unanswerable"), decision("aggregate", metric="rating", stat="mean", group_by="weekend")])
    server = mcp_server.build_server({"days": days, "collection": coll, "client": fc})

    print("tools")
    tools = asyncio.run(server.list_tools())
    check("exactly ask_pixels and list_days", sorted(t.name for t in tools) == ["ask_pixels", "list_days"])
    check("both carry descriptions", all(t.description for t in tools))

    print("list_days")
    out = payload(asyncio.run(server.call_tool("list_days", {"start": "2026-06-08", "end": "2026-06-14"})))
    rows = out["days"]
    check("the week's seven days come back with the listed fields", len(rows) == 7 and out["count"] == 7 and rows[0]["date"] == "2026-06-08" and "sleep_score" in rows[0] and "text" not in rows[0])
    check("no model call was made", len(fc.requests) == 0)
    big = payload(asyncio.run(server.call_tool("list_days", {"start": "2026-01-01", "end": "2026-12-31"})))
    check("the list is capped at 100 and says so", len(big["days"]) == 100 and big["count"] == 118 and big["capped"] is True)

    print("ask_pixels")
    out = payload(asyncio.run(server.call_tool("ask_pixels", {"question": "What was my blood pressure?"})))
    check("an unanswerable question comes back as an admission with the route", out["route"] == "unanswerable" and out["unanswerable"] is True and out["validated"] is True)
    fc.replies.append(answer("Weekends 3.4, weekdays 3.0.", [], claims=[]))
    out = payload(asyncio.run(server.call_tool("ask_pixels", {"question": "average rating on weekends versus weekdays?"})))
    check("an aggregate question narrates the table through the same validator", out["route"] == "aggregate" and out["validated"] and "3.4" in out["answer"])

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
