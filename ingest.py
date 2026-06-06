"""Fetch sanitized behavioral data from the Life in Pixels Apps Script endpoint."""

import os

import requests
from dotenv import load_dotenv

load_dotenv()


def fetch_pixels_data(days: int = 180) -> list[dict]:
    """
    Fetch sanitized rows from the Life in Pixels Apps Script endpoint.

    Returns a list of row dicts. Raises on auth failure or bad response.
    """
    url = os.getenv("PIXELS_URL")
    token = os.getenv("PIXELS_TOKEN")

    if not url or not token:
        raise RuntimeError(
            "Missing PIXELS_URL or PIXELS_TOKEN. Copy .env.example to .env and fill them in."
        )

    resp = requests.get(
        url,
        params={
            "action": "rag_export",
            "days": days,
            "token": token,
        },
        timeout=60,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Endpoint returned {resp.status_code}")

    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")

    rows = data.get("rows", [])

    # Filter out rows with no date or completely empty
    rows = [r for r in rows if r.get("date") and r.get("ratingNum", 0) > 0]

    print(f"Fetched {len(rows)} logged days (of {data.get('totalRows')} total)")
    return rows
