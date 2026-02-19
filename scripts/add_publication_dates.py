#!/usr/bin/env python3
"""
Extend text_summarization_goldstandard_data.json with publication dates
fetched from the CrossRef API using each entry's DOI.

Adds three fields to each entry:
  - publication_date:        best available date as "YYYY-MM-DD" (or "YYYY-MM" / "YYYY")
  - publication_date_source: which CrossRef field was used
  - publication_date_raw:    the raw date-parts array from CrossRef

The script is resumable: entries that already have a publication_date are skipped.
Progress is saved every --save-every entries (default 50).

Usage:
    python scripts/add_publication_dates.py [--input FILE] [--output FILE]
                                            [--concurrency N] [--save-every N]
                                            [--email EMAIL]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# CrossRef "polite pool" — providing an email moves you to a faster queue
DEFAULT_EMAIL = None
CROSSREF_API = "https://api.crossref.org/works/"

# Date fields in CrossRef metadata, ordered by preference
DATE_FIELDS = [
    "published-online",
    "published-print",
    "published",
    "issued",
    "created",
]


def date_parts_to_str(date_parts: list[list[int]]) -> str:
    """Convert CrossRef date-parts [[2025, 6, 12]] → '2025-06-12'."""
    parts = date_parts[0]  # first (usually only) element
    if len(parts) == 1:
        return f"{parts[0]}"
    if len(parts) == 2:
        return f"{parts[0]}-{parts[1]:02d}"
    return f"{parts[0]}-{parts[1]:02d}-{parts[2]:02d}"


def extract_doi(entry: dict) -> str:
    """Strip the https://doi.org/ prefix from the id field."""
    doi = entry.get("id", "")
    for prefix in ("https://doi.org/", "http://doi.org/"):
        if doi.startswith(prefix):
            return doi[len(prefix):]
    return doi


async def fetch_date(
    client: httpx.AsyncClient,
    doi: str,
    semaphore: asyncio.Semaphore,
    rate_delay: float = 0.05,
) -> dict:
    """
    Query CrossRef for a single DOI; return a dict with date info or error.
    """
    url = f"{CROSSREF_API}{doi}"
    async with semaphore:
        for attempt in range(4):
            try:
                resp = await client.get(url, timeout=30)
                if resp.status_code == 404:
                    return {"error": "DOI not found in CrossRef"}
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = 2 ** attempt
                    log.warning("HTTP %s for %s — retrying in %ss", resp.status_code, doi, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                msg = resp.json().get("message", {})
                for field in DATE_FIELDS:
                    val = msg.get(field)
                    if val and val.get("date-parts") and val["date-parts"][0]:
                        return {
                            "publication_date": date_parts_to_str(val["date-parts"]),
                            "publication_date_source": field,
                            "publication_date_raw": val["date-parts"],
                        }
                return {"error": "No date fields in CrossRef response"}
            except httpx.HTTPStatusError as exc:
                return {"error": f"HTTP {exc.response.status_code}"}
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                wait = 2 ** attempt
                log.warning("%s for %s — retrying in %ss", type(exc).__name__, doi, wait)
                await asyncio.sleep(wait)
        return {"error": "Max retries exceeded"}
    # small politeness delay between requests from same worker
    await asyncio.sleep(rate_delay)


async def process_batch(
    data: list[dict],
    indices: list[int],
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> int:
    """Fetch dates for a batch of entries. Returns count of successes."""
    tasks = {}
    for idx in indices:
        doi = extract_doi(data[idx])
        tasks[idx] = asyncio.create_task(fetch_date(client, doi, semaphore))

    successes = 0
    for idx, task in tasks.items():
        result = await task
        if "error" in result:
            log.warning("  [%d] %s — %s", idx, data[idx]["id"], result["error"])
        else:
            data[idx].update(result)
            successes += 1
    return successes


def save(data: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


async def main(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    todo = [i for i, d in enumerate(data) if "publication_date" not in d]
    already = total - len(todo)
    if already:
        log.info("Skipping %d entries that already have dates", already)
    log.info("Fetching dates for %d / %d entries (concurrency=%d)", len(todo), total, args.concurrency)

    if not todo:
        log.info("Nothing to do.")
        return

    headers = {"User-Agent": f"PublicationDateFetcher/1.0 (mailto:{args.email})" if args.email else "PublicationDateFetcher/1.0"}
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        done = 0
        ok = 0
        for batch_start in range(0, len(todo), args.save_every):
            batch = todo[batch_start : batch_start + args.save_every]
            t0 = time.monotonic()
            batch_ok = await process_batch(data, batch, client, semaphore)
            elapsed = time.monotonic() - t0
            done += len(batch)
            ok += batch_ok
            log.info(
                "Progress: %d/%d  (batch %d ok in %.1fs)  total ok: %d",
                done, len(todo), batch_ok, elapsed, ok,
            )
            save(data, output_path)
            log.info("Saved → %s", output_path)

    failed = len(todo) - ok
    log.info("Done. %d succeeded, %d failed out of %d.", ok, failed, len(todo))


def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input", "-i",
        default="Resources/text_summarization_goldstandard_data.json",
        help="Input JSON file (default: Resources/text_summarization_goldstandard_data.json)",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file (default: overwrite input in-place)",
    )
    p.add_argument(
        "--concurrency", "-c",
        type=int, default=10,
        help="Max concurrent requests (default: 10)",
    )
    p.add_argument(
        "--save-every", "-s",
        type=int, default=50,
        help="Save progress every N entries (default: 50)",
    )
    p.add_argument(
        "--email", "-e",
        default=DEFAULT_EMAIL,
        help="Email for CrossRef polite pool (faster rate limits)",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(cli()))
