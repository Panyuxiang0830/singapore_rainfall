"""Fetch metadata and bounded, non-random API previews. No Spark job required."""
import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
COLLECTION_URL = "https://api-production.data.gov.sg/v2/public/api/collections/2279/metadata"
DATASTORE_URL = "https://data.gov.sg/api/action/datastore_search"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_json(session, url, **kwargs):
    for attempt in range(3):
        response = session.get(url, timeout=(10, 45), **kwargs)
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("Request attempts exhausted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2017, 2025)))
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error("--limit must be between 1 and 1000")
    if not set(args.years) <= set(range(2017, 2025)):
        parser.error("Only project years 2017–2024 are supported")

    session = requests.Session()
    session.headers["User-Agent"] = "singapore-rainfall-course-project/0.1"
    collection = fetch_json(session, COLLECTION_URL)
    if collection.get("code") != 0:
        raise RuntimeError(f"Collection API error: {collection}")
    save_json(ROOT / "reports/collection_metadata.json", collection)
    summaries = []
    for dataset_id in collection["data"]["collectionMetadata"]["childDatasets"]:
        preview = fetch_json(session, DATASTORE_URL, params={"resource_id": dataset_id, "limit": args.limit})
        if not preview.get("success"):
            raise RuntimeError(f"Datastore API error for {dataset_id}: {preview}")
        result = preview["result"]
        records = result.get("records", [])
        if not records:
            raise RuntimeError(f"No preview records returned for {dataset_id}")
        # Derive the year from actual observation timestamps, not list position.
        first_time = str(records[0].get("timestamp", ""))
        if not re.match(r"^\d{4}-", first_time):
            raise RuntimeError(f"Unexpected timestamp field: {records[0]}")
        year = int(first_time[:4])
        if year in args.years:
            save_json(ROOT / f"data/samples/{year}.json", preview)
            timestamps = sorted({str(row.get("timestamp", "")) for row in records})
            values = [float(row["reading_value"]) for row in records if row.get("reading_value") not in (None, "")]
            summary = {
                "year": year,
                "dataset_id": dataset_id,
                "api_reported_rows": result.get("total"),
                "preview_rows": len(records),
                "fields": result.get("fields", []),
                "preview_time_min": timestamps[0],
                "preview_time_max": timestamps[-1],
                "preview_station_count": len({row.get("station_id") for row in records}),
                "reading_types": sorted({str(row.get("reading_type")) for row in records}),
                "reading_units": sorted({str(row.get("reading_unit")) for row in records}),
                "preview_negative_readings": sum(value < 0 for value in values),
                "warning": "Sequential API preview, not representative of the full year. No weather findings can be inferred.",
            }
            summaries.append(summary)
            print(f"{year}: API rows={summary['api_reported_rows']}, preview={len(records)}, time={timestamps[0]} .. {timestamps[-1]}", flush=True)
        time.sleep(0.5)
    missing_years = set(args.years) - {row["year"] for row in summaries}
    if missing_years:
        raise RuntimeError(f"Missing requested years: {sorted(missing_years)}")
    save_json(ROOT / "reports/inspection_summary.json", {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "preview_limit": args.limit,
        "warning": "Schema inspection only. Full-data cleaning and analysis have not run.",
        "datasets": sorted(summaries, key=lambda item: item["year"]),
    })


if __name__ == "__main__":
    main()
