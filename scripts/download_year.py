"""Stream one official annual CSV into gzip, with bounded disk/network use."""
import argparse
import gzip
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int, choices=range(2017, 2025))
    parser.add_argument('--mbps', type=float, default=8)
    args = parser.parse_args()
    if args.mbps <= 0:
        parser.error('--mbps must be positive')
    summary = json.loads((ROOT / 'reports/inspection_summary.json').read_text())
    item = next(row for row in summary['datasets'] if row['year'] == args.year)
    dataset_id = item['dataset_id']
    dest = ROOT / f'data/raw/{args.year}.csv.gz'
    manifest = ROOT / f'reports/downloads/{args.year}.json'
    if dest.exists() and manifest.exists():
        print(f'{args.year}: compressed download already exists', flush=True)
        return
    if shutil.disk_usage(ROOT).free < 5.25 * 1024**3:
        raise RuntimeError('Download would compromise 5 GiB disk reserve')
    dest.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    base = f'https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}'
    # Space requests by >=13 seconds to respect the public 5/minute quota.
    response = session.get(base + '/initiate-download', timeout=(15, 90))
    response.raise_for_status()
    if response.json().get('code') != 0:
        raise RuntimeError(response.json())
    link = None
    for attempt in range(20):
        time.sleep(13)
        response = session.get(base + '/poll-download', timeout=(15, 90))
        if response.status_code == 429:
            continue
        response.raise_for_status()
        payload = response.json()
        link = payload.get('data', {}).get('url')
        if link:
            break
        print(f'{args.year}: download preparation {payload.get("data", {}).get("status")}', flush=True)
    if not link:
        raise RuntimeError('Download preparation did not return a URL')
    partial = dest.with_suffix('.gz.part')
    for attempt in range(1, 4):
        # Restart only this incomplete annual file; completed years stay intact.
        digest = hashlib.sha256()
        count = 0
        started = time.monotonic()
        try:
            with session.get(link, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with gzip.open(partial, 'wb', compresslevel=1) as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
                        delay = count / (args.mbps * 1024**2) - (time.monotonic() - started)
                        if delay > 0:
                            time.sleep(delay)
                        if count % (64 * 1024**2) < len(chunk):
                            free = shutil.disk_usage(ROOT).free
                            if free < 5 * 1024**3:
                                raise RuntimeError('Stopping download to preserve disk reserve')
                            print(f'{args.year}: streamed {count / 1024**2:.0f} MiB; free {free / 1024**3:.2f} GiB', flush=True)
            break
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as error:
            print(f'{args.year}: network interruption ({type(error).__name__}), attempt {attempt}/3', flush=True)
            if attempt == 3:
                raise
            time.sleep(13 * attempt)
    partial.rename(dest)
    manifest.write_text(json.dumps({
        'year': args.year, 'dataset_id': dataset_id,
        'dataset_page': f'https://data.gov.sg/datasets/{dataset_id}/view',
        'download_api': base, 'api_reported_rows': item['api_reported_rows'],
        'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
        'uncompressed_bytes': count, 'compressed_bytes': dest.stat().st_size,
        'sha256_uncompressed': digest.hexdigest(),
        'compression': 'gzip level 1; original CSV bytes preserved',
    }, indent=2) + '\n')
    print(f'{args.year}: finished {count / 1024**2:.1f} MiB -> {dest.stat().st_size / 1024**2:.1f} MiB gzip', flush=True)


if __name__ == '__main__':
    main()
