"""Sequential, resumable Task 2 label preparation with the existing RSS guard."""
import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

from run_task1 import ROOT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2017, 2025)))
    parser.add_argument('--task1-check', action='store_true')
    args = parser.parse_args()
    if not set(args.years) <= set(range(2017, 2025)):
        parser.error('Only 2017--2024 are supported')
    work = ROOT / 'work/task2'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'labels.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another project label-preparation job is active')
    if hasattr(os, 'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))[:2]
        os.sched_setaffinity(0, cpus)
    env = os.environ.copy()
    env.update({
        'JAVA_HOME': str(ROOT / '.runtime/java'),
        'PYTHONPATH': str(ROOT / '.runtime/python'),
        'JAVA_TOOL_OPTIONS': '-XX:ActiveProcessorCount=2',
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
        'SPARK_LOCAL_IP': '127.0.0.1', 'PYSPARK_PYTHON': sys.executable,
        'TMPDIR': str(ROOT / 'work/tmp'),
    })
    env['PATH'] = env['JAVA_HOME'] + '/bin:' + env['PATH']
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    print('TASK2_RUN_START', ROOT, 'affinity', sorted(os.sched_getaffinity(0)), flush=True)
    if args.task1_check:
        run([sys.executable, '-u', 'scripts/task2_labels.py', '--task1-check'], env)
        return
    run([sys.executable, '-u', 'scripts/task2_labels.py', '--check-only'], env)
    for year in sorted(set(args.years)):
        run([sys.executable, '-u', 'scripts/task2_labels.py', '--year', str(year)], env)
    summaries = [json.loads((work / f'thresholds_{y}.json').read_text()) for y in sorted(set(args.years))]
    groups = []
    for split in ['train', 'validation', 'test']:
        rows = [r for r in summaries if r['split'] == split]
        if not rows:
            continue
        count = sum(r['valid_windows'] for r in rows)
        for q in [0, 5, 25, 50]:
            positive = sum(r[f'positive_gt_{q}mm'] for r in rows)
            groups.append({
                'split': split, 'years': [r['year'] for r in rows],
                'threshold_mm': q, 'valid_windows': count,
                'positive_windows': positive, 'positive_fraction': positive / count,
                'positive_days': sum(r[f'positive_days_gt_{q}mm'] for r in rows),
                'positive_station_days': sum(r[f'positive_station_days_gt_{q}mm'] for r in rows),
            })
    (work / 'threshold_summary.json').write_text(json.dumps(groups, indent=2) + '\n')
    print('TASK2_LABELS_COMPLETE', json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
