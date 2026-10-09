"""Sequential server feature jobs with inherited CPU and sampled RSS limits."""
import argparse
import fcntl
import json
import os
import sys

from run_task1 import ROOT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2017, 2025)))
    parser.add_argument('--months', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    if not set(args.years) <= set(range(2017, 2025)) or not set(args.months) <= set(range(1, 13)):
        parser.error('Only years 2017--2024 and months 1--12 are supported')
    work = ROOT / 'work/task2'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'features.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another project feature job is active')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    env = os.environ.copy()
    env.update({'JAVA_HOME': str(ROOT / '.runtime/java'),
                'PYTHONPATH': str(ROOT / '.runtime/python'),
                'JAVA_TOOL_OPTIONS': '-XX:ActiveProcessorCount=2',
                'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                'SPARK_LOCAL_IP': '127.0.0.1', 'PYSPARK_PYTHON': sys.executable,
                'TMPDIR': str(ROOT / 'work/tmp')})
    env['PATH'] = env['JAVA_HOME'] + '/bin:' + env['PATH']
    print('TASK2_FEATURE_RUN_START', 'thresholds_mm', [0, 5, 25], flush=True)
    base = [sys.executable, '-u', 'scripts/task2_features.py']
    if args.metadata_only:
        run(base + ['--metadata-only'], env)
        return
    run(base + ['--check-only'], env)
    if args.check_only:
        return
    run(base + ['--metadata-only'], env)
    for year in sorted(set(args.years)):
        run(base + ['--year', str(year), '--months'] + [str(m) for m in sorted(set(args.months))], env)
    months = [json.loads((work / f'features_{y}_{m:02d}.json').read_text())
              for y in sorted(set(args.years)) for m in sorted(set(args.months))]
    summary = {'status': 'COMPLETE', 'years': sorted(set(args.years)),
               'months': sorted(set(args.months)), 'rows': sum(r['rows'] for r in months),
               'numeric_features': months[0]['numeric_features'],
               'categorical_features': months[0]['categorical_features'],
               'asof_or_graph_violations': sum(r['asof_or_graph_violations'] for r in months),
               'recent_own_rows': sum(r['recent_own_rows'] for r in months),
               'with_neighbor_rows': sum(r['with_neighbor_rows'] for r in months),
               'feature_version': months[0]['feature_version'],
               'thresholds_mm': [0, 5, 25]}
    save = work / 'feature_summary.json'
    save.write_text(json.dumps(summary, indent=2) + '\n')
    print('TASK2_FEATURES_COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
