"""Resumable bounded driver for the final distance-aware benchmark."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_driver import run_phase

WORK = ROOT / 'work/cloud/experiments/full_variants'
ALGORITHMS = ('logistic', 'svm', 'xgb')
POLICIES = ('natural', 'balanced', 'smote')


def write_status(state):
    WORK.mkdir(parents=True, exist_ok=True)
    pending = WORK / 'status.pending.json'
    pending.write_text(json.dumps(state, indent=2) + '\n')
    pending.replace(WORK / 'status.json')


def phases(variant):
    yield ('%s_prepare' % variant, '--variant', variant, '--phase', 'prepare')
    for threshold in (0, 5, 25):
        yield ('%s_smote_%s' % (variant, threshold), '--variant', variant, '--phase', 'smote', '--threshold', str(threshold))
    yield ('%s_smote_vectors' % variant, '--variant', variant, '--phase', 'smote-vectors')
    for algorithm in ALGORITHMS:
        for policy in POLICIES:
            yield ('%s_fit_%s_%s' % (variant, algorithm, policy), '--variant', variant, '--phase', 'fit', '--algorithm', algorithm, '--policy', policy)
    yield ('%s_select' % variant, '--variant', variant, '--phase', 'select')
    for algorithm in ALGORITHMS:
        yield ('%s_test_%s' % (variant, algorithm), '--variant', variant, '--phase', 'test', '--algorithm', algorithm)
    yield ('%s_train_diagnostics' % variant, '--variant', variant, '--phase', 'train-diagnostics')
    yield ('%s_curves' % variant, '--variant', variant, '--phase', 'curves')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', choices=('distance',), default=('distance',))
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'driver.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:20])
    start = time.monotonic()
    completed = []
    try:
        for variant in args.variants:
            for name, *extra in phases(variant):
                write_status({'status': 'RUNNING', 'variant': variant, 'phase': name,
                              'completed': completed, 'elapsed_s': round(time.monotonic()-start, 1)})
                run_phase(name, 'full_variant_models.py', *extra)
                completed.append(name)
        write_status({'status': 'COMPLETE', 'variants': list(args.variants), 'completed': completed,
                      'elapsed_s': round(time.monotonic()-start, 1)})
    except BaseException as error:
        write_status({'status': 'FAILED', 'completed': completed, 'error': str(error),
                      'elapsed_s': round(time.monotonic()-start, 1)})
        raise


if __name__ == '__main__':
    main()
