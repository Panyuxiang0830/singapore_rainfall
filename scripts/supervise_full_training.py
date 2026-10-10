"""Observe an existing project run and permit one bounded XGBoost memory retry.

The Spark XGBoost adapter retains both workers' dense batches in memory before
constructing quantized matrices. Retry only a measured XGBoost resource-guard
failure, without changing rows, predictors, labels or model parameters.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'work/task2/full_run'


def wait(pid):
    while True:
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return
            args = p.cmdline()
            if not any(x.endswith('run_full_training.py') for x in args):
                return
            assert Path(p.cwd()).resolve() == ROOT
        except psutil.NoSuchProcess:
            return
        time.sleep(10)


def available_gib():
    available = psutil.virtual_memory().available
    for limit, usage in [('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes'),
                         ('memory.max', 'memory.current')]:
        try:
            total = int((Path('/sys/fs/cgroup')/limit).read_text())
            used = int((Path('/sys/fs/cgroup')/usage).read_text())
            available = min(available, max(0, total-used))
        except (FileNotFoundError, ValueError):
            pass
    return available/1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner-pid', type=int, required=True)
    parser.add_argument('--xgb-memory-gib', type=float, default=40.)
    parser.add_argument('--cpu-threads', type=int, default=20)
    args = parser.parse_args()
    assert 4 < args.xgb_memory_gib <= 40
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK/'supervision.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpu_threads])
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    print('OBSERVING_EXISTING_FULL_RUN', args.runner_pid, flush=True)
    wait(args.runner_pid)
    status = json.loads((WORK/'status.json').read_text())
    if status['status'] == 'COMPLETE':
        print('FULL_RUN_COMPLETE', flush=True)
        return
    text = (WORK/'run.log').read_text()[-12000:]
    if (status['status'] != 'FAILED' or not status['phase'].startswith('xgb_')
            or status.get('peak_rss_gib', 0) <= status.get('phase_memory_limit_gib', 4)
            or 'resource guard stopped phase; RSS=' not in text):
        raise RuntimeError('Run stopped for another reason; inspect its phase before resuming')
    if available_gib() < args.xgb_memory_gib+8:
        raise RuntimeError('Insufficient free memory for the bounded full-data retry')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT/'.runtime/python')
    with (WORK/'run.log').open('ab') as out:
        child = subprocess.Popen([sys.executable, '-u', 'scripts/run_full_training.py',
                                  '--cpu-threads', str(args.cpu_threads),
                                  '--xgb-memory-gib', str(args.xgb_memory_gib)],
                                 cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        print('FULL_DATA_XGB_RETRY', child.pid, args.xgb_memory_gib, flush=True)
        child.wait()
    status = json.loads((WORK/'status.json').read_text())
    if child.returncode or status['status'] != 'COMPLETE':
        raise RuntimeError('Bounded retry did not complete; no further automatic retry')
    print('FULL_RUN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
