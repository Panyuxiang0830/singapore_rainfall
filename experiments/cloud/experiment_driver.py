"""Run one bounded rainfall experiment at a time on this VM."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work/cloud/experiments'
PHASES = [
    ('error_audit_25', 'experiment_error_audit.py'),
    ('xgb25_small_leaf', 'experiment_xgb25.py', 'small_leaf'),
    ('xgb25_longer_trees', 'experiment_xgb25.py', 'longer_trees'),
    ('xgb25_sampled_trees', 'experiment_xgb25.py', 'sampled_trees'),
]


def write_status(status):
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / 'driver_status.json'
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(status, indent=2) + '\n')
    pending.replace(path)


def memory_available_gib():
    return psutil.virtual_memory().available / 1024**3


def run_phase(name, script, *extra):
    # A terminated Spark JVM can take a few seconds to release its pages.
    for attempt in range(25):
        if memory_available_gib() >= 48 and shutil.disk_usage(ROOT).free / 1024**3 >= 5:
            break
        if attempt == 24:
            raise RuntimeError('Preflight resource guard: require >=48 GiB available RAM and 5 GiB disk')
        time.sleep(5)
    env = os.environ.copy()
    env.update({
        'JAVA_HOME': str(ROOT / '.runtime/java'),
        'JAVA_TOOL_OPTIONS': '-XX:ActiveProcessorCount=20',
        'PYSPARK_PYTHON': sys.executable,
        'RAINFALL_CPU_THREADS': '20',
        'RAINFALL_TASK_CPUS': '1' if name == 'error_audit_25' else '10',
        'RAINFALL_DRIVER_HEAP_GIB': '6',
        'RAINFALL_RSS_LIMIT_GIB': '40',
        'RAINFALL_PYTHON_WORKER_REUSE': 'false',
        'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'CUDA_VISIBLE_DEVICES': '',
        'SPARK_LOCAL_IP': '127.0.0.1',
        'TMPDIR': str(ROOT / 'work/tmp'),
    })
    env['PATH'] = env['JAVA_HOME'] + '/bin:' + env['PATH']
    (ROOT / 'work/tmp').mkdir(parents=True, exist_ok=True)
    logfile = WORK / (name + '.log')
    started = time.monotonic()
    with logfile.open('ab') as out:
        process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve().parent / script), *extra],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
            start_new_session=True)
        while process.poll() is None:
            try:
                root = psutil.Process(process.pid)
                members = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                members = []
            rss = 0
            for member in members:
                try:
                    rss += member.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            rss_gib = rss / 1024**3
            free_disk_gib = shutil.disk_usage(ROOT).free / 1024**3
            state = {'status': 'RUNNING', 'phase': name, 'pid': process.pid,
                     'elapsed_s': round(time.monotonic()-started, 1),
                     'rss_gib': round(rss_gib, 2), 'free_disk_gib': round(free_disk_gib, 2)}
            write_status(state)
            if rss_gib > 40 or free_disk_gib < 5:
                os.killpg(process.pid, signal.SIGTERM)
                raise RuntimeError(f'Resource guard stopped {name}: {state}')
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f'{name} failed ({process.returncode}); inspect {logfile}')
    print('PHASE_COMPLETE', name, round(time.monotonic()-started, 1), flush=True)


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:20])
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    if shutil.which('ionice'):
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    started = time.monotonic()
    try:
        for phase in PHASES:
            run_phase(*phase)
        write_status({'status': 'COMPLETE', 'phases': [p[0] for p in PHASES],
                      'elapsed_s': round(time.monotonic()-started, 1)})
    except BaseException as error:
        write_status({'status': 'FAILED', 'error': str(error),
                      'elapsed_s': round(time.monotonic()-started, 1)})
        raise


if __name__ == '__main__':
    main()
