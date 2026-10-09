"""Sequential, resumable pipeline; all project children inherit CPU affinity."""
import json
import os
import shutil
import subprocess
import sys
import time
import signal
import platform
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]


def run(args, env):
    print('RUN', ' '.join(args), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(args, env=env, cwd=ROOT, start_new_session=True)
    log = ROOT / 'reports/resource_usage.jsonl'
    while process.poll() is None:
        try:
            parent = psutil.Process(process.pid)
            members = [parent] + parent.children(recursive=True)
        except psutil.NoSuchProcess:
            members = []
        rss = 0
        for member in members:
            try:
                rss += member.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        free = shutil.disk_usage(ROOT).free
        with log.open('a') as out:
            out.write(json.dumps({'phase': args[-2:], 'elapsed_s': round(time.monotonic() - started, 1),
                                  'rss_gib': round(rss / 1024**3, 3), 'free_disk_gib': round(free / 1024**3, 3),
                                  'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                                  'platform': platform.system(), 'spark_worker_threads': 2}) + '\n')
        if free < 5 * 1024**3 or rss > 4 * 1024**3:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
            raise RuntimeError('Stopped project subprocess: 5 GiB disk reserve or 4 GiB process-group RSS bound reached')
        time.sleep(5)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args)


def main():
    env = os.environ.copy()
    env['JAVA_HOME'] = str(ROOT / '.runtime/java')
    env['PYTHONPATH'] = str(ROOT / '.runtime/python')
    if (ROOT / '.runtime/spark/deps/bin/spark-submit').exists():
        env['SPARK_HOME'] = str(ROOT / '.runtime/spark/deps')
        env['PYTHONPATH'] = os.pathsep.join([str(ROOT / '.runtime/python'), str(ROOT / '.runtime/spark'),
                                           str(ROOT / '.runtime/spark/lib/py4j-0.10.9.7-src.zip')])
    env['PATH'] = env['JAVA_HOME'] + '/bin:' + env['PATH']
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'
    env['JAVA_TOOL_OPTIONS'] = '-XX:ActiveProcessorCount=2'
    env['SPARK_LOCAL_IP'] = '127.0.0.1'
    env['PYSPARK_PYTHON'] = sys.executable
    (ROOT / 'work/tmp').mkdir(parents=True, exist_ok=True)
    env['TMPDIR'] = str(ROOT / 'work/tmp')
    begin = time.monotonic()
    run([sys.executable, '-u', 'scripts/check_windows.py'], env)
    for year in range(2017, 2025):
        if shutil.disk_usage(ROOT).free < 5 * 1024**3:
            raise RuntimeError('5 GiB disk reserve reached')
        target = ROOT / f'data/annual/{year}/_SUCCESS'
        audit = ROOT / f'reports/quality/{year}.json'
        if not (target.exists() and audit.exists()):
            run([sys.executable, '-u', 'scripts/download_year.py', str(year)], env)
            run([sys.executable, '-u', 'scripts/task1_spark.py', '--clean-year', str(year)], env)
        # Raw bytes can be re-downloaded and are documented with SHA256. Only
        # remove this project's completed annual source after audit succeeds.
        raw = ROOT / f'data/raw/{year}.csv.gz'
        if target.exists() and audit.exists() and raw.exists():
            raw.unlink()
        print(f'YEAR {year} DONE; free GiB={shutil.disk_usage(ROOT).free / 1024**3:.2f}', flush=True)
    run([sys.executable, '-u', 'scripts/task1_spark.py', '--aggregate'], env)
    print(f'TASK1 PIPELINE COMPLETE after {time.monotonic() - begin:.1f}s', flush=True)


if __name__ == '__main__':
    main()
