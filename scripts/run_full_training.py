"""Sequential, resumable full-training ablation, benchmark and curve recovery.

All eligible 2017--2022 records fit preprocessing and every base classifier.
2023 H1 calibrates, 2023 H2 selects, and 2024 tests. No training inclusion
fraction exists. Run on the existing gpushare project; progress stays in work/.
"""
import argparse
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

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT/'work/task2/full_run'
BASE = ROOT/'work/task2/logistic_full'
BENCH = ROOT/'work/task2/model_comparison_full'
ALGORITHMS = ['logistic', 'svm', 'xgb']
POLICIES = ['natural', 'balanced', 'smote']
GROUPS = ['calendar_station', 'own_history', 'neighbors']


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(path.read_text())


def terminate(process):
    # Each phase and its Java/Python descendants share the new process group.
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def phase(name, command, env, memory_gib):
    started = time.monotonic()
    state = {'status': 'RUNNING', 'phase': name, 'started_epoch': time.time(),
             'full_training_rows': 34542285, 'cpu_threads': int(env['RAINFALL_CPU_THREADS']),
             'driver_heap_gib': int(env['RAINFALL_DRIVER_HEAP_GIB']),
             'phase_memory_limit_gib': memory_gib}
    save(PROGRESS/'status.json', state)
    print('FULL_PHASE_START', name, flush=True)
    log_path = PROGRESS/'logs'/f'{name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('ab') as out:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        state['phase_pid'] = process.pid
        save(PROGRESS/'status.json', state)
        peak = 0.
        try:
            while process.poll() is None:
                try:
                    parent = psutil.Process(process.pid)
                    members = [parent]+parent.children(recursive=True)
                except psutil.NoSuchProcess:
                    members = []
                rss = 0
                for member in members:
                    try:
                        rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                rss_gib = rss/1024**3
                peak = max(peak, rss_gib)
                free_gib = shutil.disk_usage(ROOT).free/1024**3
                with (PROGRESS/'resources.jsonl').open('a') as log:
                    log.write(json.dumps({'phase': name, 'rss_gib': rss_gib,
                                          'free_disk_gib': free_gib,
                                          'elapsed_s': time.monotonic()-started})+'\n')
                if rss_gib > memory_gib or free_gib < 5:
                    terminate(process)
                    raise RuntimeError(f'{name}: resource guard stopped phase; RSS={rss_gib:.2f} GiB, free disk={free_gib:.2f} GiB')
                time.sleep(5)
            if process.returncode:
                raise RuntimeError(f'{name}: phase failed, code {process.returncode}; inspect {log_path}')
        except BaseException as error:
            if process.poll() is None:
                terminate(process)
            state.update(status='FAILED', elapsed_s=time.monotonic()-started,
                         peak_rss_gib=peak, failure_error=str(error))
            save(PROGRESS/'status.json', state)
            raise
    state.update(status='PHASE_COMPLETE', elapsed_s=time.monotonic()-started,
                 peak_rss_gib=peak)
    save(PROGRESS/'status.json', state)
    print('FULL_PHASE_COMPLETE', name, state['elapsed_s'], 'seconds', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-gib', type=float, default=12.)
    parser.add_argument('--xgb-memory-gib', type=float, default=24.)
    parser.add_argument('--cpu-threads', type=int, default=20)
    parser.add_argument('--driver-heap-gib', type=int, default=6)
    parser.add_argument('--prepare-benchmark-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.cpu_threads <= 20:
        parser.error('--cpu-threads must be between 1 and 20')
    if args.driver_heap_gib < 1 or args.driver_heap_gib + 2 > args.memory_gib:
        parser.error('Driver heap must leave at least 2 GiB within the process-tree bound')
    PROGRESS.mkdir(parents=True, exist_ok=True)
    lock = (PROGRESS/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(PROGRESS/'runner.json', {'pid': os.getpid(), 'started_epoch': time.time()})
    save(PROGRESS/'resource_limits.json', {'cpu_threads': args.cpu_threads, 'driver_heap_gib': args.driver_heap_gib,
         'spark_master': f'local[{args.cpu_threads}]',
         'shuffle_partitions': max(8, args.cpu_threads * 4),
         'default_parallelism': args.cpu_threads,
         'xgb_workers': min(2, args.cpu_threads),
         'xgb_threads_per_worker': max(1, args.cpu_threads // min(2, args.cpu_threads)),
         'spark_phase_rss_limit_gib': args.memory_gib,
         'xgb_phase_rss_limit_gib': args.xgb_memory_gib,
         'free_disk_reserve_gib': 5, 'gpu': False, 'nice': 19, 'ionice_class': 3})
    if hasattr(os, 'sched_getaffinity'):
        available_cpus = sorted(os.sched_getaffinity(0))
        if len(available_cpus) < args.cpu_threads:
            raise RuntimeError('Insufficient CPUs in the allowed affinity set')
        os.sched_setaffinity(0, available_cpus[:args.cpu_threads])
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    if shutil.which('ionice'):
        subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], check=True)
    env = os.environ.copy()
    env.update({'JAVA_HOME': str(ROOT/'.runtime/java'), 'PYTHONPATH': str(ROOT/'.runtime/python'),
                'JAVA_TOOL_OPTIONS': f'-XX:ActiveProcessorCount={args.cpu_threads}', 'OPENBLAS_NUM_THREADS': '1',
                'RAINFALL_CPU_THREADS': str(args.cpu_threads), 'RAINFALL_TASK_CPUS': '1',
                'RAINFALL_RSS_LIMIT_GIB': str(args.memory_gib),
                'RAINFALL_DRIVER_HEAP_GIB': str(args.driver_heap_gib),
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '',
                'SPARK_LOCAL_IP': '127.0.0.1', 'PYSPARK_PYTHON': sys.executable,
                'TMPDIR': str(ROOT/'work/tmp')})
    env['PATH'] = env['JAVA_HOME']+'/bin:'+env['PATH']
    (ROOT/'work/tmp').mkdir(parents=True, exist_ok=True)
    def run(name, script, *arguments, xgb=False):
        from supervise_full_training import available_gib
        required_memory = args.xgb_memory_gib if xgb else args.memory_gib
        if available_gib() < required_memory+8:
            reason = 'Insufficient free memory for the bounded full-data phase'
            save(PROGRESS/'status.json', {'status': 'FAILED', 'phase': name,
                 'failure_error': reason, 'full_training_rows': 34542285})
            raise RuntimeError(reason)
        phase_env = env.copy()
        if xgb:
            phase_env['RAINFALL_TASK_CPUS'] = str(max(1, args.cpu_threads // min(2, args.cpu_threads)))
            phase_env['RAINFALL_RSS_LIMIT_GIB'] = str(args.xgb_memory_gib)
        phase(name, [sys.executable, '-u', 'scripts/'+script, *arguments], phase_env,
              required_memory)
    # A separately launched initial preparation may still be finishing.
    initial_pid = BASE/'prepare.pid'
    if initial_pid.exists():
        pid = int(initial_pid.read_text())
        while True:
            try:
                parent = psutil.Process(pid)
                command = parent.cmdline()
                preparing = (any(x.endswith('run_task2_logistic.py') for x in command) and '--prepare-only' in command
                             or any(x.endswith('task2_logistic.py') for x in command) and '--prepare' in command)
                if parent.status() == psutil.STATUS_ZOMBIE or Path(parent.cwd()).resolve() != ROOT or not preparing:
                    break
            except psutil.NoSuchProcess:
                break
            print('WAITING_FOR_INITIAL_FULL_PREPARATION', pid, flush=True)
            time.sleep(10)
    run('full_preprocessing', 'task2_logistic.py', '--prepare')
    if args.prepare_benchmark_only:
        run('full_benchmark_preparation', 'task2_models.py', '--phase', 'prepare')
        return
    run('smote_fixtures', 'task2_smote.py', '--check-only')
    for group in GROUPS:
        run('ablation_'+group, 'task2_logistic.py', '--group', group)
    baseline = {'status': 'COMPLETE', 'groups': GROUPS,
                'preparation': read(BASE/'preparation.json'),
                'results': {g: read(BASE/f'{g}.json') for g in GROUPS}}
    baseline['selected_group_by_validation_pr_auc'] = {
        str(q): max(GROUPS, key=lambda g: baseline['results'][g]['models'][str(q)]['validation']['pr_auc'])
        for q in [0, 5, 25]}
    save(BASE/'summary.json', baseline)
    save(ROOT/'models/logistic_baseline_full/manifest.json', {
        'preparation': baseline['preparation'],
        'selected_group_by_validation_pr_auc': baseline['selected_group_by_validation_pr_auc'],
        'thresholds_mm': [0, 5, 25], 'probability_policy': 'Independent natural-distribution classifiers'})
    run('estimator_fixtures', 'task2_models.py', '--phase', 'check')
    run('full_benchmark_preparation', 'task2_models.py', '--phase', 'prepare')
    for q in [0, 5, 25]:
        run(f'smote_gt_{q}mm', 'task2_smote.py', '--threshold', str(q))
    run('smote_vectors', 'task2_models.py', '--phase', 'smote-vectors')
    for algorithm in ALGORITHMS:
        for policy in POLICIES:
            run(algorithm+'_'+policy, 'task2_models.py', '--phase', 'fit',
                '--algorithm', algorithm, '--policy', policy, xgb=algorithm == 'xgb')
    run('validation_selection', 'task2_models.py', '--phase', 'select')
    for algorithm in ALGORITHMS:
        run(algorithm+'_test', 'task2_models.py', '--phase', 'test', '--algorithm', algorithm, xgb=algorithm == 'xgb')
    run('full_train_diagnostics', 'task2_models.py', '--phase', 'train-diagnostics', xgb=True)
    run('model_artifact_checks', 'task2_models.py', '--phase', 'artifacts', xgb=True)
    run('final_curves', 'task2_curves.py', '--worker', xgb=True)
    summary = {'status': 'COMPLETE', 'preparation': read(BENCH/'preparation.json'),
               'selection': read(BENCH/'selection.json'),
               'smote': {str(q): read(BENCH/'smote'/f'gt_{q}mm'/'generation.json') for q in [0,5,25]},
               'candidates': {a: {p: read(BENCH/'results'/f'{a}_{p}.json') for p in POLICIES} for a in ALGORITHMS},
               'test': {a: read(BENCH/'test'/f'{a}.json') for a in ALGORITHMS},
               'train_final': {str(q): read(BENCH/'train_final'/f'gt_{q}mm.json') for q in [0,5,25]},
               'resource_configuration': read(PROGRESS/'resource_limits.json'),
               'artifact_checks': read(BENCH/'artifact_checks.json')}
    save(BENCH/'summary.json', summary)
    save(PROGRESS/'status.json', {'status': 'COMPLETE', 'full_training_rows': 34542285,
                                'completed_epoch': time.time()})
    print('ALL_FULL_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
