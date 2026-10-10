"""Sequential, resumable 36-candidate Spark model/imbalance benchmark."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time

from run_task1 import ROOT, run

ALGORITHMS = ['logistic', 'svm', 'gbt', 'xgb']
POLICIES = ['natural', 'balanced', 'smote']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--algorithms', nargs='+', choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=POLICIES)
    parser.add_argument('--predict-station')
    parser.add_argument('--predict-time')
    parser.add_argument('--threshold', type=int, choices=[0, 5, 25])
    parser.add_argument('--predict-algorithm', choices=ALGORITHMS)
    args = parser.parse_args()
    work = ROOT/'work/task2/model_comparison_full'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'run.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another model comparison run is active')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    env = os.environ.copy()
    env.update({'JAVA_HOME': str(ROOT/'.runtime/java'), 'PYTHONPATH': str(ROOT/'.runtime/python'),
                'JAVA_TOOL_OPTIONS': '-XX:ActiveProcessorCount=2', 'OPENBLAS_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '',
                'SPARK_LOCAL_IP': '127.0.0.1', 'PYSPARK_PYTHON': sys.executable,
                'TMPDIR': str(ROOT/'work/tmp')})
    env['PATH'] = env['JAVA_HOME']+'/bin:'+env['PATH']
    if args.predict_station:
        if not args.predict_time or args.threshold is None:
            parser.error('Prediction requires --predict-time and --threshold')
        command = [sys.executable, '-u', 'scripts/predict_task2_model.py',
                   '--station', args.predict_station, '--time', args.predict_time,
                   '--threshold', str(args.threshold)]
        if args.predict_algorithm:
            command += ['--algorithm', args.predict_algorithm]
        run(command, env)
        return
    base = [sys.executable, '-u', 'scripts/task2_models.py']
    started = time.monotonic()
    # Recheck SMOTE only when its source changes; estimator fixtures have a separate marker.
    checks_path = work/'checks_passed.json'
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    smote_hash = hashlib.sha256((ROOT/'scripts/task2_smote.py').read_bytes()).hexdigest()
    if args.check_only or checks.get('smote_source_sha256') != smote_hash:
        run([sys.executable, '-u', 'scripts/task2_smote.py', '--check-only'], env)
        checks['smote_source_sha256'] = smote_hash
        checks_path.write_text(json.dumps(checks)+'\n')
    if args.check_only or not checks.get('passed'):
        run(base+['--phase', 'check', '--job', 'model_fixtures'], env)
        checks['passed'] = True
        checks_path.write_text(json.dumps(checks)+'\n')
    if args.check_only:
        return
    run(base+['--phase', 'prepare', '--job', 'model_preparation'], env)
    for q in [0, 5, 25]:
        run([sys.executable, '-u', 'scripts/task2_smote.py', '--threshold', str(q)], env)
    run(base+['--phase', 'smote-vectors', '--job', 'smote_vectors'], env)
    for algorithm in args.algorithms:
        for policy in args.policies:
            run(base+['--phase', 'fit', '--algorithm', algorithm, '--policy', policy,
                      '--job', f'{algorithm}_{policy}'], env)
    if args.fit_only:
        return
    if set(args.algorithms) != set(ALGORITHMS) or set(args.policies) != set(POLICIES):
        print('PARTIAL_BENCHMARK_COMPLETE; full selection requires all four algorithms and three policies', flush=True)
        return
    run(base+['--phase', 'select', '--job', 'validation_selection'], env)
    for algorithm in args.algorithms:
        run(base+['--phase', 'test', '--algorithm', algorithm, '--job', f'{algorithm}_test'], env)
    run(base+['--phase', 'train-diagnostics', '--job', 'full_train_diagnostics'], env)
    run(base+['--phase', 'artifacts', '--job', 'model_artifact_checks'], env)
    summary = {'status': 'COMPLETE', 'preparation': json.loads((work/'preparation.json').read_text()),
               'selection': json.loads((work/'selection.json').read_text()),
               'smote': {str(q): json.loads((work/'smote'/f'gt_{q}mm'/'generation.json').read_text()) for q in [0,5,25]},
               'candidates': {a: {p: json.loads((work/'results'/f'{a}_{p}.json').read_text()) for p in POLICIES} for a in ALGORITHMS},
               'test': {a: json.loads((work/'test'/f'{a}.json').read_text()) for a in ALGORITHMS},
               'train_final': {str(q): json.loads((work/'train_final'/f'gt_{q}mm.json').read_text()) for q in [0,5,25]},
               'artifact_checks': json.loads((work/'artifact_checks.json').read_text()),
               'current_invocation_seconds': time.monotonic()-started}
    (work/'summary.json').write_text(json.dumps(summary, indent=2, default=str)+'\n')
    print('FULL_MODEL_COMPARISON_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
