"""Run/resume nine MLlib logistic ablations sequentially under resource guards."""
import argparse
import fcntl
import json
import os
import sys

from run_task1 import ROOT, run

GROUPS = ['calendar_station', 'own_history', 'neighbors']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--predict-station')
    parser.add_argument('--predict-time')
    parser.add_argument('--threshold', type=int, choices=[0, 5, 25])
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--groups', choices=GROUPS, nargs='+', default=GROUPS)
    parser.add_argument('--train-fraction', type=float, default=0.05)
    parser.add_argument('--reg-param', type=float, default=0.001)
    parser.add_argument('--max-iter', type=int, default=80)
    parser.add_argument('--tol', type=float, default=1e-6)
    args = parser.parse_args()
    work = ROOT / 'work/task2/logistic'
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'run.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another logistic experiment is active')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
    env = os.environ.copy()
    env.update({'JAVA_HOME': str(ROOT/'.runtime/java'), 'PYTHONPATH': str(ROOT/'.runtime/python'),
                'JAVA_TOOL_OPTIONS': '-XX:ActiveProcessorCount=2', 'OPENBLAS_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1', 'SPARK_LOCAL_IP': '127.0.0.1',
                'PYSPARK_PYTHON': sys.executable, 'TMPDIR': str(ROOT/'work/tmp')})
    env['PATH'] = env['JAVA_HOME']+'/bin:'+env['PATH']
    base = [sys.executable, '-u', 'scripts/task2_logistic.py', '--train-fraction', str(args.train_fraction),
            '--reg-param', str(args.reg_param), '--max-iter', str(args.max_iter), '--tol', str(args.tol)]
    if args.check_only:
        run([sys.executable, '-u', 'scripts/check_task2_logistic.py'], env)
        return
    if args.predict_station:
        if not args.predict_time or args.threshold is None:
            parser.error('Prediction requires --predict-time and --threshold')
        run([sys.executable, '-u', 'scripts/predict_task2_logistic.py',
             '--station', args.predict_station, '--time', args.predict_time,
             '--threshold', str(args.threshold)], env)
        return
    if not args.summarize_only:
        run(base+['--prepare'], env)
        for group in args.groups:
            run(base+['--group', group], env)
    summary = {'status': 'COMPLETE', 'groups': args.groups,
               'preparation': json.loads((work/'preparation.json').read_text()),
               'results': {g: json.loads((work/f'{g}.json').read_text()) for g in args.groups}}
    summary['selected_group_by_validation_pr_auc'] = {
        str(q): max(args.groups, key=lambda g: summary['results'][g]['models'][str(q)]['validation']['pr_auc'])
        for q in [0, 5, 25]}
    (work/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    model_manifest = {k: summary[k] for k in ['selected_group_by_validation_pr_auc', 'preparation']}
    model_manifest['probability_policy'] = 'Independent threshold classifiers; raw probabilities, no cross-threshold monotonicity guarantee'
    model_manifest['thresholds_mm'] = [0, 5, 25]
    model_manifest['model_format'] = 'Spark 3.5.8 PipelineModel (training-fitted imputer, station encoder, assembler, LR)'
    (ROOT/'models/logistic_baseline/manifest.json').write_text(json.dumps(model_manifest, indent=2)+'\n')
    print('LOGISTIC_COMPARISON_COMPLETE', json.dumps(summary['selected_group_by_validation_pr_auc']), flush=True)


if __name__ == '__main__':
    main()
