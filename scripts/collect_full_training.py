"""Copy completed full-training artifacts from gpushare and verify every byte.

Working metrics remain in ignored work/. Native pipelines enter models/.
By default the complete experiment is required; --checkpoint permits copying
only finished pipelines while the remaining phases continue on the server.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def ssh(host, command, **kwargs):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                           '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=1',
                           '-o', 'ControlMaster=no', '-o', 'ControlPath=none', host, command],
                          check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpushare')
    parser.add_argument('--remote-root', default='/home/research_pyx/singapore')
    parser.add_argument('--checkpoint', action='store_true')
    args = parser.parse_args()
    code = '''import hashlib,json
from pathlib import Path
root=Path(REMOTE_ROOT)
base=root/'work/task2/logistic_full'
bench=root/'work/task2/model_comparison_full'
prepared=json.loads((base/'preparation.json').read_text())
assert prepared['train_rows']==34542285 and prepared['config']['training_policy']=='all_eligible_2017_2022'
status=json.loads((root/'work/task2/full_run/status.json').read_text())
if not CHECKPOINT:
 assert status['status']=='COMPLETE','Full training is not complete'
summary=bench/'summary.json'
if summary.exists():
 finished=json.loads(summary.read_text())
 assert finished['status']=='COMPLETE' and finished['preparation']['train_rows']==34542285
models=[root/'models/logistic_baseline_full/preprocessing']
for group in ['calendar_station','own_history','neighbors']:
 for q in [0,5,25]:
  marker=base/f'{group}_gt_{q}mm.json'
  if marker.exists():
   result=json.loads(marker.read_text())
   assert result['serialization_roundtrip_passed']
   models.append(root/result['model_relative_path'])
for algorithm in ['logistic','svm','gbt','xgb']:
 for policy in ['natural','balanced','smote']:
  for q in [0,5,25]:
   marker=bench/'results'/f'{algorithm}_{policy}_gt_{q}mm.json'
   if marker.exists():
    result=json.loads(marker.read_text())
    assert result['real_training_rows']==34542285
    if result.get('calibration'):
     models.append(root/result['model_relative_path'])
files=set()
for folder in models:
 assert folder.is_dir() and folder.resolve().is_relative_to(root.resolve())
 files.update(p for p in folder.rglob('*') if p.is_file())
for folder in [root/'models/logistic_baseline_full',root/'models/model_comparison_full']:
 marker=folder/'manifest.json'
 if marker.exists():files.add(marker)
for folder in [base,bench]:
 files.update(folder.glob('*.json'))
for name in ['results','test','train_final','curves']:
 files.update((bench/name).glob('*.json'))
files.update((bench/'smote').glob('gt_*mm/generation.json'))
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
print(json.dumps({'status':status,'pipelines':len(models)-1,'files':manifest}))
'''.replace('REMOTE_ROOT', repr(args.remote_root)).replace('CHECKPOINT', repr(args.checkpoint))
    result = ssh(args.host, 'python3 -c '+shlex.quote(code), capture_output=True, text=True, timeout=45)
    manifest = json.loads(result.stdout)
    paths = list(manifest['files'])
    allowed = ['models/logistic_baseline_full/', 'models/model_comparison_full/',
               'work/task2/logistic_full/', 'work/task2/model_comparison_full/']
    assert all(any(p.startswith(a) for a in allowed) for p in paths)
    assert all('\n' not in p for p in paths)
    work = ROOT/'work/task2/collection'
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='snapshot_', dir=work) as temporary:
        archive = Path(temporary)/'transfer.tar.gz'
        command = 'tar -czf - -C '+shlex.quote(args.remote_root)+' --verbatim-files-from -T -'
        with archive.open('wb') as output:
            ssh(args.host, command, input=('\n'.join(paths)+'\n').encode(), stdout=output, timeout=180)
        with tarfile.open(archive, 'r:gz') as source:
            for member in source.getmembers():
                assert member.isfile() and member.name in manifest['files']
                assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            safe_filter = {'filter': 'data'} if 'filter' in inspect.signature(source.extractall).parameters else {}
            source.extractall(Path(temporary)/'files', **safe_filter)
        import shutil
        for relative, digest in manifest['files'].items():
            incoming = Path(temporary)/'files'/relative
            assert hashlib.sha256(incoming.read_bytes()).hexdigest()==digest, relative
        for relative in paths:
            target = ROOT/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(temporary)/'files'/relative, target)
    (work/'latest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'server_status':manifest['status']['status'],
                      'finished_pipelines':manifest['pipelines'],
                      'verified_files':len(paths), 'train_rows':34542285}, indent=2))


if __name__ == '__main__':
    main()
