"""Install a project-local Spark/Java runtime without changing global packages."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    scratch = ROOT / 'work/tmp'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(scratch)
    runtime = ROOT / '.runtime'
    runtime.mkdir(exist_ok=True)
    if shutil.disk_usage(ROOT).free < 5.7 * 1024**3:
        raise RuntimeError('Insufficient space for isolated runtime while preserving 5 GiB free')
    if not (runtime / 'java/bin/java').exists():
        generic = ROOT / 'work/java_package.json'
        if generic.exists():
            pkg = json.loads(generic.read_text())
        else:
            asset = json.loads((ROOT / 'work/java_asset.json').read_text())
            pkg = asset['binary']['package']
        archive = runtime / 'java.tar.gz'
        if archive.exists():
            with archive.open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != pkg['checksum']:
                    archive.unlink()  # Only this project's interrupted archive.
        if not archive.exists():
            with requests.get(pkg['link'], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with archive.open('wb') as out:
                    for chunk in response.iter_content(256 * 1024):
                        out.write(chunk)
        with archive.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256')
        if digest.hexdigest() != pkg['checksum']:
            raise RuntimeError('Java archive checksum mismatch')
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for member in members:
                target = (runtime / member.name).resolve()
                if not target.is_relative_to(runtime.resolve()):
                    raise RuntimeError('Unsafe archive path')
            top = members[0].name.split('/')[0]
            tar.extractall(runtime, filter='data')
        extracted = runtime / top
        nested_home = extracted / 'Contents/Home'
        if nested_home.exists():
            nested_home.rename(runtime / 'java')
            shutil.rmtree(extracted)  # Only the package extracted by this script.
        else:
            extracted.rename(runtime / 'java')
        archive.unlink()
        print('Project-local Java installed', flush=True)
    spark_archive = runtime / 'pyspark-3.5.8.tar.gz'
    if '--java-only' not in sys.argv and spark_archive.exists() and not (runtime / 'spark/deps/bin/spark-submit').exists():
        asset = json.loads((ROOT / 'work/spark_asset.json').read_text())
        with spark_archive.open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != asset['digests']['sha256']:
                raise RuntimeError('Spark archive checksum mismatch')
        with tarfile.open(spark_archive) as tar:
            for member in tar.getmembers():
                if not (runtime / member.name).resolve().is_relative_to(runtime.resolve()):
                    raise RuntimeError('Unsafe Spark archive path')
            tar.extractall(runtime, filter='data')
        (runtime / 'pyspark-3.5.8').rename(runtime / 'spark')
        python = runtime / 'spark/deps/python'
        python.mkdir(exist_ok=True)
        (python / 'lib').symlink_to('../../lib')
        spark_archive.unlink()
        print('Verified packaged Spark runtime extracted without wheel build', flush=True)
    if ('--java-only' not in sys.argv and not (runtime / 'spark/deps/bin/spark-submit').exists()
            and not (runtime / 'python/pyspark').exists()):
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
            '--target', str(runtime / 'python'), '--no-deps',
            'pyspark==3.5.8', 'py4j==0.10.9.7',
        ], check=True)
    subprocess.run([str(runtime / 'java/bin/java'), '-version'], check=True)
    print('Spark/Java runtime ready; free GiB:', round(shutil.disk_usage(ROOT).free / 1024**3, 2), flush=True)


if __name__ == '__main__':
    main()
