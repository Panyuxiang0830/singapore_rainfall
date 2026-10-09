"""Select checksum-pinned official Java metadata for the supported platforms."""
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata = ROOT / 'runtime_metadata'
    system, machine = platform.system(), platform.machine().lower()
    if system == 'Darwin' and machine in ('arm64', 'aarch64'):
        asset = json.loads((metadata / 'mac_java_asset.json').read_text())
        package = asset['binary']['package']
    elif system == 'Linux' and machine in ('x86_64', 'amd64'):
        asset = json.loads((metadata / 'zulu_asset.json').read_text())
        package = {'link': asset['download_url'], 'checksum': asset['sha256_hash'], 'size': asset['size']}
    else:
        raise RuntimeError('Provide official Java 17 package metadata for this OS/architecture in work/java_package.json')
    work = ROOT / 'work'
    work.mkdir(exist_ok=True)
    (work / 'java_package.json').write_text(json.dumps(package, indent=2) + '\n')
    (work / 'spark_asset.json').write_text((metadata / 'spark_asset.json').read_text())
    print('Selected official Java package:', package['link'])


if __name__ == '__main__':
    main()
