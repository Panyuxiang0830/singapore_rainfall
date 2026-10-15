"""Record a health snapshot for the enhanced full-experiment driver every 30 minutes."""
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'work/cloud/experiments/full_variants'
DRIVER_STATUS = ROOT / 'work/cloud/experiments/driver_status.json'
OUT = WORK / 'health.jsonl'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def snapshot():
    driver_pid = None
    try:
        driver_pid = int((WORK / 'driver.pid').read_text().strip())
    except (OSError, ValueError):
        pass
    alive = bool(driver_pid and psutil.pid_exists(driver_pid))
    state = read_json(WORK / 'status.json') or {}
    phase = read_json(DRIVER_STATUS) or {}
    disk = shutil.disk_usage(ROOT)
    return {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'driver_pid': driver_pid,
        'driver_alive': alive,
        'experiment_status': state.get('status'),
        'variant': state.get('variant'),
        'phase': state.get('phase'),
        'phase_monitor_status': phase.get('status'),
        'phase_elapsed_s': phase.get('elapsed_s'),
        'available_memory_gib': round(psutil.virtual_memory().available / 1024**3, 2),
        'free_disk_gib': round(disk.free / 1024**3, 2),
        'healthy': alive and state.get('status') == 'RUNNING' and phase.get('status') != 'FAILED',
    }


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    while True:
        state = snapshot()
        with OUT.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(state) + '\n')
        if state['experiment_status'] in {'COMPLETE', 'FAILED'} or not state['driver_alive']:
            return
        time.sleep(1800)


if __name__ == '__main__':
    main()
