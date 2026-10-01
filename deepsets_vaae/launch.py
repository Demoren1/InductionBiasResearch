"""Run one independent repeat on every GPU with exactly zero utilization."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bank-steps', type=int, default=800)
    parser.add_argument('--eval-steps', type=int, default=800)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'allocation.json').exists():
        raise SystemExit('Allocation already exists; use a fresh output directory.')
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    snapshot = []
    for line in raw.strip().splitlines():
        index, uuid, utilization = (part.strip() for part in line.split(','))
        snapshot.append({'index': int(index), 'uuid': uuid, 'utilization': int(utilization)})
    selected = [gpu for gpu in snapshot if gpu['utilization'] == 0]
    if not selected:
        raise SystemExit('No GPU currently has zero utilization.')
    allocation = {'date_moscow': '2026-10-01', 'launcher_pid': os.getpid(),
                  'memory_considered': False, 'snapshot': snapshot, 'workers': []}
    processes = []
    for gpu in selected:
        seed = 4100 + gpu['index']
        folder = args.out / f'seed_{seed}'
        folder.mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1')
        command = [sys.executable, '-m', 'deepsets_vaae.run', '--out', str(folder),
                   '--seed', str(seed), '--bank-steps', str(args.bank_steps),
                   '--eval-steps', str(args.eval_steps)]
        log = (folder / 'run.log').open('w')
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((proc, log))
        allocation['workers'].append({**gpu, 'seed': seed, 'pid': proc.pid,
                                      'command': command, 'output': str(folder)})
    save(args.out / 'allocation.json', allocation)
    print(json.dumps({'launched': allocation['workers']}, ensure_ascii=False), flush=True)
    remaining = set(range(len(processes)))
    while remaining:
        for index in list(remaining):
            proc, log = processes[index]
            code = proc.poll()
            if code is not None:
                log.close()
                allocation['workers'][index]['exit_code'] = code
                remaining.remove(index)
                save(args.out / 'allocation.json', allocation)
                print(json.dumps({'seed': allocation['workers'][index]['seed'], 'exit_code': code}), flush=True)
        if remaining:
            time.sleep(5)
    successful = [worker for worker in allocation['workers'] if worker['exit_code'] == 0]
    if successful:
        with (args.out / 'report.log').open('w') as report_log:
            result = subprocess.run([sys.executable, '-m', 'deepsets_vaae.report',
                                     '--out', str(args.out)], stdout=report_log,
                                    stderr=subprocess.STDOUT)
        allocation['report_exit_code'] = result.returncode
        save(args.out / 'allocation.json', allocation)
    if len(successful) != len(processes) or allocation.get('report_exit_code', 1) != 0:
        raise SystemExit('Some workers or the report failed; inspect logs.')
    print('All selected GPUs completed; report saved.', flush=True)


if __name__ == '__main__':
    main()
