#!/usr/bin/env bash
# Run one profile, both registered radii, in the active ras environment.
# Example: CUDA_VISIBLE_DEVICES=2 bash scripts/11_oracle_ideal.sh interp
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u - "$@" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

profile = sys.argv[1] if len(sys.argv) > 1 else 'interp'
root = Path('outputs/oracle_ideal/seed_20260906').resolve()
protocol_path = root / 'protocol.json'
protocol = json.loads(protocol_path.read_text())
source = protocol['profiles'][profile]
for kind in ('checkpoint', 'split'):
    assert hashlib.sha256(Path(source[kind]).read_bytes()).hexdigest() == source[kind + '_sha256']
for radius in protocol['radii']:
    name = f'{profile}_radius_{radius:g}'
    destination = root / name
    command = [sys.executable, '-u', 'evaluation/oracle_ideal.py',
               '--checkpoint', source['checkpoint'], '--split', source['split'],
               '--out-dir', str(destination), '--protocol', str(protocol_path),
               '--radius', str(radius), '--device', 'cuda']
    for key, flag in [('n_starts', '--n-starts'), ('steps', '--steps'), ('seed', '--seed'),
                      ('lr', '--lr'), ('temperature', '--temperature')]:
        command += [flag, str(protocol[key])]
    print('Starting', name, flush=True)
    with (root / f'{name}.log').open('x') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Completed', name, flush=True)
PY
