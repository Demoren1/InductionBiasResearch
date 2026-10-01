"""Content-address all new experiment artifacts; preserve cross-root reuse links."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['20261001_other_generators', '20261001_residual_fields',
    '20261001_other_generators_corrected', '20261001_generative_extraction_controls']
OUT = ROOT / 'outputs/deepsets_vaae/20261001_other_generators_corrected'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()

def main():
    files, links = {}, {}
    destination = OUT / 'artifact_manifest.json'
    for name in NAMES:
        base = ROOT / 'outputs/deepsets_vaae' / name
        for current, folders, names in os.walk(base, followlinks=False):
            for child in folders + names:
                path = Path(current) / child
                key = str(path.relative_to(ROOT))
                if path == destination:
                    continue
                if path.is_symlink():
                    resolved = path.resolve(strict=True)
                    links[key] = dict(target=os.readlink(path), resolved=str(resolved),
                        covered_by_manifest_roots=any(resolved.is_relative_to(ROOT / 'outputs/deepsets_vaae' / n) for n in NAMES))
                elif path.is_file():
                    files[key] = dict(bytes=path.stat().st_size, sha256=digest(path))
    report = ROOT / 'mds/FINAL_REPORT_2026-09-27.md'
    files[str(report.relative_to(ROOT))] = dict(bytes=report.stat().st_size, sha256=digest(report))
    result = dict(status='PASS', roots=NAMES, algorithm='SHA256', files=files, reuse_symlinks=links,
        file_count=len(files), bytes=sum(v['bytes'] for v in files.values()),
        limitation='Snapshot of saved files; original input-bank hashes are separately frozen in each protocol. The manifest does not hash itself.')
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['status','file_count','bytes']}))

if __name__ == '__main__':
    main()
