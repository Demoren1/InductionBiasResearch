"""Export target inputs only after the required source/development choices exist."""

import argparse
import json
import time
from pathlib import Path

from ..protocol.real_target_export import export as export_real
from ..protocol.target_export import export as export_synthetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--real", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    root = args.synthetic.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    expected = len(protocol["cases"]) * len(protocol["conditions"])
    while True:
        export_synthetic(root)
        for real in args.real:
            export_real(real.resolve())
        if len(list((root / "targets").glob("*/ready.json"))) == expected and all(
            (real / "targets" / "ready.json").exists() for real in args.real
        ):
            break
        time.sleep(10)


if __name__ == "__main__":
    main()
