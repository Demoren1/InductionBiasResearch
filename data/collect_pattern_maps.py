#!/usr/bin/env python3
"""Collect sparse weights, support-BCE gradients and functional maps (IMP K=32).

From the project root:
    python data/collect_pattern_maps.py --bank-id imp32_wgf_v2 --device auto
See pattern/README.md for configuration and model training.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pattern.bank.collect import main

if __name__ == "__main__":
    main()
