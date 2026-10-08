#!/usr/bin/env python3
"""Collect a new immutable pattern IMP bank with exactly 32 connections.

From the project root:
    python data/collect_pattern_maps.py --bank-id imp32_v1 --device auto
See pattern/README.md for configuration and model training.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pattern.bank.collect import main

if __name__ == "__main__":
    main()
