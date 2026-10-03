"""Run unittest classes and plain test functions without a pytest dependency."""
import importlib
import inspect
from pathlib import Path
import unittest

import torch

torch.set_num_threads(1)
suite = unittest.TestSuite()
for path in sorted(Path(__file__).parent.glob("test_*.py")):
    module = importlib.import_module(f"generator_evaluator.tests.{path.stem}")
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    for name, func in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("test_") and not inspect.signature(func).parameters:
            suite.addTest(unittest.FunctionTestCase(func))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
