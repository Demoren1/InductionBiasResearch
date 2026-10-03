"""Public entry point for the evaluation.frozen runner."""
import sys
from .evaluation import frozen as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
