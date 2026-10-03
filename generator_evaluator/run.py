"""Public entry point for the runners.legacy runner."""
import sys
from .runners import legacy as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
