"""Mask generators, a frozen critic and measured functional banks."""
import sys as _sys
from .data import adapters
from .storage import banks as bank_storage

# Older FunctionalBank artifacts refer to these original module paths.
_sys.modules[__name__ + ".adapters"] = adapters
_sys.modules[__name__ + ".bank_storage"] = bank_storage
