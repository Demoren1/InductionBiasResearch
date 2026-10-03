from . import updates as _implementation
from .updates import *


def __getattr__(name):
    return getattr(_implementation, name)
