from importlib.metadata import version, PackageNotFoundError

from .clear import CLEAR
from . import metrics
from . import utils
from . import models

try:
    __version__ = version("clear-uq")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["CLEAR", "metrics", "utils", "models", "__version__"]
