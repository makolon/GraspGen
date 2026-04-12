from importlib import import_module

from grasp_gen.utils.logging_config import setup_logging

# Set up logging when the package is imported
setup_logging()

# Version info
__version__ = "0.1.0"

_EXPORTED_SUBMODULES = {
    "assets",
    "config",
    "dataset",
    "models",
    "serving",
    "utils",
}


def __getattr__(name):
    if name in _EXPORTED_SUBMODULES:
        return import_module("{}.{}".format(__name__, name))
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))


__all__ = ["__version__"] + sorted(_EXPORTED_SUBMODULES)
