
import importlib


_SUBMODULES = {"dataset", "metrics", "rta", "layer"}


def __getattr__(name):
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_SUBMODULES)
