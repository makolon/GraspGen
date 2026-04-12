from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def get_path():
    return PACKAGE_ROOT


__all__ = ["PACKAGE_ROOT", "get_path"]
