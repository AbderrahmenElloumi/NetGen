"""NetGen — network traffic dataset generator."""

import tomllib
from importlib import metadata
from pathlib import Path


def _version() -> str:
    """The installed package's version, or the repository's `pyproject.toml` version.

    The containers mount the source and set `PYTHONPATH` without installing
    the package, so there the package metadata is missing.
    """
    try:
        return metadata.version("netgen")
    except metadata.PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        return str(project["version"])


__version__ = _version()
