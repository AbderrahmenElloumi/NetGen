"""The reader of the public allowlist: which repository paths go into the public tree.

The format is fixed for both readers, these tests and the release script:
one entry per line; blank lines and lines starting with `#` are ignored; an
entry ending in `/` matches that directory and everything below it; any other
entry matches exactly one file; a line starting with `!` excludes what it
matches, and an exclusion always wins. Paths are relative to the repository
root, with forward slashes.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# Private: absent from the public repository, where its readers skip.
ALLOWLIST = REPO / "scripts" / "public-allowlist.txt"


@dataclass(frozen=True)
class Allowlist:
    include: tuple[str, ...]
    exclude: tuple[str, ...]


def parse(text: str) -> Allowlist:
    """The entries of an allowlist; ValueError names the first malformed line."""
    include: list[str] = []
    exclude: list[str] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        excluded = line.startswith("!")
        entry = line[1:] if excluded else line
        segments = entry.rstrip("/").split("/")
        if (
            not entry
            or entry.startswith("/")
            or "\\" in entry
            or any(char.isspace() for char in entry)
            or any(segment in ("", ".", "..") for segment in segments)
        ):
            raise ValueError(f"allowlist line {number}: malformed entry {raw!r}")
        (exclude if excluded else include).append(entry)
    return Allowlist(tuple(include), tuple(exclude))


def load(path: Path = ALLOWLIST) -> Allowlist:
    """The allowlist at `path`; skips the calling test when the file is absent."""
    if not path.exists():
        pytest.skip(f"{path.name} is absent")
    return parse(path.read_text(encoding="utf-8"))


def _matches(entry: str, path: str) -> bool:
    if entry.endswith("/"):
        return path == entry.rstrip("/") or path.startswith(entry)
    return path == entry


def is_public(path: str, allowlist: Allowlist) -> bool:
    """Whether the repository path `path` (a file or a directory) is in the public tree."""
    path = path.rstrip("/")
    if any(_matches(entry, path) for entry in allowlist.exclude):
        return False
    return any(_matches(entry, path) for entry in allowlist.include)
