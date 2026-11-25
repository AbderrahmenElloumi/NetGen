"""Atomic output writing: a command's output appears whole or not at all.

The caller writes into a temporary folder beside the target. The module
renames the result into place only when the write succeeds. An existing
target is refused unless `force` is given, and with `force` it is replaced
only once the new output is complete.
"""

import os
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from netgen.errors import NetGenError


@contextmanager
def write_directory(target: Path, *, force: bool = False) -> Iterator[Path]:
    """Yield an empty folder to fill; it becomes `target` on success."""
    with _staged([target], force=force, directory=True) as (folder,):
        yield folder


@contextmanager
def write_files(*targets: Path, force: bool = False) -> Iterator[tuple[Path, ...]]:
    """Yield one path per target file; all of them are put in place together on success."""
    parents = {target.parent for target in targets}
    if len(parents) != 1 or len(set(targets)) != len(targets):
        raise ValueError("file targets must be distinct files in one folder")
    with _staged(targets, force=force, directory=False) as paths:
        yield paths


@contextmanager
def _staged(
    targets: Sequence[Path], *, force: bool, directory: bool
) -> Iterator[tuple[Path, ...]]:
    if not force:
        _refuse_existing(targets)
    parent = targets[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix=f".{targets[0].name}.", suffix=".tmp", dir=parent)
    )
    old = work / "old"
    keep = False
    try:
        new = work / "new"
        new.mkdir()
        paths = tuple(new / target.name for target in targets)
        if directory:
            paths[0].mkdir()
        yield paths
        _put_in_place(paths, targets, old, force=force)
    except BaseException as error:
        # An old target the rollback could not restore is still in `old`:
        # keep the temporary folder so the previous output is not deleted.
        keep = old.exists() and any(old.iterdir())
        if keep:
            error.add_note(
                f"The previous output could not be restored; it is kept in {old}."
            )
        raise
    finally:
        if not keep:
            # A failed cleanup must not replace the caller's exception, nor
            # report a failure once the output is in place.
            shutil.rmtree(work, ignore_errors=True)


def _refuse_existing(targets: Sequence[Path]) -> None:
    for target in targets:
        if target.exists(follow_symlinks=False):
            raise NetGenError(f"{target} already exists. Pass --force to replace it.")


def _put_in_place(
    paths: Sequence[Path], targets: Sequence[Path], old: Path, *, force: bool
) -> None:
    """Rename each path onto its target; on any failure restore the old targets."""
    if not force:
        # A target may have appeared since the check before the write.
        _refuse_existing(targets)
    old.mkdir()
    set_aside: list[tuple[Path, Path]] = []
    placed: list[Path] = []
    try:
        for target in targets:
            if force and target.exists(follow_symlinks=False):
                aside = old / target.name
                os.replace(target, aside)
                set_aside.append((aside, target))
        for path, target in zip(paths, targets, strict=True):
            os.replace(path, target)
            placed.append(target)
    except BaseException:
        _roll_back(placed, set_aside)
        raise


def _roll_back(placed: Sequence[Path], set_aside: Sequence[tuple[Path, Path]]) -> None:
    """Undo a partial put-in-place; an old target that cannot be restored stays aside."""
    for target in placed:
        with suppress(OSError):
            _remove(target)
    for aside, target in set_aside:
        with suppress(OSError):
            os.replace(aside, target)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
