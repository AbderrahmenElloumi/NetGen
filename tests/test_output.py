"""Tests of atomic output writing (netgen.output): all or nothing, no overwrite without force."""

import os
from pathlib import Path

import pytest

from netgen import output
from netgen.errors import NetGenError


class Boom(Exception):
    """The failure a test raises inside a write."""


def entries(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def make_old_directory(target: Path) -> None:
    target.mkdir(parents=True)
    (target / "old.txt").write_text("old")


def make_old_files(dataset: Path, report: Path) -> None:
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("old dataset")
    report.write_text("old report")


def test_d03_failed_write_leaves_nothing_and_overwrite_needs_force(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artefacts"
    with pytest.raises(Boom), output.write_directory(target) as folder:
        (folder / "manifest.json").write_text("{}")
        raise Boom
    assert entries(tmp_path) == []

    make_old_directory(target)
    with pytest.raises(NetGenError, match="--force"), output.write_directory(target):
        pass
    assert entries(target) == ["old.txt"]


def test_directory_write_appears_complete_and_leaves_no_temporary_folder(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artefacts"
    with output.write_directory(target) as folder:
        (folder / "manifest.json").write_text("{}")
        (folder / "fill.json").write_text("{}")
    assert entries(tmp_path) == ["artefacts"]
    assert entries(target) == ["fill.json", "manifest.json"]


def test_file_group_write_puts_both_files_in_place(tmp_path: Path) -> None:
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    with output.write_files(dataset, report) as (new_dataset, new_report):
        new_dataset.write_text("dataset")
        new_report.write_text("report")
    assert entries(tmp_path) == ["data.parquet", "data.report.json"]
    assert dataset.read_text() == "dataset"
    assert report.read_text() == "report"


def test_failing_file_group_write_leaves_nothing(tmp_path: Path) -> None:
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    with pytest.raises(Boom), output.write_files(dataset, report) as (new_dataset, _):
        new_dataset.write_text("partial")
        raise Boom
    assert entries(tmp_path) == []


def test_existing_file_target_is_refused_without_force(tmp_path: Path) -> None:
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    report.write_text("old report")
    with (
        pytest.raises(NetGenError, match="--force") as refusal,
        output.write_files(dataset, report),
    ):
        pytest.fail("the write must not start")
    assert str(report) in str(refusal.value)
    assert entries(tmp_path) == ["data.report.json"]
    assert report.read_text() == "old report"


def test_existing_directory_target_is_refused_before_work_starts(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artefacts"
    make_old_directory(target)
    with (
        pytest.raises(NetGenError, match="--force") as refusal,
        output.write_directory(target),
    ):
        pytest.fail("the write must not start")
    assert str(target) in str(refusal.value)
    assert entries(tmp_path) == ["artefacts"]
    assert entries(target) == ["old.txt"]


def test_force_replaces_directory_after_successful_write(tmp_path: Path) -> None:
    target = tmp_path / "artefacts"
    make_old_directory(target)
    with output.write_directory(target, force=True) as folder:
        assert entries(target) == ["old.txt"]
        (folder / "new.txt").write_text("new")
    assert entries(tmp_path) == ["artefacts"]
    assert entries(target) == ["new.txt"]


def test_force_replaces_file_group_after_successful_write(tmp_path: Path) -> None:
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    make_old_files(dataset, report)
    with output.write_files(dataset, report, force=True) as (new_dataset, new_report):
        assert dataset.read_text() == "old dataset"
        new_dataset.write_text("new dataset")
        new_report.write_text("new report")
    assert entries(tmp_path) == ["data.parquet", "data.report.json"]
    assert dataset.read_text() == "new dataset"
    assert report.read_text() == "new report"


def test_force_with_failing_directory_write_keeps_old_target(tmp_path: Path) -> None:
    target = tmp_path / "artefacts"
    make_old_directory(target)
    with pytest.raises(Boom), output.write_directory(target, force=True) as folder:
        (folder / "new.txt").write_text("new")
        raise Boom
    assert entries(tmp_path) == ["artefacts"]
    assert entries(target) == ["old.txt"]


def test_force_with_failing_file_group_write_keeps_old_targets(tmp_path: Path) -> None:
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    make_old_files(dataset, report)
    with (
        pytest.raises(Boom),
        output.write_files(dataset, report, force=True) as (new_dataset, _),
    ):
        new_dataset.write_text("new dataset")
        raise Boom
    assert entries(tmp_path) == ["data.parquet", "data.report.json"]
    assert dataset.read_text() == "old dataset"
    assert report.read_text() == "old report"


def test_failed_rename_into_place_restores_old_targets(tmp_path: Path) -> None:
    # The caller never writes the report: the rename of the second file fails
    # after the first one is in place, and the group is rolled back.
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    make_old_files(dataset, report)
    with (
        pytest.raises(FileNotFoundError),
        output.write_files(dataset, report, force=True) as (new_dataset, _),
    ):
        new_dataset.write_text("new dataset")
    assert entries(tmp_path) == ["data.parquet", "data.report.json"]
    assert dataset.read_text() == "old dataset"
    assert report.read_text() == "old report"


def test_failed_restore_keeps_the_old_target_in_the_temporary_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The rename of the report into place fails (the caller never writes it),
    # and then the restore of the old report fails too.
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    make_old_files(dataset, report)
    replace = os.replace

    def failing_restore(src: Path, dst: Path) -> None:
        source = Path(src)
        if (
            Path(dst) == report
            and source.is_file()
            and source.read_text() == "old report"
        ):
            raise PermissionError("restore fails")
        replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_restore)
    with (
        pytest.raises(FileNotFoundError) as failure,
        output.write_files(dataset, report, force=True) as (new_dataset, _),
    ):
        new_dataset.write_text("new dataset")
    monkeypatch.undo()
    assert dataset.read_text() == "old dataset"
    (kept,) = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and path.read_text() == "old report"
    ]
    assert kept != report
    assert any(str(kept.parent) in note for note in failure.value.__notes__)


def test_failed_removal_of_a_placed_file_still_restores_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    make_old_files(dataset, report)
    unlink = Path.unlink

    def failing_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == dataset:
            raise PermissionError("removal fails")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    with (
        pytest.raises(FileNotFoundError),
        output.write_files(dataset, report, force=True) as (new_dataset, _),
    ):
        new_dataset.write_text("new dataset")
    monkeypatch.undo()
    assert entries(tmp_path) == ["data.parquet", "data.report.json"]
    assert dataset.read_text() == "old dataset"
    assert report.read_text() == "old report"


def test_target_that_appears_during_a_write_is_refused_without_force(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artefacts"
    with (
        pytest.raises(NetGenError, match="--force") as refusal,
        output.write_directory(target) as outer,
    ):
        (outer / "outer.txt").write_text("outer")
        with output.write_directory(target) as inner:
            (inner / "inner.txt").write_text("inner")
    assert str(target) in str(refusal.value)
    assert entries(tmp_path) == ["artefacts"]
    assert entries(target) == ["inner.txt"]


def test_file_that_appears_during_a_write_is_refused_without_force(
    tmp_path: Path,
) -> None:
    dataset, report = tmp_path / "data.parquet", tmp_path / "data.report.json"
    with (
        pytest.raises(NetGenError, match="--force"),
        output.write_files(dataset, report) as (new_dataset, new_report),
    ):
        new_dataset.write_text("new dataset")
        new_report.write_text("new report")
        report.write_text("other report")
    assert entries(tmp_path) == ["data.report.json"]
    assert report.read_text() == "other report"


def test_dangling_symlink_target_is_refused_without_force(tmp_path: Path) -> None:
    target = tmp_path / "artefacts"
    target.symlink_to(tmp_path / "missing")
    with (
        pytest.raises(NetGenError, match="--force"),
        output.write_directory(target),
    ):
        pytest.fail("the write must not start")
    assert target.is_symlink()
    assert entries(tmp_path) == ["artefacts"]


def test_cleanup_failure_does_not_replace_the_outcome(tmp_path: Path) -> None:
    # A read-only folder inside the temporary folder makes its removal fail.
    target = tmp_path / "artefacts"
    try:
        with pytest.raises(Boom), output.write_directory(target) as folder:
            (folder / "locked").mkdir()
            (folder / "locked" / "file.txt").write_text("new")
            (folder / "locked").chmod(0o500)
            raise Boom

        make_old_directory(target)
        (target / "locked").mkdir()
        (target / "locked" / "file.txt").write_text("old")
        (target / "locked").chmod(0o500)
        with output.write_directory(target, force=True) as folder:
            (folder / "new.txt").write_text("new")
        assert entries(target) == ["new.txt"]
    finally:
        for path in tmp_path.rglob("locked"):
            path.chmod(0o700)


def test_parent_directories_are_created(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "artefacts"
    with output.write_directory(target) as folder:
        (folder / "manifest.json").write_text("{}")
    dataset = tmp_path / "c" / "d" / "data.csv"
    with output.write_files(dataset) as (new_dataset,):
        new_dataset.write_text("dataset")
    assert entries(target) == ["manifest.json"]
    assert dataset.read_text() == "dataset"


def test_temporary_folder_sits_in_the_target_parent(tmp_path: Path) -> None:
    target = tmp_path / "out" / "artefacts"
    with output.write_directory(target) as folder:
        (temporary,) = list(target.parent.iterdir())
        assert temporary.is_dir()
        assert temporary.name.startswith(".")
        assert folder.is_relative_to(temporary)
    dataset = tmp_path / "out" / "data.csv"
    with output.write_files(dataset) as (new_dataset,):
        temporary = next(
            path for path in dataset.parent.iterdir() if path.name != "artefacts"
        )
        assert new_dataset.is_relative_to(temporary)
        new_dataset.write_text("dataset")


def test_temporary_folders_of_two_writes_do_not_collide(tmp_path: Path) -> None:
    target = tmp_path / "artefacts"
    with (
        pytest.raises(Boom),
        output.write_directory(target) as first,
        output.write_directory(target) as second,
    ):
        assert first != second
        assert len(entries(tmp_path)) == 2
        raise Boom
    assert entries(tmp_path) == []


@pytest.mark.parametrize(
    "names",
    [
        pytest.param([], id="no file"),
        pytest.param(["a/data.csv", "b/data.report.json"], id="two folders"),
        pytest.param(["data.csv", "data.csv"], id="same file twice"),
    ],
)
def test_file_group_must_be_distinct_files_in_one_folder(
    tmp_path: Path, names: list[str]
) -> None:
    with (
        pytest.raises(ValueError, match="distinct files in one folder"),
        output.write_files(*(tmp_path / name for name in names)),
    ):
        pytest.fail("the write must not start")
    assert entries(tmp_path) == []
