"""Tests of the `build` command (netgen.cli): split, fit on the training split, transform both splits."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from netgen import cli
from netgen.cli import build_parser, main
from netgen.preparation import Artefacts, Settings, fit_preparation
from netgen.readers import reader_named
from netgen.record import FlowSettings
from netgen.stages import autoencoder
from netgen.table import FlowTable, stratified_split

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO = SAMPLES / "demo_flows.csv"
EXCERPT = SAMPLES / "real" / "cicids2017_excerpt.csv"
CAPTURE = SAMPLES / "real" / "pcapng-example.pcapng"
QUICK_AUTOENCODER = ["--autoencoder", "--epochs", "1", "--latent-dim", "2"]
ARTEFACT_FILES = ["fill.json", "manifest.json", "report.json", "scaler.json"]
TRAIN_LABELS = {"BENIGN": 240, "DDoS": 48, "PortScan": 32}
TEST_LABELS = {"BENIGN": 60, "DDoS": 12, "PortScan": 8}


def run(*argv: str | Path) -> int:
    """`main`'s exit code, also when argparse exits on a usage error."""
    try:
        return main([str(arg) for arg in argv])
    except SystemExit as exit:
        assert isinstance(exit.code, int)
        return exit.code


def built(out: Path, *options: str, source: Path = DEMO) -> Path:
    assert run("build", source, "--out", out, *options) == 0
    return out


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def read_table(path: Path) -> FlowTable:
    return (
        reader_named("flow-export").read(path, FlowSettings(), label_col="Label").table
    )


def demo_lines() -> tuple[str, list[str]]:
    header, *rows = DEMO.read_text(encoding="utf-8").splitlines()
    return header, rows


def write_flows(path: Path, header: str, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def refused(
    capsys: pytest.CaptureFixture[str], code: int, out: Path, *phrases: str
) -> None:
    """Exit code 2, each phrase on standard error, no traceback, and nothing written."""
    captured = capsys.readouterr()
    assert code == 2
    for phrase in phrases:
        assert phrase in captured.err
    assert "Traceback" not in captured.err
    assert not out.exists()
    assert not out.parent.exists() or names(out.parent) == []


def test_d01_nothing_is_fitted_on_test_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D01: the split comes before every fit, so test rows never reach a learning stage."""
    header, rows = demo_lines()
    table = read_table(DEMO)
    assert table.to_frame().index.equals(pd.RangeIndex(len(rows)))
    _, test = stratified_split(table, seed=42)
    test_rows = set(test.to_frame().index)
    column = [name.strip() for name in header.split(",")].index("Flow Duration")
    extreme = 10**15  # microseconds in the input, seconds after the reader
    for row in test_rows:
        fields = rows[row].split(",")
        fields[column] = str(extreme)
        rows[row] = ",".join(fields)
    source = write_flows(tmp_path / "in" / "flows.csv", header, rows)
    train, test = stratified_split(read_table(source), seed=42)
    assert set(test.to_frame().index) == test_rows
    assert (test.to_frame()["duration"] == extreme / 1e6).all()
    fitted_tables: list[pd.DataFrame] = []

    def spy(table: FlowTable, settings: Settings) -> Artefacts:
        fitted_tables.append(table.to_frame())
        return fit_preparation(table, settings)

    monkeypatch.setattr(cli, "fit_preparation", spy)

    out = built(tmp_path / "out", source=source)

    (fitted_table,) = fitted_tables
    assert set(fitted_table.index).isdisjoint(test_rows)
    assert len(fitted_table) == len(rows) - len(test_rows)
    assert fitted_table["duration"].max() < extreme / 1e6
    reference = tmp_path / "reference"
    reference.mkdir()
    fit_preparation(train, Settings(fingerprint="", label_column="Label")).save(
        reference
    )
    for name in ["fill.json", "scaler.json"]:
        assert read_json(out / "artefacts" / name) == read_json(reference / name)
    assert read_json(out / "artefacts" / "fill.json")["duration"]["max"] < extreme / 1e6


def test_d02_both_splits_take_the_same_transform_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D02: one `transform` makes both datasets; no output comes from training mode."""
    transformed: list[int] = []
    transform = Artefacts.transform

    def spy(self: Artefacts, table: FlowTable) -> Any:
        transformed.append(len(table.to_frame()))
        return transform(self, table)

    monkeypatch.setattr(Artefacts, "transform", spy)

    out = built(tmp_path / "out", *QUICK_AUTOENCODER, "--latents")

    assert transformed == [320, 80]
    monkeypatch.undo()
    train, test = stratified_split(read_table(DEMO), seed=42)
    loaded = Artefacts.load(out / "artefacts")
    for name, rows in [("train", train), ("test", test)]:
        # Read back through Parquet too, so both frames get the same dtypes.
        reference = tmp_path / f"reference_{name}.parquet"
        loaded.transform(rows).table.to_frame().to_parquet(reference, index=False)
        dataset = pd.read_parquet(out / f"{name}.parquet")
        assert {"anomaly_score", "latent_0", "latent_1"} <= set(dataset.columns)
        pd.testing.assert_frame_equal(dataset, pd.read_parquet(reference))


@pytest.mark.parametrize("output_format", ["parquet", "csv"])
def test_build_writes_the_artefacts_the_two_splits_and_the_report(
    tmp_path: Path, output_format: str, capsys: pytest.CaptureFixture[str]
) -> None:
    out = built(tmp_path / "out", "--output-format", output_format)

    assert names(tmp_path) == ["out"]
    assert names(out) == sorted(
        ["artefacts", "report.json", f"test.{output_format}", f"train.{output_format}"]
    )
    assert names(out / "artefacts") == ARTEFACT_FILES
    read = pd.read_csv if output_format == "csv" else pd.read_parquet
    assert len(read(out / f"train.{output_format}")) == 320
    assert len(read(out / f"test.{output_format}")) == 80
    printed = capsys.readouterr().out
    assert "train: 320 rows" in printed
    assert "test: 80 rows" in printed
    assert str(out) in printed


def test_the_split_is_stratified_and_visible_in_the_report(tmp_path: Path) -> None:
    splits = read_json(built(tmp_path / "out") / "report.json")["splits"]

    assert list(splits) == ["train", "test"]
    assert splits["train"]["labels"] == TRAIN_LABELS
    assert splits["test"]["labels"] == TEST_LABELS


def test_a_capture_builds_unlabelled_splits(tmp_path: Path) -> None:
    report = read_json(built(tmp_path / "out", source=CAPTURE) / "report.json")

    assert report["header"]["start_time_zone"] == "utc"
    for entry in report["splits"].values():
        assert entry["labels"] == {}
        assert entry["rows"] > 0


def test_the_artefacts_report_holds_the_training_split_only(tmp_path: Path) -> None:
    out = built(tmp_path / "out")
    inside = read_json(out / "artefacts" / "report.json")
    outside = read_json(out / "report.json")

    for report in (inside, outside):
        assert report["header"]["command"] == "build"
        assert report["header"]["seed"] == 42
        assert report["header"]["start_time_zone"] == "local"
        assert report["benchmark"] is None
    assert list(inside["splits"]) == ["train"]
    assert inside["splits"]["train"] == outside["splits"]["train"]
    assert inside["header"] == outside["header"]


def test_the_same_seed_gives_identical_datasets_and_artefacts(tmp_path: Path) -> None:
    first = built(tmp_path / "first", *QUICK_AUTOENCODER, "--seed", "3")
    second = built(tmp_path / "second", *QUICK_AUTOENCODER, "--seed", "3")

    for name in ["train.parquet", "test.parquet"]:
        pd.testing.assert_frame_equal(
            pd.read_parquet(first / name), pd.read_parquet(second / name)
        )
    assert names(first / "artefacts") == names(second / "artefacts")
    for path in (first / "artefacts").iterdir():
        assert path.read_bytes() == (second / "artefacts" / path.name).read_bytes()


def test_fit_on_label_restricts_the_autoencoder_rows_not_the_scaler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    header, rows = demo_lines()
    source = write_flows(tmp_path / "in" / "doubled.csv", header, rows + rows)
    trained: list[int] = []
    train_network = autoencoder.train

    def spy(matrix: np.ndarray, settings: autoencoder.Settings) -> Any:
        trained.append(len(matrix))
        return train_network(matrix, settings)

    monkeypatch.setattr(autoencoder, "train", spy)

    out = built(
        tmp_path / "out", *QUICK_AUTOENCODER, "--fit-on-label", "BENIGN", source=source
    )

    assert trained == [480]
    train, _ = stratified_split(read_table(source), seed=42)
    reference = tmp_path / "reference"
    reference.mkdir()
    fit_preparation(train, Settings(fingerprint="", label_column="Label")).save(
        reference
    )
    assert read_json(out / "artefacts" / "scaler.json") == read_json(
        reference / "scaler.json"
    )
    splits = read_json(out / "report.json")["splits"]
    assert splits["train"]["rows"] == 640
    for entry in splits.values():
        assert entry["anomaly_score"]["auc"] is not None


@pytest.mark.parametrize(
    ("options", "runs"),
    [([], ["features"]), (QUICK_AUTOENCODER, ["features", "features_plus_derived"])],
)
def test_benchmark_scores_the_test_split_in_the_report(
    tmp_path: Path,
    options: list[str],
    runs: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = built(tmp_path / "out", "--benchmark", *options)

    benchmark = read_json(out / "report.json")["benchmark"]
    assert list(benchmark["runs"]) == runs
    assert benchmark["settings"]["seed"] == 42
    assert benchmark["unlabelled_skipped"] == {"train": 0, "test": 0}
    for entry in benchmark["runs"].values():
        support = sum(
            scores["support"] for scores in entry["model"]["per_class"].values()
        )
        assert support == 80
    assert read_json(out / "artefacts" / "report.json")["benchmark"] is None
    printed = capsys.readouterr().out
    for name in runs:
        assert f"benchmark {name}: macro F1" in printed
    assert "baseline macro F1" in printed


def test_the_build_options_and_their_defaults() -> None:
    args = vars(build_parser().parse_args(["build", "in.csv", "--out", "dir"]))

    del args["run"]
    assert args == {
        "command": "build",
        "input": Path("in.csv"),
        "out": Path("dir"),
        "force": False,
        "input_format": None,
        "label_col": "Label",
        "idle_timeout": 120.0,
        "active_timeout": None,
        "dst_port_as_feature": False,
        "seed": 42,
        "autoencoder": False,
        "latents": False,
        "latent_dim": 8,
        "epochs": 50,
        "fit_on_label": None,
        "output_format": "parquet",
        "benchmark": False,
    }


def test_build_has_no_artefacts_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "dir"
    code = run("build", DEMO, "--out", out, "--artefacts", tmp_path / "a")
    refused(capsys, code, out, "unrecognized arguments", "--artefacts")


def test_benchmark_on_an_unlabelled_input_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "dir"
    code = run("build", CAPTURE, "--out", out, "--benchmark")
    refused(capsys, code, out, "needs labels", "leave out --benchmark")


def test_benchmark_with_one_class_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header, rows = demo_lines()
    benign = [row for row in rows if row.endswith(",BENIGN")]
    source = write_flows(tmp_path / "in" / "benign.csv", header, benign)
    out = tmp_path / "out" / "dir"

    code = run("build", source, "--out", out, "--benchmark")

    refused(capsys, code, out, "at least 2 classes", "leave out --benchmark")


def test_latents_without_the_autoencoder_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "dir"
    code = run("build", DEMO, "--out", out, "--latents")
    refused(capsys, code, out, "--latents requires --autoencoder")


def test_fit_on_label_without_the_autoencoder_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "dir"
    code = run("build", DEMO, "--out", out, "--fit-on-label", "BENIGN")
    refused(
        capsys,
        code,
        out,
        "--fit-on-label requires --autoencoder. Add --autoencoder, or drop --fit-on-label.",
    )


def test_the_autoencoder_with_too_few_training_flows_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "dir"
    code = run("build", EXCERPT, "--out", out, *QUICK_AUTOENCODER)
    refused(capsys, code, out, "256")


def test_fit_on_label_matching_no_row_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out" / "dir"
    code = run(
        "build", DEMO, "--out", out, *QUICK_AUTOENCODER, "--fit-on-label", "NOPE"
    )
    refused(capsys, code, out, "NOPE")


def test_an_existing_folder_is_refused_before_reading_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "old.txt").write_text("old", encoding="utf-8")

    code = run("build", tmp_path / "missing.csv", "--out", out)

    captured = capsys.readouterr()
    assert code == 2
    assert "already exists. Pass --force to replace it." in captured.err
    assert names(out) == ["old.txt"]
    built(out, "--force")
    assert "old.txt" not in names(out)
    assert "train.parquet" in names(out)


def test_an_input_of_one_flow_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    header, rows = demo_lines()
    source = write_flows(tmp_path / "in" / "one.csv", header, rows[:1])
    out = tmp_path / "out" / "dir"

    code = run("build", source, "--out", out)

    refused(capsys, code, out, "test split", "at least 2 flows")
