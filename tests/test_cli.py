"""Tests of the command line (netgen.cli): the fit and transform commands, their options and exit codes."""

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from netgen import cli, record
from netgen.cli import build_parser, main
from netgen.errors import NetGenError
from netgen.record import Role

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO = SAMPLES / "demo_flows.csv"
EXCERPT = SAMPLES / "real" / "cicids2017_excerpt.csv"
CAPTURE = SAMPLES / "real" / "pcapng-example.pcapng"
DEMO_LABELS = {"BENIGN": 300, "DDoS": 60, "PortScan": 40}
QUICK_AUTOENCODER = ["--autoencoder", "--epochs", "1", "--latent-dim", "2"]
IDENTITY = [field.name for field in record.fields_by_role(Role.IDENTITY)]
FEATURES = [field.name for field in record.fields_by_role(Role.FEATURE)]


def run(*argv: str | Path) -> int:
    """`main`'s exit code, also when argparse exits on a usage error."""
    try:
        return main([str(arg) for arg in argv])
    except SystemExit as exit:
        assert isinstance(exit.code, int)
        return exit.code


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def fitted(folder: Path, *options: str, source: Path = DEMO) -> Path:
    artefacts = folder / "artefacts"
    assert run("fit", source, "--artefacts", artefacts, *options) == 0
    return artefacts


def names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def refused(capsys: pytest.CaptureFixture[str], code: int, *phrases: str) -> None:
    """Exit code 2, each phrase on standard error, and no traceback."""
    captured = capsys.readouterr()
    assert code == 2
    for phrase in phrases:
        assert phrase in captured.err
    assert "Traceback" not in captured.err


def test_importing_configures_no_logging_and_no_warning_filter_until_main_runs() -> (
    None
):
    """D09: logging and warning filters are configured in `main` only."""
    script = textwrap.dedent(
        """
        import json, logging, warnings

        # numpy adds its own warning filters when it is imported; they are
        # not NetGen's, so they are part of the state before.
        import numpy

        def state():
            root = logging.getLogger()
            return {
                "handlers": [repr(handler) for handler in root.handlers],
                "level": root.level,
                "filters": [repr(entry) for entry in warnings.filters],
            }

        before = state()
        import netgen
        import netgen.cli
        import netgen.readers.capture
        imported = state()
        netgen.cli.main(["fit", "missing.pcap", "--artefacts", "artefacts"])
        after = state()
        print(json.dumps({"before": before, "imported": imported, "after": after}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd="/tmp",
    )
    states = json.loads(completed.stdout.splitlines()[-1])
    assert states["imported"] == states["before"]
    assert states["after"]["handlers"] != states["before"]["handlers"]
    assert states["after"]["filters"] != states["before"]["filters"]


def test_fit_writes_the_artefacts_and_their_report_and_no_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefacts = fitted(tmp_path)

    assert names(artefacts) == [
        "fill.json",
        "manifest.json",
        "report.json",
        "scaler.json",
    ]
    assert names(tmp_path) == ["artefacts"]
    header = read_json(artefacts / "report.json")["header"]
    assert header["command"] == "fit"
    assert header["seed"] == 42
    assert header["start_time_zone"] == "local"
    assert header["input_fingerprint"] == hashlib.sha256(DEMO.read_bytes()).hexdigest()
    splits = read_json(artefacts / "report.json")["splits"]
    assert list(splits) == ["train"]
    assert splits["train"]["rows"] == 400
    assert splits["train"]["labels"] == DEMO_LABELS
    assert splits["train"]["counts"]["encoding"] == "utf-8"
    assert splits["train"]["anomaly_score"] is None
    out = capsys.readouterr().out
    assert "train: 400 rows" in out
    assert str(artefacts) in out


def test_fit_with_the_autoencoder_saves_its_weights_and_records_its_settings(
    tmp_path: Path,
) -> None:
    artefacts = fitted(
        tmp_path, *QUICK_AUTOENCODER, "--fit-on-label", "BENIGN", "--seed", "7"
    )

    assert (artefacts / "autoencoder.pt").is_file()
    manifest = read_json(artefacts / "manifest.json")
    assert manifest["seed"] == 7
    assert manifest["autoencoder"]["fit_on_label"] == "BENIGN"
    assert manifest["autoencoder"]["latent_dim"] == 2
    assert manifest["autoencoder"]["epochs"] == 1
    assert manifest["autoencoder"]["latents"] is False
    score = read_json(artefacts / "report.json")["splits"]["train"]["anomaly_score"]
    assert score["auc"] is not None
    assert set(score["median_by_label"]) == set(DEMO_LABELS)


def test_transform_writes_the_dataset_and_its_report_beside_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefacts = fitted(
        tmp_path, *QUICK_AUTOENCODER, "--latents", "--fit-on-label", "BENIGN"
    )
    out = tmp_path / "flows.parquet"

    assert run("transform", DEMO, "--artefacts", artefacts, "--out", out) == 0

    assert names(tmp_path) == ["artefacts", "flows.parquet", "flows.report.json"]
    dataset = pd.read_parquet(out)
    assert list(dataset.columns) == [
        *IDENTITY,
        "label",
        *FEATURES,
        "anomaly_score",
        "latent_0",
        "latent_1",
    ]
    assert len(dataset) == 400
    written = read_json(tmp_path / "flows.report.json")
    assert written["header"]["command"] == "transform"
    assert list(written["splits"]) == ["transform"]
    assert written["splits"]["transform"]["rows"] == 400
    assert written["splits"]["transform"]["anomaly_score"]["auc"] is not None
    assert "transform: 400 rows" in capsys.readouterr().out


def test_the_output_format_option_decides_the_format_never_the_file_name(
    tmp_path: Path,
) -> None:
    artefacts = fitted(tmp_path)
    as_csv = tmp_path / "csv.parquet"
    as_parquet = tmp_path / "parquet.csv"

    code = run(
        "transform",
        DEMO,
        "--artefacts",
        artefacts,
        "--out",
        as_csv,
        "--output-format",
        "csv",
    )
    assert code == 0
    assert run("transform", DEMO, "--artefacts", artefacts, "--out", as_parquet) == 0

    from_csv = pd.read_csv(as_csv)
    from_parquet = pd.read_parquet(as_parquet)
    assert list(from_csv.columns) == [*IDENTITY, "label", *FEATURES]
    assert list(from_parquet.columns) == [*IDENTITY, "label", *FEATURES]
    assert len(from_csv) == len(from_parquet) == 400


def test_transform_takes_the_flow_settings_and_roles_from_the_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "default").mkdir()
    default = fitted(tmp_path / "default", source=CAPTURE)
    artefacts = fitted(
        tmp_path, "--idle-timeout", "0.001", "--dst-port-as-feature", source=CAPTURE
    )
    out = tmp_path / "flows.parquet"

    assert run("transform", CAPTURE, "--artefacts", artefacts, "--out", out) == 0

    fit_rows = read_json(artefacts / "report.json")["splits"]["train"]["rows"]
    default_rows = read_json(default / "report.json")["splits"]["train"]["rows"]
    written = read_json(tmp_path / "flows.report.json")
    assert fit_rows != default_rows
    assert written["splits"]["transform"]["rows"] == fit_rows
    assert written["header"]["start_time_zone"] == "utc"
    assert "dst_port" in written["splits"]["transform"]["features"]
    identity = [name for name in IDENTITY if name != "dst_port"]
    assert list(pd.read_parquet(out).columns[: len(identity)]) == identity


def test_d05_label_column_named_at_fit_is_recorded_and_reused_by_transform(
    tmp_path: Path,
) -> None:
    header, rest = DEMO.read_text(encoding="utf-8").split("\n", 1)
    source = tmp_path / "renamed.csv"
    source.write_text(
        header.replace(" Label", " Attack") + "\n" + rest, encoding="utf-8"
    )
    artefacts = fitted(tmp_path, "--label-col", "Attack", source=source)
    out = tmp_path / "flows.parquet"

    manifest = read_json(artefacts / "manifest.json")
    assert manifest["label_column"] == "Attack"
    assert manifest["columns"]["label"] == "label"
    splits = read_json(artefacts / "report.json")["splits"]
    assert splits["train"]["labels"] == DEMO_LABELS
    assert run("transform", source, "--artefacts", artefacts, "--out", out) == 0
    assert pd.read_parquet(out)["label"].value_counts().to_dict() == DEMO_LABELS


def test_transform_reads_the_default_label_column_when_the_manifest_names_none(
    tmp_path: Path,
) -> None:
    artefacts = fitted(tmp_path)
    manifest = read_json(artefacts / "manifest.json")
    write_json(artefacts / "manifest.json", {**manifest, "label_column": None})
    out = tmp_path / "flows.parquet"

    assert run("transform", DEMO, "--artefacts", artefacts, "--out", out) == 0

    assert pd.read_parquet(out)["label"].value_counts().to_dict() == DEMO_LABELS


def test_input_format_selects_the_reader_without_detection(tmp_path: Path) -> None:
    source = tmp_path / "flows.data"
    source.write_bytes(DEMO.read_bytes())
    artefacts = fitted(tmp_path, "--input-format", "flow-export", source=source)
    out = tmp_path / "flows.parquet"

    code = run(
        "transform",
        source,
        "--artefacts",
        artefacts,
        "--out",
        out,
        "--input-format",
        "flow-export",
    )
    assert code == 0
    assert len(pd.read_parquet(out)) == 400


def test_the_options_and_their_defaults() -> None:
    parser = build_parser()
    fit = vars(parser.parse_args(["fit", "in.csv", "--artefacts", "dir"]))
    transform = vars(
        parser.parse_args(["transform", "in.csv", "--artefacts", "dir", "--out", "f"])
    )

    del fit["run"], transform["run"]
    assert fit == {
        "command": "fit",
        "input": Path("in.csv"),
        "artefacts": Path("dir"),
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
    }
    assert transform == {
        "command": "transform",
        "input": Path("in.csv"),
        "artefacts": Path("dir"),
        "out": Path("f"),
        "force": False,
        "input_format": None,
        "output_format": "parquet",
    }


@pytest.mark.parametrize(
    "options",
    [
        ["--idle-timeout", "60"],
        ["--active-timeout", "60"],
        ["--label-col", "Attack"],
        ["--dst-port-as-feature"],
        ["--seed", "1"],
        ["--autoencoder"],
        ["--latents"],
        ["--benchmark"],
    ],
)
def test_transform_has_no_fit_option_and_no_benchmark(
    tmp_path: Path, options: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "flows.parquet"
    code = run("transform", DEMO, "--artefacts", tmp_path, "--out", out, *options)
    refused(capsys, code, "unrecognized arguments", options[0])
    assert names(tmp_path) == []


@pytest.mark.parametrize(
    "options",
    [["--out", "flows.parquet"], ["--benchmark"], ["--output-format", "csv"]],
)
def test_fit_has_no_out_benchmark_or_output_format_option(
    tmp_path: Path, options: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code = run("fit", DEMO, "--artefacts", tmp_path / "a", *options)
    refused(capsys, code, "unrecognized arguments", options[0])
    assert names(tmp_path) == []


@pytest.mark.parametrize(
    "options",
    [
        ["--idle-timeout", "0"],
        ["--idle-timeout", "-1"],
        ["--idle-timeout", "nan"],
        ["--idle-timeout", "soon"],
        ["--active-timeout", "0"],
        ["--latent-dim", "0"],
        ["--epochs", "0"],
        ["--epochs", "2.5"],
        ["--seed", "1.5"],
    ],
)
def test_numeric_options_are_checked_when_parsed(
    tmp_path: Path, options: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    code = run("fit", DEMO, "--artefacts", tmp_path / "a", *options)
    refused(capsys, code, f"argument {options[0]}")
    assert names(tmp_path) == []


def test_a_user_error_exits_2_with_its_message_on_standard_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run("fit", tmp_path / "missing.csv", "--artefacts", tmp_path / "a")

    refused(capsys, code, "missing.csv: no such file. Check the input path.")
    assert names(tmp_path) == []


def test_a_user_error_prints_nothing_on_standard_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run("fit", tmp_path / "missing.csv", "--artefacts", tmp_path / "a")

    assert capsys.readouterr().out == ""


def test_a_user_error_prints_the_notes_added_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def refusing(*args: object, **kwargs: object) -> None:
        error = NetGenError("the cause. The fix.")
        error.add_note("The previous output is kept in a folder.")
        raise error

    monkeypatch.setattr(cli, "fit_preparation", refusing)

    code = run("fit", DEMO, "--artefacts", tmp_path / "a")

    refused(
        capsys,
        code,
        "netgen: error: the cause. The fix.\nThe previous output is kept in a folder.",
    )


def test_an_unexpected_exception_keeps_its_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("a bug")

    monkeypatch.setattr(cli, "fit_preparation", broken)

    with pytest.raises(RuntimeError, match="a bug"):
        main(["fit", str(DEMO), "--artefacts", str(tmp_path / "a")])
    assert names(tmp_path) == []


def test_latents_without_the_autoencoder_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run("fit", DEMO, "--artefacts", tmp_path / "a", "--latents")

    refused(capsys, code, "--latents requires --autoencoder", "Add --autoencoder")
    assert names(tmp_path) == []


def test_fit_on_label_without_the_autoencoder_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run("fit", DEMO, "--artefacts", tmp_path / "a", "--fit-on-label", "BENIGN")

    refused(
        capsys,
        code,
        "--fit-on-label requires --autoencoder. Add --autoencoder, or drop --fit-on-label.",
    )
    assert names(tmp_path) == []


def test_fit_refuses_an_existing_artefacts_folder_before_reading_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefacts = fitted(tmp_path)
    capsys.readouterr()

    code = run("fit", tmp_path / "missing.csv", "--artefacts", artefacts)
    refused(capsys, code, f"{artefacts} already exists", "--force")
    assert read_json(artefacts / "manifest.json")["seed"] == 42

    assert run("fit", DEMO, "--artefacts", artefacts, "--seed", "7", "--force") == 0
    assert read_json(artefacts / "manifest.json")["seed"] == 7
    assert names(tmp_path) == ["artefacts"]


@pytest.mark.parametrize("existing", ["flows.parquet", "flows.report.json"])
def test_transform_refuses_an_existing_dataset_or_report_before_reading_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], existing: str
) -> None:
    artefacts = fitted(tmp_path)
    out = tmp_path / "flows.parquet"
    (tmp_path / existing).write_text("old", encoding="utf-8")
    capsys.readouterr()

    code = run(
        "transform",
        tmp_path / "missing.csv",
        "--artefacts",
        tmp_path / "none",
        "--out",
        out,
    )
    refused(capsys, code, f"{tmp_path / existing} already exists", "--force")
    assert (tmp_path / existing).read_text(encoding="utf-8") == "old"

    code = run("transform", DEMO, "--artefacts", artefacts, "--out", out, "--force")
    assert code == 0
    assert len(pd.read_parquet(out)) == 400
    assert read_json(tmp_path / "flows.report.json")["header"]["command"] == (
        "transform"
    )


def test_an_unknown_input_is_refused_naming_the_two_formats(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "notes.txt"
    source.write_text("not traffic\n", encoding="utf-8")

    code = run("fit", source, "--artefacts", tmp_path / "a")

    refused(
        capsys, code, "capture (pcap or pcapng)", "CICFlowMeter CSV", "--input-format"
    )
    assert names(tmp_path) == ["notes.txt"]


def test_fit_refuses_an_unknown_input_format_name_listing_the_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run("fit", DEMO, "--artefacts", tmp_path / "a", "--input-format", "zeek")

    refused(capsys, code, "'zeek'", "Valid names: capture, flow-export")
    assert names(tmp_path) == []


def test_transform_refuses_an_unknown_input_format_name_listing_the_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefacts = fitted(tmp_path)
    capsys.readouterr()

    code = run(
        "transform",
        DEMO,
        "--artefacts",
        artefacts,
        "--out",
        tmp_path / "f.parquet",
        "--input-format",
        "zeek",
    )

    refused(capsys, code, "'zeek'", "Valid names: capture, flow-export")
    assert names(tmp_path) == ["artefacts"]


def test_transform_refuses_an_input_lacking_a_needed_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefacts = fitted(tmp_path)
    source = tmp_path / "lacking.csv"
    pd.read_csv(DEMO).drop(columns=[" Flow IAT Std"]).to_csv(source, index=False)
    capsys.readouterr()

    code = run(
        "transform",
        source,
        "--artefacts",
        artefacts,
        "--out",
        tmp_path / "f.parquet",
    )

    refused(capsys, code, "lacks the columns", "Flow IAT Std")
    assert names(tmp_path) == ["artefacts", "lacking.csv"]


def test_transform_refuses_an_incompatible_format_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefacts = fitted(tmp_path)
    manifest = read_json(artefacts / "manifest.json")
    write_json(artefacts / "manifest.json", {**manifest, "format_version": 2})
    capsys.readouterr()

    code = run(
        "transform", DEMO, "--artefacts", artefacts, "--out", tmp_path / "f.parquet"
    )

    refused(capsys, code, "format", "netgen fit")
    assert names(tmp_path) == ["artefacts"]


def test_transform_refuses_feature_columns_that_differ_from_the_scaler(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    artefacts = fitted(tmp_path)
    scaler = read_json(artefacts / "scaler.json")
    del scaler["duration"]
    write_json(artefacts / "scaler.json", scaler)
    capsys.readouterr()

    code = run(
        "transform", DEMO, "--artefacts", artefacts, "--out", tmp_path / "f.parquet"
    )

    refused(capsys, code, "scaler.json", "duration")
    assert names(tmp_path) == ["artefacts"]


def test_transform_refuses_a_missing_artefacts_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "none"

    code = run(
        "transform", DEMO, "--artefacts", missing, "--out", tmp_path / "f.parquet"
    )

    refused(capsys, code, str(missing), "netgen fit")
    assert names(tmp_path) == []


def test_the_autoencoder_with_too_few_rows_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run("fit", EXCERPT, "--artefacts", tmp_path / "a", *QUICK_AUTOENCODER)

    refused(capsys, code, "256")
    assert names(tmp_path) == []


def test_python_dash_m_netgen_runs_the_commands(tmp_path: Path) -> None:
    command = [sys.executable, "-m", "netgen", "fit", str(DEMO)]
    artefacts = ["--artefacts", str(tmp_path / "artefacts")]

    first = subprocess.run(
        [*command, *artefacts], capture_output=True, text=True, check=False
    )
    second = subprocess.run(
        [*command, *artefacts], capture_output=True, text=True, check=False
    )

    assert first.returncode == 0, first.stderr
    assert (tmp_path / "artefacts" / "report.json").is_file()
    assert second.returncode == 2
    assert "already exists. Pass --force to replace it." in second.stderr
