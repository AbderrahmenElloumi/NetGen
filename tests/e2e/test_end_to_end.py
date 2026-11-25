"""End-to-end tests: the demo commands and the failure paths through `python -m netgen`.

Each command runs in a subprocess from the repository root, so the entry point,
the exit code and the standard streams are the real ones. Outputs go to the
temporary folder: the repository mount is read-only in the test services.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]

# The demo commands, which the README copies. `{out}` is the output folder.
CAPTURE_DEMO = (
    "build data/samples/demo_capture.pcap --out {out} --autoencoder --latents"
)
FLOWS_DEMO = "build data/samples/demo_flows.csv --out {out} --benchmark"

# The output folders of the demo commands in the `check` service.
CAPTURE_DEMO_OUT = "out/demo_capture"
FLOWS_DEMO_OUT = "out/demo_flows"

DEMO_FLOWS = "data/samples/demo_flows.csv"
EXCERPT = "data/samples/real/cicids2017_excerpt.csv"
DERIVED = ["anomaly_score", *(f"latent_{index}" for index in range(8))]
BUILD_TREE = ["artefacts", "report.json", "test.parquet", "train.parquet"]
TWO_FORMATS = ("capture (pcap or pcapng)", "CICFlowMeter CSV flow export")


def netgen(*argv: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "netgen", *(str(arg) for arg in argv)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def demo(command: str, out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return netgen(*command.format(out=out).split(), *extra)


def succeeded(completed: subprocess.CompletedProcess[str]) -> None:
    assert completed.returncode == 0, completed.stderr
    assert "Traceback" not in completed.stderr


def refused(completed: subprocess.CompletedProcess[str], *phrases: str) -> None:
    """Exit code 2, each phrase on standard error, no traceback."""
    assert completed.returncode == 2
    assert completed.stdout == ""
    for phrase in phrases:
        assert phrase in completed.stderr
    assert "Traceback" not in completed.stderr


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def names(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


def columns_with_role(manifest: Any, role: str) -> list[str]:
    return [name for name, value in manifest["columns"].items() if value == role]


def snapshot(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def capture_demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("capture") / "demo"
    succeeded(demo(CAPTURE_DEMO, out))
    return out


@pytest.fixture(scope="module")
def flows_demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("flows") / "demo"
    succeeded(demo(FLOWS_DEMO, out))
    return out


@pytest.mark.slow
def test_capture_demo_writes_the_build_tree_with_the_autoencoder_artefact(
    capture_demo: Path,
) -> None:
    assert names(capture_demo) == BUILD_TREE
    artefacts = names(capture_demo / "artefacts")
    assert {"autoencoder.pt", "manifest.json", "report.json"} <= set(artefacts)


@pytest.mark.slow
@pytest.mark.parametrize("split", ["train", "test"])
def test_capture_demo_splits_hold_identity_features_then_derived_and_no_label(
    capture_demo: Path, split: str
) -> None:
    manifest = read_json(capture_demo / "artefacts" / "manifest.json")
    identity = columns_with_role(manifest, "identity")
    features = columns_with_role(manifest, "feature")
    dataset = pd.read_parquet(capture_demo / f"{split}.parquet")
    assert identity and features
    assert list(dataset.columns) == [*identity, *features, *DERIVED]
    assert len(dataset) > 0


@pytest.mark.slow
def test_capture_demo_report_has_an_anomaly_score_without_auc_and_counts_non_ip(
    capture_demo: Path,
) -> None:
    report = read_json(capture_demo / "report.json")
    for split in ("train", "test"):
        section = report["splits"][split]
        assert section["anomaly_score"]["auc"] is None
        assert section["anomaly_score"]["median"] is not None
        assert section["counts"]["non_ip_dropped"] > 0
    assert report["benchmark"] is None


@pytest.mark.slow
def test_capture_demo_twice_with_the_same_seed_gives_equal_datasets(
    capture_demo: Path, tmp_path: Path
) -> None:
    again = tmp_path / "again"
    succeeded(demo(CAPTURE_DEMO, again))
    for split in ("train.parquet", "test.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(capture_demo / split), pd.read_parquet(again / split)
        )


def test_flows_demo_writes_labelled_splits_with_at_least_two_classes(
    flows_demo: Path,
) -> None:
    assert names(flows_demo) == BUILD_TREE
    assert "autoencoder.pt" not in names(flows_demo / "artefacts")
    for split in ("train", "test"):
        dataset = pd.read_parquet(flows_demo / f"{split}.parquet")
        assert dataset["label"].nunique() >= 2


def test_flows_demo_report_holds_one_benchmark_run_and_its_baseline(
    flows_demo: Path,
) -> None:
    benchmark = read_json(flows_demo / "report.json")["benchmark"]
    assert list(benchmark["runs"]) == ["features"]
    run = benchmark["runs"]["features"]
    assert run["model"]["macro_f1"] is not None
    assert run["baseline"]["macro_f1"] is not None


def test_fit_on_the_demo_flows_then_transform_the_excerpt(tmp_path: Path) -> None:
    artefacts = tmp_path / "artefacts"
    dataset_path = tmp_path / "excerpt" / "flows.parquet"
    succeeded(netgen("fit", DEMO_FLOWS, "--artefacts", artefacts))
    assert "report.json" in names(artefacts)
    dataset_path.parent.mkdir()
    succeeded(
        netgen("transform", EXCERPT, "--artefacts", artefacts, "--out", dataset_path)
    )

    features = columns_with_role(read_json(artefacts / "manifest.json"), "feature")
    dataset = pd.read_parquet(dataset_path)
    assert [name for name in dataset.columns if name in features] == features
    assert len(dataset) == 20
    assert names(dataset_path.parent) == ["flows.parquet", "flows.report.json"]
    report = read_json(dataset_path.parent / "flows.report.json")
    out_of_range = report["splits"]["transform"]["counts"]["out_of_range"]
    assert out_of_range
    assert all(count > 0 for count in out_of_range.values())


def test_an_existing_target_without_force_is_refused_and_left_unchanged(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "kept.txt").write_text("kept", encoding="utf-8")
    before = snapshot(tmp_path)

    refused(demo(FLOWS_DEMO, out), "already exists", "--force")
    assert snapshot(tmp_path) == before
    assert names(tmp_path) == ["out"]


def test_latents_without_the_autoencoder_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    refused(netgen("build", DEMO_FLOWS, "--out", out, "--latents"), "--autoencoder")
    assert names(tmp_path) == []


def test_an_unknown_input_is_refused_naming_the_two_supported_formats(
    tmp_path: Path,
) -> None:
    unknown = tmp_path / "notes.txt"
    unknown.write_text("not a capture, not a flow export\n", encoding="utf-8")
    out = tmp_path / "out"
    refused(netgen("build", unknown, "--out", out), *TWO_FORMATS)
    assert names(tmp_path) == ["notes.txt"]


def test_a_missing_input_is_refused(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pcap"
    refused(netgen("build", missing, "--out", tmp_path / "out"), "no such file")
    assert names(tmp_path) == []


def test_the_benchmark_on_the_unlabelled_capture_is_refused_before_writing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    command = CAPTURE_DEMO.replace(" --autoencoder --latents", " --benchmark")
    refused(demo(command, out), "labels")
    assert names(tmp_path) == []


def test_the_check_service_runs_the_static_checks_then_both_demo_commands() -> None:
    """The steps run in order, joined by `&&`: any failing step stops `check`."""
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    steps = [
        "ruff check",
        "ruff format --check",
        "mypy",
        "python -m netgen " + CAPTURE_DEMO.format(out=CAPTURE_DEMO_OUT) + " --force",
        "python -m netgen " + FLOWS_DEMO.format(out=FLOWS_DEMO_OUT) + " --force",
    ]
    check = compose[compose.index("\n  check:") : compose.index("\n  dev:")]
    folded = check.split("- >-", 1)[1]
    assert " ".join(folded.split()) == " && ".join(steps)


def test_the_output_folder_is_tracked_so_a_fresh_clone_has_it() -> None:
    if not (ROOT / ".git").exists():
        pytest.skip("not a git clone")
    tracked = subprocess.run(
        ["git", "-c", "safe.directory=*", "ls-files", "--", "out"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    ).stdout.split()
    assert tracked == ["out/.gitkeep"]
