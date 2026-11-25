"""Tests of the benchmark (netgen.benchmark): the preflight, the runs, the metrics and the baseline."""

import importlib
import json
import sys
from typing import Any

import pandas as pd
import pytest
import sklearn.ensemble

from netgen import benchmark, record, report
from netgen.errors import BenchmarkUnavailable, NetGenError
from netgen.record import Role
from netgen.table import FlowTable

SETTINGS = {"trees": 100, "class_weight": "balanced", "seed": 7, "n_jobs": 2}
METRIC_KEYS = {"macro_f1", "balanced_accuracy", "per_class", "confusion_matrix"}
CLASS_KEYS = {"precision", "recall", "f1", "support"}


def table(
    labels: list[str | None] | None,
    *,
    duration: list[float] | None = None,
    scores: list[float] | None = None,
) -> FlowTable:
    """A small prepared table: `src_ip` (identity), `duration` and `payload_bytes` (features),
    an optional `label` and an optional `anomaly_score` (derived).

    Without `duration`, a label's rows sit at its own value (`A` near 0, `B` near 10,
    `C` near 20), so the classes are clearly separable.
    """
    rows = len(labels) if labels is not None else len(duration or [])
    if duration is None:
        assert labels is not None
        centre = {"A": 0.0, "B": 10.0, "C": 20.0}
        duration = [
            centre.get(label or "A", 0.0) + 0.1 * (i % 5)
            for i, label in enumerate(labels)
        ]
    columns: dict[str, Any] = {
        "src_ip": pd.Series([f"192.0.2.{i}" for i in range(rows)], dtype="string"),
        "duration": pd.Series(duration, dtype=float),
        "payload_bytes": pd.Series([2.0 * value for value in duration], dtype=float),
    }
    roles = {
        "src_ip": Role.IDENTITY,
        "duration": Role.FEATURE,
        "payload_bytes": Role.FEATURE,
    }
    if labels is not None:
        columns["label"] = pd.Series(labels, dtype="string")
        roles["label"] = Role.LABEL
    flows = FlowTable(pd.DataFrame(columns), roles)
    if scores is not None:
        flows = flows.with_derived(record.ANOMALY_SCORE, scores)
    return flows


def separable(rows: int = 20, *, scores: bool = False) -> FlowTable:
    labels: list[str | None] = ["A" if i % 2 else "B" for i in range(rows)]
    anomaly = [1.0 if label == "B" else 0.0 for label in labels] if scores else None
    return table(labels, scores=anomaly)


# The runs and the layout (D50).


def test_a_separable_table_scores_well_above_the_baseline() -> None:
    result = benchmark.run(separable(), separable(10), seed=7)

    assert set(result) == {"settings", "unlabelled_skipped", "runs"}
    assert set(result["runs"]) == {"features"}
    run = result["runs"]["features"]
    assert set(run) == {"columns", "model", "baseline"}
    assert set(run["model"]) == METRIC_KEYS
    assert set(run["baseline"]) == METRIC_KEYS
    assert run["model"]["macro_f1"] == 1.0
    assert run["model"]["balanced_accuracy"] == 1.0
    assert run["baseline"]["macro_f1"] < 0.5
    assert run["baseline"]["balanced_accuracy"] == 0.5


def test_the_result_is_fixed_by_the_seed() -> None:
    train = table(["A", "B"] * 10, duration=[float(i % 7) for i in range(20)])
    test = table(["A", "B"] * 5, duration=[float(i % 3) for i in range(10)])

    assert benchmark.run(train, test, seed=3) == benchmark.run(train, test, seed=3)


def test_the_settings_are_recorded_and_given_to_the_forest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params: list[dict[str, Any]] = []
    fit = sklearn.ensemble.RandomForestClassifier.fit

    def spy(self: Any, *args: Any, **kwargs: Any) -> Any:
        params.append(self.get_params())
        return fit(self, *args, **kwargs)

    monkeypatch.setattr(sklearn.ensemble.RandomForestClassifier, "fit", spy)

    result = benchmark.run(separable(), separable(10), seed=7)

    assert result["settings"] == SETTINGS
    assert len(params) == 1
    assert {
        key: params[0][key]
        for key in ("n_estimators", "class_weight", "random_state", "n_jobs")
    } == {
        "n_estimators": 100,
        "class_weight": "balanced",
        "random_state": 7,
        "n_jobs": 2,
    }


def test_without_derived_columns_one_run_uses_the_feature_columns_only() -> None:
    result = benchmark.run(separable(), separable(10), seed=7)

    assert list(result["runs"]) == ["features"]
    assert result["runs"]["features"]["columns"] == ["duration", "payload_bytes"]


def test_with_an_anomaly_score_a_second_run_adds_the_derived_column() -> None:
    result = benchmark.run(separable(scores=True), separable(10, scores=True), seed=7)

    assert list(result["runs"]) == ["features", "features_plus_derived"]
    assert result["runs"]["features"]["columns"] == ["duration", "payload_bytes"]
    second = result["runs"]["features_plus_derived"]
    assert second["columns"] == ["duration", "payload_bytes", record.ANOMALY_SCORE]
    assert set(second) == {"columns", "model", "baseline"}


def test_the_baseline_matches_hand_worked_values() -> None:
    # Majority training label: A (2 of 3). Every test row is predicted A.
    # A: precision 1/3, recall 1, F1 1/2; B and C: 0. Macro F1 1/6.
    # Balanced accuracy: mean recall over the classes with support, (1 + 0 + 0) / 3.
    result = benchmark.run(table(["A", "A", "B"]), table(["A", "B", "C"]), seed=7)
    baseline = result["runs"]["features"]["baseline"]

    assert baseline["per_class"] == {
        "A": {
            "precision": pytest.approx(1 / 3),
            "recall": 1.0,
            "f1": 0.5,
            "support": 1,
        },
        "B": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
        "C": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1},
    }
    assert baseline["macro_f1"] == pytest.approx(1 / 6)
    assert baseline["balanced_accuracy"] == pytest.approx(1 / 3)
    assert baseline["confusion_matrix"] == {
        "labels": ["A", "B", "C"],
        "matrix": [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
    }


def test_a_tie_for_the_majority_label_goes_to_the_first_sorted_label() -> None:
    result = benchmark.run(table(["B", "A", "B", "A"]), table(["A", "B"]), seed=7)
    matrix = result["runs"]["features"]["baseline"]["confusion_matrix"]

    assert matrix == {"labels": ["A", "B"], "matrix": [[1, 0], [1, 0]]}


def test_a_class_only_in_the_test_split_gets_recall_0_and_its_support() -> None:
    test = table(["A", "B", "C", "C"])

    model = benchmark.run(separable(), test, seed=7)["runs"]["features"]["model"]

    assert model["per_class"]["C"]["recall"] == 0.0
    assert model["per_class"]["C"]["support"] == 2
    assert model["confusion_matrix"]["labels"] == ["A", "B", "C"]


def test_a_class_only_in_the_training_split_has_support_0_and_no_balanced_accuracy_share() -> (
    None
):
    result = benchmark.run(table(["A", "A", "B"]), table(["A", "A"]), seed=7)
    baseline = result["runs"]["features"]["baseline"]

    assert baseline["per_class"]["B"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }
    assert baseline["balanced_accuracy"] == 1.0


# Unlabelled rows (D48) and the given split (D49).


def test_unlabelled_rows_are_skipped_and_counted_and_the_tables_are_unchanged() -> None:
    train = table(["A", None, "B", "A", None, "B"])
    test = table(["A", None, "B"])
    before = (train.to_frame(), test.to_frame())

    result = benchmark.run(train, test, seed=7)

    assert result["unlabelled_skipped"] == {"train": 2, "test": 1}
    pd.testing.assert_frame_equal(train.to_frame(), before[0])
    pd.testing.assert_frame_equal(test.to_frame(), before[1])
    run = result["runs"]["features"]
    for scores in (run["model"], run["baseline"]):
        assert sum(sum(row) for row in scores["confusion_matrix"]["matrix"]) == 2


def test_the_given_test_split_is_scored_as_it_is() -> None:
    test = table(["A", "B", "B", "A", "B"])

    run = benchmark.run(separable(), test, seed=7)["runs"]["features"]

    assert {
        label: entry["support"] for label, entry in run["model"]["per_class"].items()
    } == {
        "A": 2,
        "B": 3,
    }


def test_a_test_split_with_no_labelled_row_has_no_balanced_accuracy() -> None:
    result = benchmark.run(separable(), table([None, None]), seed=7)

    assert result["unlabelled_skipped"]["test"] == 2
    assert result["runs"]["features"]["model"]["balanced_accuracy"] is None


# Strict serialisation.


def test_the_result_holds_no_nan_and_serialises_under_the_strict_writer(
    tmp_path: Any,
) -> None:
    result = benchmark.run(
        separable(scores=True), table(["A", "B", "C"], scores=[0.0, 1.0, 2.0]), seed=7
    )

    text = json.dumps(result, allow_nan=False)
    assert "NaN" not in text
    path = tmp_path / "report.json"
    report.write(
        report.build(
            command="build",
            fingerprint="ab" * 32,
            seed=7,
            start_time_zone="utc",
            splits={},
            benchmark=result,
        ),
        path,
    )
    assert json.loads(path.read_text(encoding="utf-8"))["benchmark"] == result


# Refusals.


def test_a_training_table_without_a_label_column_is_refused() -> None:
    with pytest.raises(NetGenError, match="no label column"):
        benchmark.preflight(table(None, duration=[1.0, 2.0]))


def test_a_training_table_with_one_class_after_skipping_is_refused() -> None:
    with pytest.raises(NetGenError, match=r"at least 2 classes.*found 1: \['A'\]"):
        benchmark.preflight(table(["A", None, "A"]))


def test_a_training_table_with_every_label_missing_is_refused() -> None:
    with pytest.raises(NetGenError, match=r"at least 2 classes.*found 0"):
        benchmark.preflight(table([None, None]))


def test_a_test_table_whose_columns_differ_from_the_training_table_is_refused() -> None:
    with pytest.raises(NetGenError, match="columns differ"):
        benchmark.run(separable(), table(None, duration=[1.0, 2.0]), seed=7)


def test_the_preflight_raises_before_any_model_is_trained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        sklearn.ensemble.RandomForestClassifier,
        "fit",
        lambda self, *args, **kwargs: calls.append(self),
    )

    with pytest.raises(NetGenError, match="at least 2 classes"):
        benchmark.run(table(["A", "A"]), table(["A", "B"]), seed=7)
    assert calls == []


def test_without_scikit_learn_the_modules_import_and_only_the_run_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(sys.modules):
        if name == "sklearn" or name.startswith("sklearn."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "netgen.benchmark")
    monkeypatch.delitem(sys.modules, "netgen.report")
    monkeypatch.setitem(sys.modules, "sklearn", None)

    hidden = importlib.import_module("netgen.benchmark")
    importlib.import_module("netgen.report")

    for call in (
        lambda: hidden.preflight(separable()),
        lambda: hidden.run(separable(), separable(10), seed=7),
    ):
        with pytest.raises(BenchmarkUnavailable) as raised:
            call()
        assert isinstance(raised.value, NetGenError)
        assert "scikit-learn" in str(raised.value)
        assert "pip install 'netgen[benchmark]'" in str(raised.value)
