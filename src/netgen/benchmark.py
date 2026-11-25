"""The benchmark: a Random Forest trained on the training split and scored on the test split.

`build --benchmark` calls `preflight` before it fits anything, then `run` on the
prepared training and test tables. scikit-learn is the optional `benchmark`
extra: this module imports it only inside `run`, so the core works without it.
The result holds plain Python values only, for the report's strict writer.
"""

import importlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from netgen import record
from netgen.errors import BenchmarkUnavailable, NetGenError
from netgen.record import Role
from netgen.table import FlowTable

TREES = 100
CLASS_WEIGHT = "balanced"
JOBS = 2


def preflight(train: FlowTable) -> None:
    """Raise a `NetGenError` when the benchmark cannot run on `train`.

    It refuses when scikit-learn is missing, when the table has no label column,
    and when fewer than 2 classes remain among the labelled training rows.
    """
    _ensemble()
    labels = _labels(train)
    if labels is None:
        raise NetGenError(
            "The benchmark needs labels, but the training table has no label column. "
            "Build from a labelled input, or leave out --benchmark."
        )
    classes = sorted(set(labels.dropna()))
    if len(classes) < 2:
        raise NetGenError(
            "The benchmark needs at least 2 classes among the labelled training rows; "
            f"it found {len(classes)}: {classes}. "
            "Build from an input with more classes, or leave out --benchmark."
        )


def run(train: FlowTable, test: FlowTable, *, seed: int) -> dict[str, Any]:
    """The report's `benchmark` entry: the settings, the unlabelled rows skipped, and the runs.

    The `features` run uses the feature columns. The `features_plus_derived` run
    exists only when `train` has derived columns, and adds them. Each run holds
    its `columns`, the `model` metrics and the majority-class `baseline` metrics.
    Neither table is changed.
    """
    preflight(train)
    if dict(test.roles) != dict(train.roles):
        raise NetGenError(
            "The test table's columns differ from the training table's. "
            "Give the benchmark the two splits of one build."
        )
    ensemble = _ensemble()
    # The preflight found a label column, and both tables have the same roles.
    train_labels = cast(pd.Series, _labels(train))
    test_labels = cast(pd.Series, _labels(test))
    train_rows = train_labels.notna().to_numpy()
    test_rows = test_labels.notna().to_numpy()
    y_train = train_labels[train_rows].astype(str).to_numpy()
    y_test = test_labels[test_rows].astype(str).to_numpy()
    classes = sorted(set(y_train) | set(y_test))
    majority = min(
        pd.Series(y_train).value_counts().items(), key=lambda item: (-item[1], item[0])
    )[0]
    baseline = _metrics(y_test, np.full(len(y_test), majority), classes)

    features = _names(train, Role.FEATURE)
    derived = _names(train, Role.DERIVED)
    sets = {"features": features}
    if derived:
        sets["features_plus_derived"] = features + derived
    train_frame = train.to_frame()
    test_frame = test.to_frame()
    runs: dict[str, Any] = {}
    for name, columns in sets.items():
        forest = ensemble.RandomForestClassifier(
            n_estimators=TREES,
            class_weight=CLASS_WEIGHT,
            random_state=seed,
            n_jobs=JOBS,
        )
        forest.fit(train_frame.loc[train_rows, columns].to_numpy(dtype=float), y_train)
        predicted = (
            forest.predict(test_frame.loc[test_rows, columns].to_numpy(dtype=float))
            if len(y_test)
            else np.array([], dtype=str)
        )
        runs[name] = {
            "columns": list(columns),
            "model": _metrics(y_test, np.asarray(predicted).astype(str), classes),
            "baseline": baseline,
        }
    return {
        "settings": {
            "trees": TREES,
            "class_weight": CLASS_WEIGHT,
            "seed": seed,
            "n_jobs": JOBS,
        },
        "unlabelled_skipped": {
            "train": int((~train_rows).sum()),
            "test": int((~test_rows).sum()),
        },
        "runs": runs,
    }


def _ensemble() -> ModuleType:
    """`sklearn.ensemble`, or `BenchmarkUnavailable` naming the extra when scikit-learn is missing."""
    try:
        return importlib.import_module("sklearn.ensemble")
    except ImportError as error:
        raise BenchmarkUnavailable(
            "The benchmark needs scikit-learn, which is not installed. "
            "Install the extra: pip install 'netgen[benchmark]'."
        ) from error


def _labels(table: FlowTable) -> pd.Series | None:
    """The label column, with missing labels as missing, or `None` when the table has none."""
    names = _names(table, Role.LABEL)
    if not names:
        return None
    return table.to_frame()[names[0]].astype("string").reset_index(drop=True)


def _names(table: FlowTable, role: Role) -> list[str]:
    """The columns of `role`, in dataset column order."""
    return [
        name for name in record.dataset_order(table.roles) if table.roles[name] is role
    ]


def _metrics(
    truth: npt.NDArray[Any], predicted: npt.NDArray[Any], classes: Sequence[str]
) -> dict[str, Any]:
    """Macro F1, balanced accuracy, per-class scores and the confusion matrix over `classes`.

    A score whose denominator is 0 is 0. Balanced accuracy is the mean recall over
    the classes with support, and `None` when no class has support.
    """
    index = {label: i for i, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, guess in zip(truth, predicted, strict=True):
        matrix[index[str(actual)], index[str(guess)]] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    for i, label in enumerate(classes):
        hits = int(matrix[i, i])
        support = int(matrix[i].sum())
        guessed = int(matrix[:, i].sum())
        precision = hits / guessed if guessed else 0.0
        recall = hits / support if support else 0.0
        both = precision + recall
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / both if both else 0.0,
            "support": support,
        }
        if support:
            recalls.append(recall)
    return {
        "macro_f1": float(np.mean([entry["f1"] for entry in per_class.values()])),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(classes),
            "matrix": matrix.tolist(),
        },
    }
