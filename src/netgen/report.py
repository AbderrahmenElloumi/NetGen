"""The report: one shape for `fit`, `transform` and `build`.

A pure module. It takes the prepared table of a split, the counts the reader
and the stages returned, and the header facts, and returns plain dicts. It
reads no file and runs no stage. The writer emits strict JSON: an undefined
statistic is `null`, never `NaN` or `Infinity`.
"""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

import netgen
from netgen import record
from netgen.errors import NetGenError
from netgen.record import Role
from netgen.table import Count, FlowTable

REPORT_VERSION = 1

type Command = Literal["fit", "transform", "build"]
type StartTimeZone = Literal["utc", "local"]


def build(
    *,
    command: Command,
    fingerprint: str,
    seed: int,
    start_time_zone: StartTimeZone,
    splits: Mapping[str, Mapping[str, Any]],
    benchmark: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole report: the header, the split entries as given, and the benchmark (`None` when it did not run)."""
    return {
        "header": {
            "report_version": REPORT_VERSION,
            "netgen_version": netgen.__version__,
            "command": command,
            "input_fingerprint": fingerprint,
            "seed": seed,
            "start_time_zone": start_time_zone,
        },
        "splits": dict(splits),
        "benchmark": None if benchmark is None else dict(benchmark),
    }


def split(
    table: FlowTable,
    counts: Mapping[str, Count],
    *,
    normal_label: str | None = None,
) -> dict[str, Any]:
    """The report entry of one split, from its prepared table and its counts.

    `normal_label` is the `--fit-on-label` value: with it, and labelled rows of
    both the normal class and the rest, the anomaly score gets a ROC AUC.
    """
    features = table.features()
    frame = table.to_frame()
    labels = _labels(table, frame)
    order = [str(name) for name in features.columns]
    return {
        "rows": len(frame),
        "labels": {}
        if labels is None
        else {str(k): int(v) for k, v in labels.value_counts().sort_index().items()},
        "duplicates": int(features.duplicated().sum()),
        "features": {name: _statistics(_floats(features[name])) for name in order},
        "counts": {
            "filled": _per_column(counts, "filled", order),
            "no_finite_value": list(_names(counts, "no_finite_value")),
            "non_ip_dropped": int(cast(int, counts.get("non_ip_dropped", 0))),
            "encoding": cast(str | None, counts.get("encoding")),
            "out_of_range": _per_column(counts, "out_of_range", order),
            "constant_features": list(_names(counts, "constant_features")),
            "constant_mismatch": _per_column(counts, "constant_mismatch", order),
        },
        "anomaly_score": _anomaly_score(table, frame, labels, normal_label),
    }


def rank_auc(
    scores: Sequence[float] | npt.ArrayLike, positive: Sequence[bool] | npt.ArrayLike
) -> float | None:
    """The ROC AUC by the rank (Mann-Whitney) formula, average ranks for ties.

    `scores` and `positive` must have the same length. `None` when either
    class is absent or any score is not finite: a NaN has no rank, and an
    infinity would decide its pairs by sign alone.
    """
    flags = np.asarray(positive, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    positives = int(flags.sum())
    negatives = len(flags) - positives
    if positives == 0 or negatives == 0 or not np.isfinite(values).all():
        return None
    ranks = pd.Series(values).rank(method="average").to_numpy()
    wins = float(ranks[flags].sum()) - positives * (positives + 1) / 2
    return _number(wins / (positives * negatives))


def write(report: Mapping[str, Any], path: Path) -> None:
    """Write `report` to `path` as strict JSON; a non-finite number becomes `null`."""
    path.write_text(
        json.dumps(_plain(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def summary(report: Mapping[str, Any], warnings: Sequence[str]) -> str:
    """The short console text: per split its rows, labels and count totals, then each warning."""
    lines: list[str] = []
    for name, entry in report["splits"].items():
        labels = entry["labels"]
        described = (
            "labels: "
            + ", ".join(f"{label} {count}" for label, count in labels.items())
            if labels
            else "unlabelled"
        )
        counts = entry["counts"]
        constant = ", ".join(counts["constant_features"]) or "none"
        lines += [
            f"{name}: {entry['rows']} rows ({described})",
            f"  filled values: {sum(counts['filled'].values())}",
            f"  non-IP packets dropped: {counts['non_ip_dropped']}",
            f"  constant features: {constant}",
            f"  out-of-range values: {sum(counts['out_of_range'].values())}",
        ]
    lines += [f"Warning: {warning}" for warning in warnings]
    return "\n".join(lines)


def _labels(table: FlowTable, frame: pd.DataFrame) -> pd.Series | None:
    """The labelled rows' labels, or `None` when the table has no label column."""
    names = [name for name, role in table.roles.items() if role is Role.LABEL]
    if not names:
        return None
    labels = frame[names[0]]
    return labels[labels.notna()].astype(str)


def _anomaly_score(
    table: FlowTable,
    frame: pd.DataFrame,
    labels: pd.Series | None,
    normal_label: str | None,
) -> dict[str, Any] | None:
    if table.roles.get(record.ANOMALY_SCORE) is not Role.DERIVED:
        return None
    scores = frame[record.ANOMALY_SCORE].astype(np.float64)
    values = scores.to_numpy()
    empty = len(values) == 0
    by_label: dict[str, float | None] = {}
    auc = None
    if labels is not None:
        labelled = scores[labels.index]
        by_label = {
            str(label): _number(median)
            for label, median in labelled.groupby(labels).median().sort_index().items()
        }
        if normal_label is not None:
            auc = rank_auc(labelled.to_numpy(), (labels != normal_label).to_numpy())
    return {
        "median": None if empty else _number(np.median(values)),
        "p90": None if empty else _number(np.percentile(values, 90)),
        "p99": None if empty else _number(np.percentile(values, 99)),
        "max": None if empty else _number(values.max()),
        "median_by_label": by_label,
        "auc": auc,
    }


def _statistics(column: npt.NDArray[np.float64]) -> dict[str, float | None]:
    if len(column) == 0:
        return dict.fromkeys(("mean", "std", "min", "max", "median", "share_zero"))
    # A non-finite value makes a statistic undefined: it becomes None, not a warning.
    with np.errstate(invalid="ignore"):
        return {
            "mean": _number(column.mean()),
            "std": _number(column.std(ddof=0)),
            "min": _number(column.min()),
            "max": _number(column.max()),
            "median": _number(np.median(column)),
            "share_zero": _number((column == 0).mean()),
        }


def _floats(series: pd.Series) -> npt.NDArray[np.float64]:
    return np.asarray(series.astype(np.float64).to_numpy(), dtype=np.float64)


def _per_column(
    counts: Mapping[str, Count], key: str, order: Sequence[str]
) -> dict[str, int]:
    """The count per column of `key`, only above 0, in feature order."""
    values = cast(Mapping[str, int], counts.get(key, {}))
    return {name: int(values[name]) for name in order if values.get(name, 0) > 0}


def _names(counts: Mapping[str, Count], key: str) -> tuple[str, ...]:
    return cast(tuple[str, ...], counts.get(key, ()))


def _number(value: object) -> float | None:
    """`value` as a float, or `None` when it is not finite."""
    number = float(cast(float, value))
    return number if math.isfinite(number) else None


def _plain(value: object) -> object:
    """`value` with numpy scalars as Python values and non-finite floats as `None`.

    Raises `NetGenError` naming the type of any value JSON cannot hold.
    """
    if isinstance(value, np.generic):
        return _plain(value.item())
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _number(value)
    raise NetGenError(
        f"The report cannot hold a value of type {type(value).__name__}. "
        "Give plain numbers, strings, lists and dicts."
    )
