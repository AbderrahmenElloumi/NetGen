"""Tests of the report (netgen.report): its shape, its statistics, the AUC and the strict JSON file."""

import importlib
import json
import math
import sys
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import netgen
from netgen import record, report
from netgen.errors import NetGenError
from netgen.record import Role
from netgen.table import Count, FlowTable

HEADER_KEYS = {
    "report_version",
    "netgen_version",
    "command",
    "input_fingerprint",
    "seed",
    "start_time_zone",
}
SPLIT_KEYS = {"rows", "labels", "duplicates", "features", "counts", "anomaly_score"}
STAT_KEYS = {"mean", "std", "min", "max", "median", "share_zero"}
COUNT_KEYS = {
    "filled",
    "no_finite_value",
    "non_ip_dropped",
    "encoding",
    "out_of_range",
    "constant_features",
    "constant_mismatch",
}
SCORE_KEYS = {"median", "p90", "p99", "max", "median_by_label", "auc"}
FINGERPRINT = "ab" * 32


def table(
    duration: list[float],
    *,
    labels: list[str | None] | None = None,
    scores: list[float] | None = None,
    src_ip: list[str] | None = None,
    payload: list[float] | None = None,
) -> FlowTable:
    """A small prepared table: `src_ip` (identity), `duration` and `payload_bytes` (features),
    an optional `label` and an optional `anomaly_score` (derived)."""
    rows = len(duration)
    columns: dict[str, Any] = {
        "src_ip": pd.Series(
            src_ip or [f"192.0.2.{i}" for i in range(rows)], dtype="string"
        ),
        "duration": pd.Series(duration, dtype=float),
        "payload_bytes": pd.Series(payload or [1.0] * rows, dtype=float),
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


def full_report(**splits: Mapping[str, Any]) -> dict[str, Any]:
    return report.build(
        command="build",
        fingerprint=FINGERPRINT,
        seed=42,
        start_time_zone="utc",
        splits=splits,
    )


def keys(value: object) -> set[str]:
    """Every key at every level of a nested report value."""
    if isinstance(value, Mapping):
        return set(value) | {key for item in value.values() for key in keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in keys(item)}
    return set()


def refuse_constant(token: str) -> float:
    raise AssertionError(f"The report file holds the non-standard token {token}.")


# Carried defects


def test_d44_no_reference_dataset_claim_in_the_report_or_the_summary() -> None:
    split = report.split(
        table(
            [1.0, 2.0, 3.0],
            labels=["BENIGN", "ATTACK", "BENIGN"],
            scores=[0.1, 0.9, 0.2],
        ),
        {},
        normal_label="BENIGN",
    )
    built = full_report(train=split, test=split)
    words = ("reference", "comparison", "compare")
    assert not [
        key for key in keys(built) if any(word in key.lower() for word in words)
    ]
    text = report.summary(built, ()).lower()
    assert not [word for word in words if word in text]


def test_d45_features_and_duplicates_cover_feature_columns_only() -> None:
    flows = table(
        [1.0, 1.0, 2.0, 2.0],
        payload=[5.0, 5.0, 6.0, 7.0],
        src_ip=["192.0.2.1", "192.0.2.2", "192.0.2.3", "192.0.2.3"],
        labels=["BENIGN", "ATTACK", "BENIGN", "BENIGN"],
        scores=[0.1, 0.2, 0.3, 0.4],
    )
    split = report.split(flows, {})
    assert list(split["features"]) == list(flows.features().columns)
    # Rows 0 and 1 match in features and differ in identity and label: one duplicate.
    # Rows 2 and 3 match in identity and differ in a feature: no duplicate.
    assert split["duplicates"] == 1


def test_d46_the_written_file_is_strict_json_with_null_for_undefined_statistics(
    tmp_path: Path,
) -> None:
    infinite = report.split(table([1.0, math.inf]), {})
    empty = report.split(table([], scores=[]), {})
    path = tmp_path / "report.json"
    report.write(full_report(train=infinite, test=empty), path)
    parsed = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=refuse_constant
    )
    assert parsed["splits"]["train"]["features"]["duration"]["mean"] is None
    assert parsed["splits"]["test"]["features"]["duration"] == dict.fromkeys(STAT_KEYS)
    assert parsed["splits"]["test"]["anomaly_score"]["median"] is None


def test_the_writer_turns_numpy_scalars_into_plain_json_values(tmp_path: Path) -> None:
    benchmark = {
        "support": np.int64(3),
        "f1": np.float64(0.5),
        "undefined": np.float64("nan"),
        "infinite": math.inf,
        "balanced": np.bool_(True),
        "classes": ("ATTACK", "BENIGN"),
    }
    path = tmp_path / "report.json"
    report.write(
        report.build(
            command="build",
            fingerprint=FINGERPRINT,
            seed=42,
            start_time_zone="utc",
            splits={},
            benchmark=benchmark,
        ),
        path,
    )
    parsed = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=refuse_constant
    )
    assert parsed["benchmark"] == {
        "support": 3,
        "f1": 0.5,
        "undefined": None,
        "infinite": None,
        "balanced": True,
        "classes": ["ATTACK", "BENIGN"],
    }


def test_the_writer_turns_a_numpy_string_into_a_plain_string(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    report.write({"label": np.str_("BENIGN"), "missing": None}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "label": "BENIGN",
        "missing": None,
    }


@pytest.mark.parametrize(
    ("value", "name"),
    [
        (Decimal("0.5"), "Decimal"),
        ({"BENIGN"}, "set"),
        (np.array([1, 2]), "ndarray"),
    ],
)
def test_the_writer_refuses_a_value_json_cannot_hold_by_its_type_name(
    tmp_path: Path, value: object, name: str
) -> None:
    path = tmp_path / "report.json"

    with pytest.raises(NetGenError, match=rf"\b{name}\b"):
        report.write({"benchmark": {"value": value}}, path)
    assert not path.exists()


# Shape


@pytest.mark.parametrize(
    ("flows", "normal"),
    [
        (table([1.0, 2.0], labels=["BENIGN", "ATTACK"]), None),
        (table([1.0, 2.0]), None),
        (table([1.0, 2.0], labels=["BENIGN", "ATTACK"], scores=[0.1, 0.9]), "BENIGN"),
    ],
    ids=["labelled", "unlabelled", "autoencoder"],
)
def test_the_report_has_the_spec_shape(flows: FlowTable, normal: str | None) -> None:
    built = full_report(train=report.split(flows, {}, normal_label=normal))
    assert set(built) == {"header", "splits", "benchmark"}
    assert set(built["header"]) == HEADER_KEYS
    assert built["header"]["report_version"] == 1
    assert built["header"]["netgen_version"] == netgen.__version__
    assert built["header"]["command"] == "build"
    assert built["header"]["input_fingerprint"] == FINGERPRINT
    assert built["header"]["seed"] == 42
    assert built["benchmark"] is None
    split = built["splits"]["train"]
    assert set(split) == SPLIT_KEYS
    assert set(split["counts"]) == COUNT_KEYS
    assert {name: set(stats) for name, stats in split["features"].items()} == {
        "duration": STAT_KEYS,
        "payload_bytes": STAT_KEYS,
    }
    if flows.roles.get(record.ANOMALY_SCORE) is Role.DERIVED:
        assert set(split["anomaly_score"]) == SCORE_KEYS
    else:
        assert split["anomaly_score"] is None


def test_splits_and_benchmark_are_carried_as_given() -> None:
    train = report.split(table([1.0]), {})
    test = report.split(table([2.0, 3.0]), {})
    benchmark = {"settings": {"n_estimators": 100}}
    built = report.build(
        command="build",
        fingerprint=FINGERPRINT,
        seed=7,
        start_time_zone="utc",
        splits={"train": train, "test": test},
        benchmark=benchmark,
    )
    assert list(built["splits"]) == ["train", "test"]
    assert built["splits"]["test"]["rows"] == 2
    assert built["benchmark"] == benchmark
    assert built["header"]["seed"] == 7


@pytest.mark.parametrize("zone", ["utc", "local"])
def test_the_start_time_zone_is_carried_into_the_header(zone: str) -> None:
    built = report.build(
        command="transform",
        fingerprint=FINGERPRINT,
        seed=42,
        start_time_zone=zone,  # type: ignore[arg-type]
        splits={"transform": report.split(table([1.0]), {})},
    )
    assert built["header"]["start_time_zone"] == zone
    assert built["header"]["command"] == "transform"


# Statistics


def test_an_empty_table_gives_zero_rows_and_null_statistics() -> None:
    split = report.split(table([], labels=[]), {})
    assert (split["rows"], split["labels"], split["duplicates"]) == (0, {}, 0)
    assert split["features"] == {
        "duration": dict.fromkeys(STAT_KEYS),
        "payload_bytes": dict.fromkeys(STAT_KEYS),
    }


def test_statistics_match_hand_worked_values() -> None:
    split = report.split(table([0.0, 1.0, 2.0, 5.0], payload=[0.0, 0.0, 0.0, 4.0]), {})
    # mean 2; squared deviations 4, 1, 0, 9 give a population variance of 3.5.
    assert split["features"]["duration"] == pytest.approx(
        {
            "mean": 2.0,
            "std": math.sqrt(3.5),
            "min": 0.0,
            "max": 5.0,
            "median": 1.5,
            "share_zero": 0.25,
        }
    )
    assert split["features"]["payload_bytes"]["share_zero"] == 0.75
    assert split["features"]["payload_bytes"]["std"] == pytest.approx(math.sqrt(3.0))


def test_labels_count_labelled_rows_only() -> None:
    split = report.split(
        table([1.0] * 4, labels=["BENIGN", None, "ATTACK", "BENIGN"]), {}
    )
    assert split["rows"] == 4
    assert split["labels"] == {"ATTACK": 1, "BENIGN": 2}


def test_an_unlabelled_table_gives_empty_labels() -> None:
    assert report.split(table([1.0, 2.0]), {})["labels"] == {}


# Counts


def test_counts_keep_columns_above_zero_in_feature_order() -> None:
    counts: dict[str, Count] = {
        "packets_read": 10,
        "non_ip_dropped": 3,
        "filled": {"payload_bytes": 2, "duration": 1},
        "no_finite_value": ("payload_bytes",),
        "out_of_range": {"duration": 0, "payload_bytes": 4},
        "constant_features": ("payload_bytes",),
        "constant_mismatch": {"duration": 0, "payload_bytes": 0},
    }
    split = report.split(table([1.0, 2.0]), counts)
    assert split["counts"] == {
        "filled": {"duration": 1, "payload_bytes": 2},
        "no_finite_value": ["payload_bytes"],
        "non_ip_dropped": 3,
        "encoding": None,
        "out_of_range": {"payload_bytes": 4},
        "constant_features": ["payload_bytes"],
        "constant_mismatch": {},
    }
    assert list(split["counts"]["filled"]) == ["duration", "payload_bytes"]


def test_a_flow_export_records_its_encoding_and_drops_no_packets() -> None:
    counts = report.split(table([1.0]), {"encoding": "cp1252"})["counts"]
    assert (counts["encoding"], counts["non_ip_dropped"]) == ("cp1252", 0)


def test_the_constant_mismatch_counts_appear_when_above_zero() -> None:
    counts = report.split(table([1.0]), {"constant_mismatch": {"duration": 2}})[
        "counts"
    ]
    assert counts["constant_mismatch"] == {"duration": 2}


# Anomaly score


def test_anomaly_score_percentiles_and_median_by_label_match_hand_worked_values() -> (
    None
):
    flows = table(
        [1.0] * 5,
        labels=["BENIGN", "BENIGN", "ATTACK", "ATTACK", "ATTACK"],
        scores=[1.0, 2.0, 3.0, 4.0, 10.0],
    )
    score = report.split(flows, {})["anomaly_score"]
    # Linear interpolation: p90 sits at 3.6 of 4 gaps, p99 at 3.96, between 4 and 10.
    by_label = score.pop("median_by_label")
    assert score == pytest.approx(
        {"median": 3.0, "p90": 7.6, "p99": 9.76, "max": 10.0, "auc": None}
    )
    assert by_label == {"ATTACK": 4.0, "BENIGN": 1.5}


def test_an_unlabelled_anomaly_score_has_no_median_by_label() -> None:
    score = report.split(table([1.0, 1.0], scores=[0.5, 1.5]), {})["anomaly_score"]
    assert score["median_by_label"] == {}
    assert score["median"] == 1.0


def test_without_the_derived_column_the_anomaly_score_is_null() -> None:
    flows = table([1.0, 2.0], labels=["BENIGN", "ATTACK"])
    assert report.split(flows, {}, normal_label="BENIGN")["anomaly_score"] is None


def test_the_auc_of_a_split_treats_the_rest_as_positive() -> None:
    flows = table(
        [1.0] * 5,
        labels=["BENIGN", "BENIGN", "ATTACK", None, "DOS"],
        scores=[1.0, 2.0, 3.0, 0.0, 10.0],
    )
    assert report.split(flows, {}, normal_label="BENIGN")["anomaly_score"]["auc"] == 1.0
    reversed_flows = table(
        [1.0] * 3, labels=["BENIGN", "ATTACK", "ATTACK"], scores=[9.0, 1.0, 2.0]
    )
    split = report.split(reversed_flows, {}, normal_label="BENIGN")
    assert split["anomaly_score"]["auc"] == 0.0


@pytest.mark.parametrize(
    ("scores", "positive", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], [False, False, True, True], 1.0),
        ([3.0, 4.0, 1.0, 2.0], [False, False, True, True], 0.0),
        ([5.0, 5.0, 5.0, 5.0], [False, True, False, True], 0.5),
        # Pairs (positive, negative): (2,1) (2,3) (4,1) (4,3) win 1, 0, 1, 1.
        ([1.0, 3.0, 2.0, 4.0], [False, False, True, True], 0.75),
        # Pairs: (2,1) (2,2) (3,1) (3,2) win 1, 0.5, 1, 1.
        ([1.0, 2.0, 2.0, 3.0], [False, False, True, True], 0.875),
    ],
    ids=["perfect", "reversed", "tie", "mixed", "partial-tie"],
)
def test_rank_auc_matches_hand_worked_values(
    scores: list[float], positive: list[bool], expected: float
) -> None:
    assert report.rank_auc(scores, positive) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("positive"),
    [[True, True], [False, False], []],
    ids=["no-negative", "no-positive", "empty"],
)
def test_rank_auc_is_null_with_one_group_missing(positive: list[bool]) -> None:
    assert report.rank_auc([1.0] * len(positive), positive) is None


@pytest.mark.parametrize(
    ("scores", "positive"),
    [
        ([float("nan"), 1.0], [True, False]),
        ([1.0, 2.0, float("nan")], [True, False, False]),
        ([float("inf"), 1.0], [True, False]),
        ([float("-inf"), 1.0], [True, False]),
        ([1.0, float("inf")], [True, False]),
        ([1.0, float("-inf")], [True, False]),
    ],
    ids=[
        "nan-positive",
        "nan-negative",
        "inf-positive",
        "minus-inf-positive",
        "inf-negative",
        "minus-inf-negative",
    ],
)
def test_rank_auc_is_null_for_a_score_that_is_not_finite(
    scores: list[float], positive: list[bool]
) -> None:
    assert report.rank_auc(scores, positive) is None


@pytest.mark.parametrize(
    ("labels", "normal"),
    [
        (["BENIGN", "ATTACK"], None),
        ([None, None], "BENIGN"),
        (["BENIGN", "BENIGN"], "BENIGN"),
        (["ATTACK", "DOS"], "BENIGN"),
    ],
    ids=["no-normal-label", "no-labelled-rows", "normal-only", "rest-only"],
)
def test_the_auc_is_null_without_both_groups(
    labels: list[str | None], normal: str | None
) -> None:
    flows = table([1.0, 1.0], labels=labels, scores=[0.1, 0.9])
    assert report.split(flows, {}, normal_label=normal)["anomaly_score"]["auc"] is None


def test_the_auc_is_null_for_an_unlabelled_table() -> None:
    flows = table([1.0, 1.0], scores=[0.1, 0.9])
    assert (
        report.split(flows, {}, normal_label="BENIGN")["anomaly_score"]["auc"] is None
    )


# Console summary


def test_the_summary_names_the_label_counts_the_filled_total_and_each_warning() -> None:
    labelled = report.split(
        table([1.0, 2.0, 3.0], labels=["BENIGN", "ATTACK", "BENIGN"]),
        {
            "filled": {"duration": 2, "payload_bytes": 3},
            "non_ip_dropped": 4,
            "out_of_range": {"duration": 6},
            "constant_features": ("payload_bytes",),
        },
    )
    unlabelled = report.split(table([1.0]), {})
    warnings = ("The input columns ['x'] are dropped.", "A second warning.")
    text = report.summary(full_report(train=labelled, test=unlabelled), warnings)
    assert "train: 3 rows" in text
    assert "ATTACK 1, BENIGN 2" in text
    assert "filled values: 5" in text
    assert "non-IP packets dropped: 4" in text
    assert "constant features: payload_bytes" in text
    assert "out-of-range values: 6" in text
    assert "test: 1 rows (unlabelled)" in text
    assert "constant features: none" in text
    for warning in warnings:
        assert warning in text
    assert not text.lstrip().startswith("{")


# Optional extras


def test_the_report_imports_and_runs_with_torch_and_scikit_learn_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "netgen.report")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    hidden = importlib.import_module("netgen.report")
    flows = table([1.0, 2.0], labels=["BENIGN", "ATTACK"], scores=[0.1, 0.9])
    split = hidden.split(flows, {}, normal_label="BENIGN")
    assert split["anomaly_score"]["auc"] == 1.0
