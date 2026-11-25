"""Tests of the learning-stage protocol (netgen.stages) and the fill step (netgen.stages.fill)."""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netgen.errors import ArtefactsError
from netgen.record import Role
from netgen.stages import Fitted, Stage, fill
from netgen.table import FlowTable

INF = math.inf
NAN = math.nan

ROLES = {
    "src_ip": Role.IDENTITY,
    "dst_port": Role.IDENTITY,
    "label": Role.LABEL,
    "duration": Role.FEATURE,
    "bytes_per_s": Role.FEATURE,
    "down_up_ratio": Role.FEATURE,
}
FEATURES = ["duration", "bytes_per_s", "down_up_ratio"]


def make_table(
    duration: list[float], bytes_per_s: list[float], down_up_ratio: list[float]
) -> FlowTable:
    rows = len(duration)
    frame = pd.DataFrame(
        {
            "src_ip": pd.Series([f"192.0.2.{i}" for i in range(rows)], dtype="string"),
            "dst_port": pd.Series([443] * (rows - 1) + [None], dtype="Int64"),
            "label": pd.Series(["BENIGN"] * rows, dtype="string"),
            "duration": pd.Series(duration, dtype=float),
            "bytes_per_s": pd.Series(bytes_per_s, dtype=float),
            "down_up_ratio": pd.Series(down_up_ratio, dtype=float),
        }
    )
    return FlowTable(frame, ROLES)


def training_table() -> FlowTable:
    # bytes_per_s: finite 100 and 300, so max 300 and median 200.
    # down_up_ratio: finite 0.5 and 1.0, so max 1.0 and median 0.75.
    return make_table(
        duration=[1.0, 2.0, 3.0, 10.0],
        bytes_per_s=[100.0, NAN, INF, 300.0],
        down_up_ratio=[0.5, 1.0, -INF, NAN],
    )


def column(table: FlowTable, name: str) -> list[float]:
    return [float(value) for value in table.to_frame()[name]]


# The protocol


def test_the_fill_step_conforms_to_the_stage_protocol() -> None:
    stage: Stage = fill
    assert isinstance(stage, Stage)
    fitted: Fitted = stage.fit(training_table())
    assert isinstance(fitted, Fitted)


# Carried defects, written first


def test_d36_a_fitted_fill_step_cannot_be_fitted_again() -> None:
    first = fill.fit(training_table())
    before = first.transform(training_table())
    second = fill.fit(make_table([5.0], [5.0], [5.0]))
    assert second is not first
    after = first.transform(training_table())
    pd.testing.assert_frame_equal(after.table.to_frame(), before.table.to_frame())
    assert not hasattr(first, "fit")
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.values = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.values["duration"] = first.values["bytes_per_s"]  # type: ignore[index]


def test_d28_d29_a_missing_rate_gets_the_training_median_and_is_counted() -> None:
    # A zero-duration flow has no rate: the reader leaves it missing.
    table = make_table(
        duration=[0.0, 1.0, 2.0, 4.0],
        bytes_per_s=[NAN, 100.0, 200.0, 400.0],
        down_up_ratio=[1.0, 1.0, 1.0, 1.0],
    )
    result = fill.fit(table).transform(table)
    assert column(result.table, "bytes_per_s") == [200.0, 100.0, 200.0, 400.0]
    assert result.counts["filled"] == {
        "duration": 0,
        "bytes_per_s": 1,
        "down_up_ratio": 0,
    }


def test_d31_no_infinity_reaches_the_output_and_no_missing_value_becomes_zero() -> None:
    result = fill.fit(training_table()).transform(training_table())
    features = result.table.features().to_numpy(dtype=float)
    assert np.isfinite(features).all()
    assert column(result.table, "bytes_per_s") == [100.0, 200.0, 300.0, 300.0]
    assert column(result.table, "down_up_ratio") == [0.5, 1.0, 1.0, 0.75]


def test_d31_a_finite_table_without_missing_values_is_unchanged() -> None:
    table = make_table([1.0, 2.0], [10.0, 20.0], [0.5, 2.0])
    result = fill.fit(training_table()).transform(table)
    pd.testing.assert_frame_equal(result.table.to_frame(), table.to_frame())
    assert result.counts["filled"] == dict.fromkeys(FEATURES, 0)
    assert result.warnings == ()


# Scenarios


def test_infinity_becomes_the_largest_finite_training_value() -> None:
    result = fill.fit(training_table()).transform(training_table())
    assert column(result.table, "bytes_per_s")[2] == 300.0


def test_negative_infinity_becomes_the_largest_finite_training_value() -> None:
    result = fill.fit(training_table()).transform(training_table())
    assert column(result.table, "down_up_ratio")[2] == 1.0


def test_a_missing_value_becomes_the_training_median() -> None:
    result = fill.fit(training_table()).transform(training_table())
    assert column(result.table, "bytes_per_s")[1] == 200.0
    assert column(result.table, "down_up_ratio")[3] == 0.75


def test_a_missing_value_in_a_nullable_integer_feature_becomes_the_median() -> None:
    frame = pd.DataFrame(
        {"duration": pd.Series([1, None, 4], dtype="Int64")},
    )
    table = FlowTable(frame, {"duration": Role.FEATURE})
    result = fill.fit(table).transform(table)
    assert column(result.table, "duration") == [1.0, 2.5, 4.0]
    assert result.counts["filled"] == {"duration": 1}


def test_replacements_are_counted_per_column() -> None:
    result = fill.fit(training_table()).transform(training_table())
    assert result.counts["filled"] == {
        "duration": 0,
        "bytes_per_s": 2,
        "down_up_ratio": 2,
    }
    assert result.counts["no_finite_value"] == ()


def test_the_values_come_from_the_training_table_only() -> None:
    fitted = fill.fit(training_table())
    other = make_table(
        duration=[1000.0, NAN, INF],
        bytes_per_s=[1.0, 2.0, 3.0],
        down_up_ratio=[7.0, 8.0, 9.0],
    )
    result = fitted.transform(other)
    assert column(result.table, "duration") == [1000.0, 2.5, 10.0]


def test_identity_label_and_derived_columns_are_untouched() -> None:
    table = training_table().with_derived("anomaly_score", [INF, NAN, -INF, 1.0])
    result = fill.fit(training_table()).transform(table)
    before = table.to_frame()
    after = result.table.to_frame()
    for name in ("src_ip", "dst_port", "label", "anomaly_score"):
        pd.testing.assert_series_equal(after[name], before[name])
    assert result.table.roles == table.roles


def test_fill_json_has_the_documented_shape_and_exact_values(tmp_path: Path) -> None:
    fill.fit(training_table()).save(tmp_path)
    saved = json.loads((tmp_path / "fill.json").read_text(encoding="utf-8"))
    assert saved == {
        "duration": {"max": 10.0, "median": 2.5},
        "bytes_per_s": {"max": 300.0, "median": 200.0},
        "down_up_ratio": {"max": 1.0, "median": 0.75},
    }


def test_fill_json_round_trips(tmp_path: Path) -> None:
    table = make_table(
        duration=[0.1, 0.7, NAN],
        bytes_per_s=[1 / 3, 2 / 7, INF],
        down_up_ratio=[NAN, 1e-300, 1e300],
    )
    fitted = fill.fit(table)
    fitted.save(tmp_path)
    loaded = fill.load(tmp_path, {})
    assert loaded.values == fitted.values
    pd.testing.assert_frame_equal(
        loaded.transform(table).table.to_frame(),
        fitted.transform(table).table.to_frame(),
    )


def test_a_feature_without_a_finite_training_value_is_filled_with_zero(
    tmp_path: Path,
) -> None:
    table = make_table(
        duration=[1.0, 2.0, 3.0],
        bytes_per_s=[1.0, 2.0, 3.0],
        down_up_ratio=[NAN, INF, -INF],
    )
    fitted = fill.fit(table)
    result = fitted.transform(table)
    assert column(result.table, "down_up_ratio") == [0.0, 0.0, 0.0]
    assert result.counts["no_finite_value"] == ("down_up_ratio",)
    assert len(result.warnings) == 1
    assert "down_up_ratio" in result.warnings[0]
    assert list(result.table.features().columns) == FEATURES
    fitted.save(tmp_path)
    saved = json.loads((tmp_path / "fill.json").read_text(encoding="utf-8"))
    assert saved["down_up_ratio"] == {"max": 0.0, "median": 0.0}


# Rejection branches


def test_transform_refuses_a_table_missing_a_feature_column() -> None:
    fitted = fill.fit(training_table())
    frame = training_table().to_frame().drop(columns="down_up_ratio")
    roles = {name: role for name, role in ROLES.items() if name != "down_up_ratio"}
    with pytest.raises(ArtefactsError, match="down_up_ratio"):
        fitted.transform(FlowTable(frame, roles))


def test_transform_refuses_a_table_with_an_extra_feature_column() -> None:
    fitted = fill.fit(training_table())
    frame = training_table().to_frame()
    frame["packets_per_s"] = 1.0
    with pytest.raises(ArtefactsError, match="packets_per_s"):
        fitted.transform(FlowTable(frame, {**ROLES, "packets_per_s": Role.FEATURE}))


def test_load_refuses_a_folder_without_fill_json(tmp_path: Path) -> None:
    with pytest.raises(ArtefactsError, match="fill.json"):
        fill.load(tmp_path, {})
