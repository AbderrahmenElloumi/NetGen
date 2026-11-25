"""Tests of the scaler (netgen.stages.scaler): a log step on `log` fields, then a Z-score."""

import dataclasses
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from netgen import record
from netgen.errors import ArtefactsError
from netgen.record import Role, Scaling
from netgen.stages import Fitted, Stage, scaler
from netgen.table import FlowTable

# Training values chosen so the hand-worked numbers are exact. `duration` is
# declared `log`: its values are expm1 of 0, 1 and 5, so after `log1p` they
# are 0, 1 and 5. `down_up_ratio` is declared `linear` and holds 0, 1 and 5.
# Both columns are 0, 1, 5 before the Z-score: mean 2, deviations -2, -1, 3,
# population variance 14/3 (the sample variance, ddof=1, would be 7).
TRAIN_DURATION = [math.expm1(0.0), math.expm1(1.0), math.expm1(5.0)]
TRAIN_RATIO = [0.0, 1.0, 5.0]
STD = math.sqrt(14 / 3)
EXPECTED = [-2 / STD, -1 / STD, 3 / STD]

ROLES = {
    "src_ip": Role.IDENTITY,
    "label": Role.LABEL,
    "duration": Role.FEATURE,
    "down_up_ratio": Role.FEATURE,
    "is_tcp": Role.FEATURE,
}
FEATURES = ["duration", "down_up_ratio", "is_tcp"]


def make_table(
    duration: list[float], down_up_ratio: list[float], is_tcp: list[float]
) -> FlowTable:
    rows = len(duration)
    frame = pd.DataFrame(
        {
            "src_ip": pd.Series([f"192.0.2.{i}" for i in range(rows)], dtype="string"),
            "label": pd.Series(["BENIGN"] * rows, dtype="string"),
            "duration": pd.Series(duration, dtype=float),
            "down_up_ratio": pd.Series(down_up_ratio, dtype=float),
            "is_tcp": pd.Series(is_tcp, dtype=float),
        }
    )
    return FlowTable(frame, ROLES)


def training_table() -> FlowTable:
    return make_table(TRAIN_DURATION, TRAIN_RATIO, [1.0, 1.0, 1.0])


def column(table: FlowTable, name: str) -> list[float]:
    return [float(value) for value in table.to_frame()[name]]


# The protocol


def test_the_scaler_conforms_to_the_stage_protocol() -> None:
    stage: Stage = scaler
    assert isinstance(stage, Stage)
    fitted: Fitted = stage.fit(training_table())
    assert isinstance(fitted, Fitted)


# Carried defects, written first


def test_d32_d33_a_log_field_gets_log1p_before_the_z_score_and_a_linear_field_does_not() -> (
    None
):
    result = scaler.fit(training_table()).transform(training_table())
    assert column(result.table, "duration") == pytest.approx(EXPECTED)
    assert column(result.table, "down_up_ratio") == pytest.approx(EXPECTED)


def test_d32_the_mean_and_std_are_taken_after_the_log_step_with_ddof_0() -> None:
    fitted = scaler.fit(training_table())
    assert fitted.values["duration"].log is True
    assert fitted.values["duration"].mean == pytest.approx(2.0)
    assert fitted.values["duration"].std == pytest.approx(STD)
    assert fitted.values["down_up_ratio"].log is False
    assert fitted.values["is_tcp"].log is False


def test_d34_a_constant_feature_is_kept_written_as_0_and_named() -> None:
    fitted = scaler.fit(training_table())
    result = fitted.transform(training_table())
    assert fitted.values["is_tcp"].constant is True
    assert fitted.values["duration"].constant is False
    assert column(result.table, "is_tcp") == [0.0, 0.0, 0.0]
    assert list(result.table.features().columns) == FEATURES
    assert result.counts["constant_features"] == ("is_tcp",)
    assert result.counts["constant_mismatch"] == {"is_tcp": 0}


def test_d34_a_feature_below_the_std_threshold_is_constant() -> None:
    # 0.1 three times: the floating mean may differ from 0.1 by a few ulps.
    table = make_table(TRAIN_DURATION, [0.1, 0.1, 0.1], [1.0, 1.0, 1.0])
    fitted = scaler.fit(table)
    result = fitted.transform(table)
    assert fitted.values["down_up_ratio"].constant is True
    assert column(result.table, "down_up_ratio") == [0.0, 0.0, 0.0]
    assert result.counts["constant_mismatch"] == {"down_up_ratio": 0, "is_tcp": 0}


def test_d34_at_transform_values_that_differ_from_the_constant_are_counted() -> None:
    fitted = scaler.fit(training_table())
    result = fitted.transform(make_table(TRAIN_DURATION, TRAIN_RATIO, [1.0, 0.0, 0.0]))
    assert column(result.table, "is_tcp") == [0.0, 0.0, 0.0]
    assert result.counts["constant_mismatch"] == {"is_tcp": 2}


def test_d35_a_value_outside_the_training_range_is_not_clipped_and_is_counted() -> None:
    fitted = scaler.fit(training_table())
    table = make_table(
        duration=[math.expm1(9.0), math.expm1(1.0), math.expm1(5.0)],
        down_up_ratio=[-1.0, 1.0, 5.0],
        is_tcp=[1.0, 1.0, 1.0],
    )
    result = fitted.transform(table)
    assert column(result.table, "duration")[0] == pytest.approx(7 / STD)
    assert column(result.table, "down_up_ratio")[0] == pytest.approx(-3 / STD)
    assert result.counts["out_of_range"] == {
        "duration": 1,
        "down_up_ratio": 1,
        "is_tcp": 0,
    }


def test_d35_the_training_table_has_no_out_of_range_value() -> None:
    result = scaler.fit(training_table()).transform(training_table())
    assert result.counts["out_of_range"] == dict.fromkeys(FEATURES, 0)
    assert result.warnings == ()


def test_min_and_max_are_taken_after_the_log_step() -> None:
    fitted = scaler.fit(training_table())
    assert fitted.values["duration"].minimum == pytest.approx(0.0)
    assert fitted.values["duration"].maximum == pytest.approx(5.0)
    assert fitted.values["down_up_ratio"].minimum == 0.0
    assert fitted.values["down_up_ratio"].maximum == 5.0


def test_d36_a_fitted_scaler_cannot_be_fitted_again() -> None:
    first = scaler.fit(training_table())
    before = first.transform(training_table())
    second = scaler.fit(make_table([1.0, 2.0], [3.0, 4.0], [0.0, 1.0]))
    assert list(second.values) == FEATURES
    assert list(first.values) == FEATURES
    after = first.transform(training_table())
    pd.testing.assert_frame_equal(after.table.to_frame(), before.table.to_frame())
    assert not hasattr(first, "fit")
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.values = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.values["duration"] = first.values["is_tcp"]  # type: ignore[index]


def test_a_promoted_destination_port_is_scaled_linearly_with_a_warning() -> None:
    frame = pd.DataFrame(
        {
            "dst_port": pd.Series([0, 1, 5], dtype="Int64"),
            "duration": pd.Series(TRAIN_DURATION, dtype=float),
        }
    )
    table = FlowTable(frame, {"dst_port": Role.FEATURE, "duration": Role.FEATURE})
    fitted = scaler.fit(table)
    result = fitted.transform(table)
    assert fitted.values["dst_port"].log is False
    assert column(result.table, "dst_port") == pytest.approx(EXPECTED)
    assert len(result.warnings) == 1
    assert "dst_port" in result.warnings[0]
    assert "categories" in result.warnings[0]


def test_identity_and_label_columns_are_untouched() -> None:
    table = training_table()
    result = scaler.fit(table).transform(table)
    for name in ("src_ip", "label"):
        pd.testing.assert_series_equal(
            result.table.to_frame()[name], table.to_frame()[name]
        )
    assert result.table.roles == table.roles


def test_scaler_json_has_the_documented_self_sufficient_shape(tmp_path: Path) -> None:
    scaler.fit(training_table()).save(tmp_path)
    saved = json.loads((tmp_path / "scaler.json").read_text(encoding="utf-8"))
    assert list(saved) == FEATURES
    assert saved["down_up_ratio"] == {
        "log": False,
        "mean": 2.0,
        "std": pytest.approx(STD),
        "min": 0.0,
        "max": 5.0,
        "constant": False,
    }
    assert saved["is_tcp"] == {
        "log": False,
        "mean": 1.0,
        "std": 0.0,
        "min": 1.0,
        "max": 1.0,
        "constant": True,
    }
    assert saved["duration"]["log"] is True
    assert saved["duration"]["max"] == pytest.approx(5.0)


def test_scaler_json_round_trips(tmp_path: Path) -> None:
    fitted = scaler.fit(training_table())
    fitted.save(tmp_path)
    loaded = scaler.load(tmp_path, {})
    assert loaded.values == fitted.values
    table = make_table([0.3, 1e9, 7.0], [1 / 3, -2.0, 1e300], [1.0, 0.0, 1.0])
    pd.testing.assert_frame_equal(
        loaded.transform(table).table.to_frame(),
        fitted.transform(table).table.to_frame(),
    )


def test_transform_uses_only_scaler_json_never_the_flow_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fitted = scaler.fit(training_table())
    fitted.save(tmp_path)
    expected = fitted.transform(training_table()).table.to_frame()

    def changed_record(name: str) -> Scaling:
        raise AssertionError(f"transform consulted the flow record for {name!r}")

    monkeypatch.setattr(record, "feature_scaling", changed_record)
    loaded = scaler.load(tmp_path, {})
    for result in (
        fitted.transform(training_table()),
        loaded.transform(training_table()),
    ):
        pd.testing.assert_frame_equal(result.table.to_frame(), expected)


# Rejection branches


def test_transform_refuses_a_table_missing_a_feature_column() -> None:
    fitted = scaler.fit(training_table())
    frame = training_table().to_frame().drop(columns="down_up_ratio")
    roles = {name: role for name, role in ROLES.items() if name != "down_up_ratio"}
    with pytest.raises(ArtefactsError, match="down_up_ratio"):
        fitted.transform(FlowTable(frame, roles))


def test_transform_refuses_a_table_with_an_extra_feature_column() -> None:
    fitted = scaler.fit(training_table())
    frame = training_table().to_frame()
    frame["packets_per_s"] = 1.0
    with pytest.raises(ArtefactsError, match="packets_per_s"):
        fitted.transform(FlowTable(frame, {**ROLES, "packets_per_s": Role.FEATURE}))


def test_load_refuses_a_folder_without_scaler_json(tmp_path: Path) -> None:
    with pytest.raises(ArtefactsError, match="scaler.json"):
        scaler.load(tmp_path, {})
