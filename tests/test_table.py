"""Tests of the FlowTable and the seam check in netgen.table."""

import importlib
import sys

import numpy as np
import pandas as pd
import pytest

from netgen import record
from netgen.errors import InputError, NetGenError
from netgen.record import Role
from netgen.table import FlowTable, TableResult, check_record, stratified_split

FEATURES = [field.name for field in record.fields_by_role(Role.FEATURE)]


def record_frame(rows: int = 3, *, numeric_label: bool = False) -> pd.DataFrame:
    """A frame holding every field of the flow record, with the record's dtypes."""
    frame = pd.DataFrame(
        {
            "src_ip": pd.Series(["192.0.2.1"] * rows, dtype="string"),
            "dst_ip": pd.Series(["198.51.100.1"] * rows, dtype="string"),
            "src_port": pd.Series([40000 + i for i in range(rows)], dtype="Int64"),
            "dst_port": pd.Series([443] * (rows - 1) + [None], dtype="Int64"),
            "protocol": pd.Series(["tcp"] * (rows - 1) + ["icmp"], dtype="string"),
            "start_time": pd.Series(
                pd.date_range("2017-07-07 03:30", periods=rows, freq="s")
            ),
            "label": pd.Series(range(rows))
            if numeric_label
            else pd.Series(["BENIGN"] * rows, dtype="string"),
        }
    )
    for offset, name in enumerate(FEATURES):
        frame[name] = np.arange(rows, dtype=float) + offset
    return frame


def record_table(
    *, numeric_label: bool = False, dst_port_as_feature: bool = False
) -> FlowTable:
    return FlowTable.from_frame(
        record_frame(numeric_label=numeric_label),
        dst_port_as_feature=dst_port_as_feature,
    )


# Named regression test (D31)


@pytest.mark.parametrize("numeric_label", [False, True])
def test_d31_identity_columns_never_reach_a_learning_stage(numeric_label: bool) -> None:
    table = record_table(numeric_label=numeric_label).with_derived(
        "anomaly_score", [0.1, 0.2, 0.3]
    )
    learned = list(table.features().columns)
    assert learned == FEATURES
    for name in (
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "start_time",
        "label",
        "anomaly_score",
    ):
        assert name not in learned

    promoted = record_table(numeric_label=numeric_label, dst_port_as_feature=True)
    assert list(promoted.features().columns) == ["dst_port", *FEATURES]
    assert "src_port" not in promoted.features().columns


# Scenario tests on roles


def test_derived_columns_enter_only_through_with_derived() -> None:
    frame = record_frame()
    frame["anomaly_score"] = 0.5
    with pytest.raises(NetGenError, match="with_derived"):
        FlowTable(frame, record.default_roles(frame.columns))


def test_derived_column_is_never_a_feature() -> None:
    table = record_table().with_derived("latent_0", np.array([1.0, 2.0, 3.0]))
    assert table.roles["latent_0"] is Role.DERIVED
    assert "latent_0" not in table.features().columns
    assert list(table.to_frame()["latent_0"]) == [1.0, 2.0, 3.0]


def test_with_derived_returns_a_new_table_and_leaves_the_old_one_unchanged() -> None:
    table = record_table()
    derived = table.with_derived("anomaly_score", [0.1, 0.2, 0.3])
    assert derived is not table
    assert "anomaly_score" not in table.roles
    assert "anomaly_score" not in table.to_frame().columns
    assert "anomaly_score" in derived.roles


@pytest.mark.parametrize("name", ["duration", "src_ip", "label"])
def test_with_derived_refuses_a_name_collision(name: str) -> None:
    with pytest.raises(NetGenError, match=name):
        record_table().with_derived(name, [0.0, 0.0, 0.0])


def test_with_derived_refuses_values_of_the_wrong_length() -> None:
    with pytest.raises(NetGenError, match="3 rows"):
        record_table().with_derived("anomaly_score", [0.0, 0.0])


def test_with_features_replaces_only_the_feature_columns() -> None:
    table = record_table().with_derived("anomaly_score", [0.1, 0.2, 0.3])
    values = table.features() * 10.0
    changed = table.with_features(values)
    assert changed is not table
    assert changed.roles == table.roles
    pd.testing.assert_frame_equal(changed.features(), values)
    others = [name for name in table.roles if name not in FEATURES]
    pd.testing.assert_frame_equal(changed.to_frame()[others], table.to_frame()[others])
    assert table.features().loc[1, "duration"] != changed.features().loc[1, "duration"]


def test_with_features_refuses_other_columns() -> None:
    table = record_table()
    with pytest.raises(NetGenError, match="duration"):
        table.with_features(table.features().drop(columns="duration"))


def test_with_features_refuses_values_of_the_wrong_length() -> None:
    table = record_table()
    with pytest.raises(NetGenError, match="3 rows"):
        table.with_features(table.features().iloc[:2])


def test_label_is_never_a_feature() -> None:
    table = record_table(numeric_label=True)
    assert table.roles["label"] is Role.LABEL
    assert "label" not in table.features().columns


def test_roles_are_not_name_based() -> None:
    frame = pd.DataFrame(
        {"label": [1.0, 2.0], "duration": pd.Series(["a", "b"], dtype="string")}
    )
    table = FlowTable(frame, {"label": Role.FEATURE, "duration": Role.LABEL})
    assert list(table.features().columns) == ["label"]
    assert table.roles["duration"] is Role.LABEL


def test_to_frame_gives_the_dataset_column_order() -> None:
    frame = record_frame()
    shuffled = frame[list(reversed(frame.columns))]
    table = FlowTable.from_frame(shuffled).with_derived(
        "anomaly_score", [0.0, 0.0, 0.0]
    )
    assert list(table.to_frame().columns) == [*frame.columns, "anomaly_score"]


def test_the_table_is_frozen() -> None:
    table = record_table()
    with pytest.raises(AttributeError):
        table.roles = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        table.roles["duration"] = Role.LABEL  # type: ignore[index]


def test_the_table_keeps_its_own_copy_of_the_frame() -> None:
    frame = record_frame()
    table = FlowTable.from_frame(frame)
    frame.loc[0, "duration"] = 99.0
    copy = table.to_frame()
    copy.loc[0, "duration"] = 99.0
    assert table.features().loc[0, "duration"] != 99.0


def test_table_result_holds_table_counts_and_warnings() -> None:
    result = TableResult(record_table(), {"dropped_non_ip": 2}, ("one warning",))
    assert result.counts["dropped_non_ip"] == 2
    assert result.warnings == ("one warning",)


# Rejection branches of the construction


def test_construction_rejects_a_column_without_a_role() -> None:
    frame = pd.DataFrame({"duration": [1.0], "extra": [2.0]})
    with pytest.raises(NetGenError, match="extra"):
        FlowTable(frame, {"duration": Role.FEATURE})


def test_from_frame_rejects_a_column_outside_the_record() -> None:
    frame = record_frame()
    frame["Flow ID"] = "x"
    with pytest.raises(NetGenError, match="Flow ID"):
        FlowTable.from_frame(frame)


def test_construction_rejects_a_role_for_a_missing_column() -> None:
    frame = pd.DataFrame({"duration": [1.0]})
    with pytest.raises(NetGenError, match="fwd_packets"):
        FlowTable(frame, {"duration": Role.FEATURE, "fwd_packets": Role.FEATURE})


def test_construction_rejects_two_labels() -> None:
    frame = pd.DataFrame({"label": ["a"], "attack": ["b"], "duration": [1.0]})
    with pytest.raises(NetGenError, match="label"):
        FlowTable(
            frame, {"label": Role.LABEL, "attack": Role.LABEL, "duration": Role.FEATURE}
        )


def test_construction_rejects_a_non_numeric_feature_column() -> None:
    frame = pd.DataFrame({"duration": ["slow"]})
    with pytest.raises(NetGenError, match="duration"):
        FlowTable(frame, {"duration": Role.FEATURE})


# The seam check against the record


def test_seam_check_accepts_a_reader_table() -> None:
    check_record(record_table())
    check_record(record_table(dst_port_as_feature=True))
    unlabelled = record_frame().drop(columns="label")
    check_record(FlowTable.from_frame(unlabelled))


def test_seam_check_accepts_derived_columns() -> None:
    check_record(record_table().with_derived("anomaly_score", [0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    "missing", ["src_ip", "start_time", "dst_port", "duration", "is_icmp"]
)
def test_seam_check_rejects_a_table_that_lacks_a_record_field(missing: str) -> None:
    frame = record_frame().drop(columns=missing)
    with pytest.raises(InputError, match=missing):
        check_record(FlowTable.from_frame(frame))


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("src_ip", pd.Series([1, 2, 3])),
        ("src_port", pd.Series([1.5, 2.5, 3.5])),
        ("start_time", pd.Series(["2017-07-07"] * 3, dtype="string")),
        (
            "start_time",
            pd.Series(pd.date_range("2017-07-07", periods=3, freq="s", tz="UTC")),
        ),
        ("label", pd.Series([1, 2, 3])),
    ],
)
def test_seam_check_rejects_a_field_of_the_wrong_type(
    name: str, values: pd.Series
) -> None:
    frame = record_frame()
    frame[name] = values
    with pytest.raises(InputError, match=name):
        check_record(FlowTable.from_frame(frame))


@pytest.mark.parametrize(
    ("name", "role"), [("src_port", Role.FEATURE), ("duration", Role.IDENTITY)]
)
def test_seam_check_rejects_a_role_other_than_the_record_allows(
    name: str, role: Role
) -> None:
    frame = record_frame()
    roles = dict(record.default_roles(frame.columns))
    roles[name] = role
    with pytest.raises(InputError, match=name):
        check_record(FlowTable(frame, roles))


def test_seam_check_rejects_a_column_outside_the_record() -> None:
    frame = record_frame()
    frame["extra"] = 1.0
    roles = dict(record.default_roles(frame.columns)) | {"extra": Role.FEATURE}
    with pytest.raises(InputError, match="extra"):
        check_record(FlowTable(frame, roles))


# The stratified split


def labelled_table(labels: list[str | None]) -> FlowTable:
    """A record table with one row per label; `None` is a row without a label."""
    frame = record_frame(len(labels))
    frame["label"] = pd.Series(labels, dtype="string")
    return FlowTable.from_frame(frame)


@pytest.mark.parametrize(
    ("rows", "test_rows"),
    [(1, 0), (2, 1), (3, 1), (4, 1), (5, 1), (8, 2), (10, 2), (13, 3), (100, 20)],
)
def test_split_test_count_is_a_rounded_fifth_with_a_minimum_of_one(
    rows: int, test_rows: int
) -> None:
    train, test = stratified_split(labelled_table(["A"] * rows), seed=1)
    assert len(test.to_frame()) == test_rows
    assert len(train.to_frame()) == rows - test_rows


def test_split_rejects_an_empty_table() -> None:
    with pytest.raises(InputError, match="no flows to split"):
        stratified_split(FlowTable.from_frame(record_frame(1).iloc[:0]), seed=1)


def split_labels(table: FlowTable) -> list[str | None]:
    return [None if pd.isna(value) else value for value in table.to_frame()["label"]]


def test_split_keeps_every_class_of_two_or_more_rows_in_both_splits() -> None:
    labels = ["A"] * 50 + ["B"] * 10 + ["C"] * 2 + ["D"]
    train, test = stratified_split(labelled_table(labels), seed=3)
    assert sorted(split_labels(test)) == ["A"] * 10 + ["B"] * 2 + ["C"]
    assert sorted(split_labels(train)) == ["A"] * 40 + ["B"] * 8 + ["C", "D"]


def test_split_puts_unlabelled_rows_in_a_group_of_their_own() -> None:
    labels: list[str | None] = ["A"] * 20 + [None] * 10
    train, test = stratified_split(labelled_table(labels), seed=3)
    assert split_labels(test).count(None) == 2
    assert split_labels(test).count("A") == 4
    assert split_labels(train).count(None) == 8
    assert split_labels(train).count("A") == 16


def test_split_of_a_table_without_a_label_column_is_one_group() -> None:
    table = FlowTable.from_frame(record_frame(13).drop(columns="label"))
    train, test = stratified_split(table, seed=3)
    assert (len(train.to_frame()), len(test.to_frame())) == (10, 3)
    assert "label" not in test.roles


def test_split_of_one_row_leaves_the_test_split_empty() -> None:
    table = labelled_table(["A"])
    train, test = stratified_split(table, seed=3)
    pd.testing.assert_frame_equal(train.to_frame(), table.to_frame())
    assert test.to_frame().empty
    assert test.roles == table.roles


def test_split_is_fixed_by_the_seed() -> None:
    table = labelled_table(["A"] * 60 + ["B"] * 40)
    first = stratified_split(table, seed=7)
    again = stratified_split(table, seed=7)
    other = stratified_split(table, seed=8)
    for part, same in zip(first, again, strict=True):
        pd.testing.assert_frame_equal(part.to_frame(), same.to_frame())
    assert list(first[1].to_frame().index) != list(other[1].to_frame().index)


def test_split_parts_are_disjoint_ordered_and_keep_the_roles() -> None:
    labels: list[str | None] = ["A"] * 30 + ["B"] * 12 + [None] * 9
    table = labelled_table(labels).with_derived("anomaly_score", np.arange(51.0))
    train, test = stratified_split(table, seed=5)
    train_rows = list(train.to_frame().index)
    test_rows = list(test.to_frame().index)
    assert train_rows == sorted(train_rows)
    assert test_rows == sorted(test_rows)
    assert set(train_rows).isdisjoint(test_rows)
    assert sorted(train_rows + test_rows) == list(range(51))
    assert train.roles == table.roles
    assert test.roles == table.roles
    pd.testing.assert_frame_equal(test.to_frame(), table.to_frame().loc[test_rows])


def test_split_leaves_the_input_table_unchanged() -> None:
    table = labelled_table(["A"] * 10 + ["B"] * 5)
    before = table.to_frame()
    stratified_split(table, seed=5)
    pd.testing.assert_frame_equal(table.to_frame(), before)


def test_split_imports_without_scikit_learn(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(sys.modules):
        if name == "sklearn" or name.startswith("sklearn."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.delitem(sys.modules, "netgen.table")
    hidden = importlib.import_module("netgen.table")
    frame = record_frame(10)
    train, test = hidden.stratified_split(hidden.FlowTable.from_frame(frame), seed=1)
    assert (len(train.to_frame()), len(test.to_frame())) == (8, 2)
