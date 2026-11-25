"""Tests of the flow-export reader (netgen.readers.flow_export): CICFlowMeter CSV."""

import csv
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from netgen import record
from netgen.errors import InputError
from netgen.readers import Reader, flow_export
from netgen.record import FlowSettings, Role
from netgen.table import TableResult, check_record

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO = SAMPLES / "demo_flows.csv"
EXCERPT = SAMPLES / "real" / "cicids2017_excerpt.csv"

# The record field filled from each CICFlowMeter column, and the factor that
# converts the column's unit: CICFlowMeter writes times in microseconds.
MICROSECONDS = 1e-6
MAPPING: dict[str, tuple[str, float]] = {
    "duration": ("Flow Duration", MICROSECONDS),
    "fwd_packets": ("Total Fwd Packets", 1),
    "bwd_packets": ("Total Backward Packets", 1),
    "fwd_bytes": ("Total Length of Fwd Packets", 1),
    "bwd_bytes": ("Total Length of Bwd Packets", 1),
    **{
        f"{d.lower()}_len_{s.lower()}": (f"{d} Packet Length {s}", 1)
        for d in ("Fwd", "Bwd")
        for s in ("Min", "Max", "Mean", "Std")
    },
    "bytes_per_s": ("Flow Bytes/s", 1),
    "packets_per_s": ("Flow Packets/s", 1),
    "fwd_packets_per_s": ("Fwd Packets/s", 1),
    "bwd_packets_per_s": ("Bwd Packets/s", 1),
    **{
        f"iat_{s.lower()}": (f"Flow IAT {s}", MICROSECONDS)
        for s in ("Mean", "Std", "Max", "Min")
    },
    "fwd_iat_total": ("Fwd IAT Total", MICROSECONDS),
    "bwd_iat_total": ("Bwd IAT Total", MICROSECONDS),
    "fwd_header_bytes": ("Fwd Header Length", 1),
    "bwd_header_bytes": ("Bwd Header Length", 1),
    "down_up_ratio": ("Down/Up Ratio", 1),
}
INDICATORS = {"is_tcp": "tcp", "is_udp": "udp", "is_icmp": "icmp"}
# Missing over zero duration (D28).
RATES = ("bytes_per_s", "packets_per_s", "fwd_packets_per_s", "bwd_packets_per_s")


def demo() -> tuple[list[str], list[list[str]]]:
    with DEMO.open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    return rows[0], rows[1:]


def position(header: Sequence[str], name: str) -> int:
    return [column.strip() for column in header].index(name)


def make_csv(
    tmp_path: Path,
    *,
    rows: int = 3,
    values: Mapping[str, Sequence[str]] | None = None,
    drop: Sequence[str] = (),
    rename: Mapping[str, str] | None = None,
    encoding: str = "utf-8",
) -> Path:
    """A CSV of the first `rows` demo flows.

    `values` sets a column (its first copy) for the first rows, one value per
    row. `drop` removes every copy of a column. `rename` gives a column a new
    raw header name.
    """
    header, body = demo()
    body = [list(row) for row in body[:rows]]
    for name, column_values in (values or {}).items():
        index = position(header, name)
        for row, value in zip(body, column_values, strict=False):
            row[index] = value
    for name, new in (rename or {}).items():
        header[position(header, name)] = new
    keep = [i for i, column in enumerate(header) if column.strip() not in drop]
    path = tmp_path / "flows.csv"
    with path.open("w", newline="", encoding=encoding) as file:
        writer = csv.writer(file)
        writer.writerow([header[i] for i in keep])
        writer.writerows([[row[i] for i in keep] for row in body])
    return path


def read(path: Path, label_col: str = "Label") -> TableResult:
    return flow_export.read(path, FlowSettings(), label_col=label_col)


def frame(path: Path) -> pd.DataFrame:
    return read(path).table.to_frame()


# The reader protocol and sniff


def test_the_reader_conforms_to_the_protocol() -> None:
    reader: Reader = flow_export
    assert isinstance(reader, Reader)
    assert reader.name == "flow-export"


def test_the_demo_and_the_excerpt_are_sniffed() -> None:
    assert flow_export.sniff(DEMO)
    assert flow_export.sniff(EXCERPT)


def test_d17_a_csv_with_only_label_bytes_proto_is_not_sniffed(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("label,bytes,proto\nBENIGN,100,6\n", encoding="utf-8")
    assert not flow_export.sniff(path)


def test_d17_a_header_missing_one_distinctive_name_is_not_sniffed(
    tmp_path: Path,
) -> None:
    assert not flow_export.sniff(make_csv(tmp_path, drop=["Flow IAT Mean"]))


@pytest.mark.parametrize(
    "content",
    [
        b"",
        bytes.fromhex("d4c3b2a1") + bytes(60),
        b"\x81\x8d\x8f\x90\x9d\n",
        b"\x1f\x8b\x08abc\rdef\n",
    ],
    ids=["empty", "pcap", "not-text", "carriage-return"],
)
def test_sniff_is_false_for_a_file_that_is_not_a_flow_export(
    tmp_path: Path, content: bytes
) -> None:
    path = tmp_path / "input"
    path.write_bytes(content)
    assert not flow_export.sniff(path)


def test_a_flow_export_with_carriage_return_line_endings_is_sniffed_and_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "flows.csv"
    path.write_bytes(make_csv(tmp_path).read_bytes().replace(b"\r\n", b"\r"))
    assert flow_export.sniff(path)
    assert len(frame(path)) == 3


def test_sniff_is_false_for_a_missing_file_or_a_folder(tmp_path: Path) -> None:
    assert not flow_export.sniff(tmp_path / "missing.csv")
    assert not flow_export.sniff(tmp_path)


# Carried defects


def test_d17_reading_an_unrecognised_header_fails(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("label,bytes,proto\nBENIGN,100,6\n", encoding="utf-8")
    with pytest.raises(InputError, match="not a CICFlowMeter header"):
        read(path)


def test_d22_header_spaces_are_stripped(tmp_path: Path) -> None:
    path = make_csv(
        tmp_path,
        values={"Total Fwd Packets": ["7"]},
        rename={"Total Fwd Packets": "  Total Fwd Packets  "},
    )
    assert frame(path)["fwd_packets"].iloc[0] == 7


def test_d22_a_duplicate_header_keeps_its_first_copy_and_warns(
    tmp_path: Path,
) -> None:
    header, _ = demo()
    second = [
        i for i, column in enumerate(header) if column.strip() == "Fwd Header Length"
    ][1]
    path = make_csv(tmp_path, values={"Fwd Header Length": ["40"]})
    lines = path.read_text(encoding="utf-8").splitlines()
    first_row = next(csv.reader([lines[1]]))
    first_row[second] = "999"
    lines[1] = ",".join(first_row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = read(path)
    assert result.table.to_frame()["fwd_header_bytes"].iloc[0] == 40
    duplicate = [warning for warning in result.warnings if "repeats" in warning]
    assert len(duplicate) == 1 and "'Fwd Header Length'" in duplicate[0]


def test_d23_a_cp1252_file_is_read_and_records_cp1252(tmp_path: Path) -> None:
    label = "Web Attack – Brute Force"
    path = make_csv(tmp_path, values={"Label": [label]}, encoding="cp1252")
    result = read(path)
    assert result.counts["encoding"] == "cp1252"
    assert result.table.to_frame()["label"].iloc[0] == label


def test_d23_a_utf8_file_records_utf8(tmp_path: Path) -> None:
    label = "Web Attack – Brute Force"
    path = make_csv(tmp_path, values={"Label": [label]}, encoding="utf-8")
    result = read(path)
    assert result.counts["encoding"] == "utf-8"
    assert result.table.to_frame()["label"].iloc[0] == label


def test_d23_the_excerpt_is_cp1252() -> None:
    assert read(EXCERPT).counts["encoding"] == "cp1252"


def test_a_file_that_is_neither_utf8_nor_cp1252_fails(tmp_path: Path) -> None:
    path = make_csv(tmp_path)
    path.write_bytes(path.read_bytes() + b"\x81\n")
    with pytest.raises(InputError, match="neither UTF-8 nor cp1252"):
        read(path)


@pytest.mark.parametrize(
    ("value", "protocol"),
    [
        ("6.0", "tcp"),
        ("6", "tcp"),
        ("17", "udp"),
        ("1", "icmp"),
        ("58", "icmp"),
        ("0", "other"),
        ("99", "other"),
        ("", "other"),
    ],
)
def test_d24_protocol_numbers_map_to_names(
    tmp_path: Path, value: str, protocol: str
) -> None:
    row = frame(make_csv(tmp_path, rows=1, values={"Protocol": [value]})).iloc[0]
    assert row["protocol"] == protocol
    for indicator, name in INDICATORS.items():
        assert row[indicator] == int(protocol == name)


def test_ports_are_missing_for_icmp_and_other(tmp_path: Path) -> None:
    path = make_csv(tmp_path, values={"Protocol": ["6", "1", "99"]})
    table = frame(path)
    assert table["src_port"].notna().iloc[0] and table["dst_port"].notna().iloc[0]
    assert table["src_port"].isna().iloc[1:].all()
    assert table["dst_port"].isna().iloc[1:].all()


def test_d25_timestamps_are_day_first_with_or_without_seconds(
    tmp_path: Path,
) -> None:
    path = make_csv(
        tmp_path, values={"Timestamp": ["03/07/2017 09:03:22", "3/7/2017 8:55"]}
    )
    start = frame(path)["start_time"]
    assert start.iloc[0] == pd.Timestamp(2017, 7, 3, 9, 3, 22)
    assert start.iloc[1] == pd.Timestamp(2017, 7, 3, 8, 55)
    assert start.dt.tz is None


@pytest.mark.parametrize("value", ["2017-07-13 09:03:22", "32/07/2017 09:03", ""])
def test_d25_an_unparseable_timestamp_fails_the_read_naming_the_line(
    tmp_path: Path, value: str
) -> None:
    path = make_csv(
        tmp_path,
        values={"Timestamp": ["13/07/2017 09:03:22", "13/07/2017 09:03", value]},
    )
    with pytest.raises(InputError, match="line 4"):
        read(path)


def test_d07_microseconds_become_seconds(tmp_path: Path) -> None:
    values = {
        column: ["1632511"]
        for column, factor in MAPPING.values()
        if factor == MICROSECONDS
    }
    row = frame(make_csv(tmp_path, rows=1, values=values)).iloc[0]
    for field, (_, factor) in MAPPING.items():
        if factor == MICROSECONDS:
            assert math.isclose(row[field], 1.632511), field


def test_d21_negative_and_infinite_become_missing(tmp_path: Path) -> None:
    path = make_csv(
        tmp_path,
        rows=4,
        values={
            "Flow Duration": ["1000000", "1000000", "-1", "Infinity"],
            "Flow IAT Min": ["-12"],
            "Fwd Header Length": ["-40"],
            "Total Length of Bwd Packets": ["-1"],
            "Flow Bytes/s": ["Infinity", "-5"],
            "Flow Packets/s": ["NaN"],
        },
    )
    table = frame(path)
    first, second = table.iloc[0], table.iloc[1]
    for field in (
        "iat_min",
        "fwd_header_bytes",
        "bwd_bytes",
        "bytes_per_s",
        "packets_per_s",
    ):
        assert pd.isna(first[field]), field
    assert pd.isna(second["bytes_per_s"])
    assert table["duration"].iloc[2:].isna().all()
    assert not np.isinf(table["bytes_per_s"].to_numpy(dtype=float)).any()


def test_d28_a_rate_over_zero_duration_is_missing(tmp_path: Path) -> None:
    rates = {MAPPING[field][0]: ["5", "5"] for field in RATES}
    path = make_csv(tmp_path, values={"Flow Duration": ["0", "1000000"], **rates})
    table = frame(path)
    for field in RATES:
        assert pd.isna(table[field].iloc[0]), field
        assert table[field].iloc[1] == 5, field


def test_d29_a_ratio_over_zero_forward_packets_is_missing(tmp_path: Path) -> None:
    path = make_csv(
        tmp_path,
        values={"Total Fwd Packets": ["0", "2"], "Down/Up Ratio": ["0", "1"]},
    )
    ratio = frame(path)["down_up_ratio"]
    assert pd.isna(ratio.iloc[0])
    assert ratio.iloc[1] == 1


# An empty or negative denominator is missing (D21), so the rate or ratio over
# it is missing too, not taken from the file.
MISSING = pytest.mark.parametrize("missing", ["", "-1"], ids=["empty", "negative"])


@MISSING
def test_d28_a_rate_over_a_missing_duration_is_missing(
    tmp_path: Path, missing: str
) -> None:
    rates = {MAPPING[field][0]: ["5", "5"] for field in RATES}
    path = make_csv(tmp_path, values={"Flow Duration": [missing, "1000000"], **rates})
    table = frame(path)
    assert pd.isna(table["duration"].iloc[0])
    for field in RATES:
        assert pd.isna(table[field].iloc[0]), field
        assert table[field].iloc[1] == 5, field


@MISSING
def test_d29_a_ratio_over_missing_forward_packets_is_missing(
    tmp_path: Path, missing: str
) -> None:
    path = make_csv(
        tmp_path,
        values={"Total Fwd Packets": [missing, "2"], "Down/Up Ratio": ["1", "1"]},
    )
    table = frame(path)
    assert pd.isna(table["fwd_packets"].iloc[0])
    assert pd.isna(table["down_up_ratio"].iloc[0])
    assert table["down_up_ratio"].iloc[1] == 1


# Scenarios


@pytest.mark.parametrize("path", [DEMO, EXCERPT], ids=["demo", "excerpt"])
def test_the_committed_samples_read_into_valid_tables(path: Path) -> None:
    result = read(path)
    check_record(result.table)
    expected = len(pd.read_csv(path, encoding="cp1252"))
    assert len(result.table.to_frame()) == expected
    assert result.table.roles["label"] is Role.LABEL


def test_the_table_has_the_record_dtypes() -> None:
    table = frame(DEMO)
    for field in record.FIELDS:
        if field.role is Role.DERIVED:
            continue
        dtype = table[field.name].dtype
        # Integer fields are nullable: a negative size becomes missing (D21).
        if field.type in (record.FieldType.NULLABLE_INTEGER, record.FieldType.INTEGER):
            assert str(dtype) == "Int64", field.name
        elif field.type is record.FieldType.DECIMAL:
            assert dtype == np.float64, field.name
        elif field.type is record.FieldType.INDICATOR:
            assert dtype == np.int64, field.name


def test_counts_and_warnings_reach_the_result() -> None:
    result = read(DEMO)
    assert result.counts == {"encoding": "utf-8"}
    assert any("repeats 'Fwd Header Length'" in warning for warning in result.warnings)
    assert any("Flow ID" in warning for warning in result.warnings)


def test_a_file_without_the_label_column_has_no_label(tmp_path: Path) -> None:
    result = read(make_csv(tmp_path, drop=["Label"]))
    check_record(result.table)
    assert "label" not in result.table.roles


def test_a_non_default_label_col_becomes_the_label(tmp_path: Path) -> None:
    path = make_csv(
        tmp_path, values={"Label": ["DDoS", "BENIGN"]}, rename={"Label": "Attack"}
    )
    result = read(path, label_col="Attack")
    assert list(result.table.to_frame()["label"].iloc[:2]) == ["DDoS", "BENIGN"]
    assert result.table.roles["label"] is Role.LABEL


def test_the_default_label_col_ignores_a_differently_named_label(
    tmp_path: Path,
) -> None:
    result = read(make_csv(tmp_path, rename={"Label": "Attack"}))
    assert "label" not in result.table.roles
    assert any("Attack" in warning for warning in result.warnings)


def test_unrecognised_columns_are_dropped_with_one_warning_listing_them() -> None:
    result = read(DEMO)
    dropped = [warning for warning in result.warnings if "Flow ID" in warning]
    assert len(dropped) == 1
    for name in (
        "Fwd Avg Bytes/Bulk",
        "Subflow Fwd Packets",
        "Active Mean",
        "Init_Win_bytes_forward",
        "SYN Flag Count",
    ):
        assert name in dropped[0], name
    assert set(result.table.roles) == {
        field.name for field in record.FIELDS if field.role is not Role.DERIVED
    }


def test_a_header_with_no_rows_gives_an_empty_valid_table(tmp_path: Path) -> None:
    result = read(make_csv(tmp_path, rows=0))
    check_record(result.table)
    assert len(result.table.to_frame()) == 0


def test_every_mapped_feature_equals_the_excerpt_after_unit_conversion() -> None:
    raw = pd.read_csv(EXCERPT, encoding="cp1252")
    raw.columns = [column.strip() for column in raw.columns]
    table = frame(EXCERPT)
    features = {field.name for field in record.fields_by_role(Role.FEATURE)}
    assert set(MAPPING) | set(INDICATORS) == features
    for field, (column, factor) in MAPPING.items():
        values = raw[column].astype(float)
        expected = (values * factor).where(np.isfinite(values) & (values >= 0))
        if field in RATES:
            expected = expected.where(raw["Flow Duration"] > 0)
        if field == "down_up_ratio":
            expected = expected.where(raw["Total Fwd Packets"] > 0)
        actual = table[field].to_numpy(dtype=float, na_value=np.nan)
        np.testing.assert_allclose(actual, expected.to_numpy(), err_msg=field)
    protocols = raw["Protocol"].map({6: "tcp", 17: "udp", 1: "icmp"}).fillna("other")
    assert list(table["protocol"]) == list(protocols)
    for indicator, name in INDICATORS.items():
        assert list(table[indicator]) == list((protocols == name).astype(int))
    assert list(table["src_ip"]) == list(raw["Source IP"])
    assert list(table["dst_ip"]) == list(raw["Destination IP"])
    tcp_or_udp = protocols.isin(["tcp", "udp"])
    assert list(table["dst_port"][tcp_or_udp]) == list(
        raw["Destination Port"][tcp_or_udp]
    )
    assert list(table["start_time"]) == list(
        pd.to_datetime(raw["Timestamp"], dayfirst=True, format="mixed")
    )
    assert list(table["label"]) == list(raw["Label"])


# Rejection branches


def test_a_missing_required_column_fails_naming_it(tmp_path: Path) -> None:
    path = make_csv(tmp_path, drop=["Timestamp", "Bwd IAT Total"])
    with pytest.raises(InputError, match="Timestamp") as error:
        read(path)
    assert "Bwd IAT Total" in str(error.value)


def test_a_missing_file_fails_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.csv"
    with pytest.raises(InputError, match="missing.csv"):
        read(path)


@pytest.mark.parametrize("row", [1, 3], ids=["first-row", "later-row"])
def test_a_row_with_extra_fields_is_read_by_position(tmp_path: Path, row: int) -> None:
    path = make_csv(tmp_path)
    expected = frame(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[row] += ",extra,9"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pd.testing.assert_frame_equal(frame(path), expected)


def test_a_quote_left_open_fails(tmp_path: Path) -> None:
    path = make_csv(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + '"192.0.2.1,1\n')
    with pytest.raises(InputError, match="not a valid CSV"):
        read(path)
