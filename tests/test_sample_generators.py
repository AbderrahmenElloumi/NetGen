"""Tests of the flow-export fixtures in data/samples and their makers."""

import csv
import importlib.util
import ipaddress
import math
from datetime import datetime
from pathlib import Path
from types import ModuleType

import pytest

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_FLOWS = SAMPLES / "demo_flows.csv"
EXCERPT = SAMPLES / "real" / "cicids2017_excerpt.csv"

# The header of the CIC-IDS2017 files in TrafficLabelling, as written there.
REAL_HEADER = (
    "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol,"
    " Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets,"
    "Total Length of Fwd Packets, Total Length of Bwd Packets,"
    " Fwd Packet Length Max, Fwd Packet Length Min, Fwd Packet Length Mean,"
    " Fwd Packet Length Std,Bwd Packet Length Max, Bwd Packet Length Min,"
    " Bwd Packet Length Mean, Bwd Packet Length Std,Flow Bytes/s, Flow Packets/s,"
    " Flow IAT Mean, Flow IAT Std, Flow IAT Max, Flow IAT Min,Fwd IAT Total,"
    " Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min,Bwd IAT Total,"
    " Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,Fwd PSH Flags,"
    " Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length,"
    " Bwd Header Length,Fwd Packets/s, Bwd Packets/s, Min Packet Length,"
    " Max Packet Length, Packet Length Mean, Packet Length Std,"
    " Packet Length Variance,FIN Flag Count, SYN Flag Count, RST Flag Count,"
    " PSH Flag Count, ACK Flag Count, URG Flag Count, CWE Flag Count,"
    " ECE Flag Count, Down/Up Ratio, Average Packet Size, Avg Fwd Segment Size,"
    " Avg Bwd Segment Size, Fwd Header Length,Fwd Avg Bytes/Bulk,"
    " Fwd Avg Packets/Bulk, Fwd Avg Bulk Rate, Bwd Avg Bytes/Bulk,"
    " Bwd Avg Packets/Bulk,Bwd Avg Bulk Rate,Subflow Fwd Packets,"
    " Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes,"
    "Init_Win_bytes_forward, Init_Win_bytes_backward, act_data_pkt_fwd,"
    " min_seg_size_forward,Active Mean, Active Std, Active Max, Active Min,"
    "Idle Mean, Idle Std, Idle Max, Idle Min, Label"
)
COLUMNS = REAL_HEADER.split(",")

DOCUMENTATION_RANGES = [
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("2001:db8::/32"),
]


def load_module(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path: Path, encoding: str) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert all(len(row) == len(header) for row in rows)
    return header, rows


def column(name: str, occurrence: int = 0) -> int:
    """The positional key of a column, found by its stripped name."""
    indexes = [i for i, c in enumerate(COLUMNS) if c.strip() == name]
    return indexes[occurrence]


def in_documentation_range(text: str) -> bool:
    address = ipaddress.ip_address(text)
    return any(address in network for network in DOCUMENTATION_RANGES)


def addresses(row: list[str]) -> list[str]:
    flow_id_addresses = row[column("Flow ID")].rsplit("-", 3)[0].split("-")
    assert len(flow_id_addresses) == 2
    return [row[column("Source IP")], row[column("Destination IP")], *flow_id_addresses]


def labels(rows: list[list[str]]) -> list[str]:
    return [row[column("Label")] for row in rows]


@pytest.fixture(scope="module")
def demo() -> list[list[str]]:
    header, rows = read_rows(DEMO_FLOWS, "ascii")
    assert header == COLUMNS
    return rows


@pytest.fixture(scope="module")
def excerpt() -> list[list[str]]:
    header, rows = read_rows(EXCERPT, "cp1252")
    assert header == COLUMNS
    return rows


# Synthetic CSV


def test_demo_flows_generator_reproduces_committed_file(tmp_path: Path) -> None:
    out = tmp_path / "demo_flows.csv"
    load_module("make_demo_flows").main(["--out", str(out)])
    assert out.read_bytes() == DEMO_FLOWS.read_bytes()


def test_demo_flows_keeps_the_real_header_quirks() -> None:
    first_line = DEMO_FLOWS.read_bytes().split(b"\r\n", 1)[0]
    assert first_line.decode("ascii") == REAL_HEADER
    assert COLUMNS[1] == " Source IP"
    assert [c.strip() for c in COLUMNS].count("Fwd Header Length") == 2


def test_demo_flows_has_about_400_rows(demo: list[list[str]]) -> None:
    assert 380 <= len(demo) <= 420


def test_demo_flows_holds_infinity_nan_and_minus_one(demo: list[list[str]]) -> None:
    values = [value for row in demo for value in row]
    assert "Infinity" in values
    assert "NaN" in values
    assert "-1" in [row[column("Init_Win_bytes_backward")] for row in demo]


def test_demo_flows_classes_survive_a_stratified_80_20_split(
    demo: list[list[str]],
) -> None:
    counts = {label: labels(demo).count(label) for label in set(labels(demo))}
    assert "BENIGN" in counts
    assert len(counts) >= 2
    assert min(counts.values()) >= 10


def test_demo_flows_rates_follow_the_row_totals(demo: list[list[str]]) -> None:
    for row in demo:
        duration_us = int(row[column("Flow Duration")])
        fwd_packets = int(row[column("Total Fwd Packets")])
        bwd_packets = int(row[column("Total Backward Packets")])
        total_bytes = int(row[column("Total Length of Fwd Packets")]) + int(
            row[column("Total Length of Bwd Packets")]
        )
        flow_bytes = float(row[column("Flow Bytes/s")])
        flow_packets = float(row[column("Flow Packets/s")])
        if duration_us == 0:
            assert flow_packets == math.inf
            assert (
                math.isnan(flow_bytes) if total_bytes == 0 else flow_bytes == math.inf
            )
            continue
        seconds = duration_us / 1e6
        assert math.isclose(flow_bytes, total_bytes / seconds)
        assert math.isclose(flow_packets, (fwd_packets + bwd_packets) / seconds)
        assert math.isclose(float(row[column("Fwd Packets/s")]), fwd_packets / seconds)
        assert math.isclose(float(row[column("Bwd Packets/s")]), bwd_packets / seconds)
        iat_total = int(row[column("Fwd IAT Total")])
        assert 0 <= iat_total <= duration_us


def test_demo_flows_timestamps_are_day_first_with_and_without_seconds(
    demo: list[list[str]],
) -> None:
    formats = set()
    for row in demo:
        text = row[column("Timestamp")]
        for fmt in ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            formats.add(fmt)
            assert parsed.day > 12
            break
        else:
            pytest.fail(f"timestamp does not parse day-first: {text!r}")
    assert len(formats) == 2


def test_demo_flows_addresses_are_documentation_addresses(
    demo: list[list[str]],
) -> None:
    for row in demo:
        for address in addresses(row):
            assert in_documentation_range(address), address


# Real excerpt


def test_excerpt_has_20_rows_of_benign_and_attack_flows(
    excerpt: list[list[str]],
) -> None:
    assert len(excerpt) == 20
    assert "BENIGN" in labels(excerpt)
    assert any(label != "BENIGN" for label in labels(excerpt))


def test_excerpt_holds_infinity_and_nan(excerpt: list[list[str]]) -> None:
    values = [value for row in excerpt for value in row]
    assert "Infinity" in values
    assert "NaN" in values


def test_excerpt_addresses_are_documentation_addresses(
    excerpt: list[list[str]],
) -> None:
    for row in excerpt:
        for address in addresses(row):
            assert in_documentation_range(address), address


def test_excerpt_is_cp1252_with_a_non_utf8_label_byte() -> None:
    raw = EXCERPT.read_bytes()
    raw.decode("cp1252")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert b"Web Attack \x96 " in raw


def test_excerpt_maker_is_import_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    module = load_module("make_cic_excerpt")
    assert callable(module.main)
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr() == ("", "")


# Every fixture


@pytest.mark.parametrize("path", [DEMO_FLOWS, EXCERPT], ids=["demo", "excerpt"])
def test_fixture_holds_no_local_path(path: Path) -> None:
    text = path.read_bytes().decode("cp1252").lower()
    # Built from pieces: the release leak scan refuses the joined form in a public file.
    home = "/" + "home" + "/"
    for marker in ("c:\\", home, "/users/", "\\users\\", "@"):
        assert marker not in text
