"""Tests of the reader protocol, the reader list and input detection in netgen.readers."""

from pathlib import Path

import pandas as pd
import pytest

from netgen.errors import InputError
from netgen.readers import READERS, Reader, capture, detect, flow_export, reader_named
from netgen.record import FlowSettings, Role
from netgen.table import FlowTable, TableResult

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
REAL = SAMPLES / "real"


class DummyReader:
    name = "dummy"

    def sniff(self, path: Path) -> bool:
        return path.suffix == ".dummy"

    def read(
        self, path: Path, flow_settings: FlowSettings, label_col: str = "Label"
    ) -> TableResult:
        table = FlowTable(
            pd.DataFrame({"duration": [flow_settings.idle_timeout]}),
            {"duration": Role.FEATURE},
        )
        return TableResult(table, {"rows": 1}, ())


class SniffsEverything(DummyReader):
    def __init__(self, name: str) -> None:
        self.name = name

    def sniff(self, path: Path) -> bool:
        return True


class SniffRaises(DummyReader):
    name = "raises"

    def sniff(self, path: Path) -> bool:
        raise AssertionError("sniff must not be called")


def test_a_dummy_reader_conforms_to_the_protocol() -> None:
    reader: Reader = DummyReader()
    assert isinstance(reader, Reader)
    assert reader.sniff(Path("flows.dummy"))
    result = reader.read(Path("flows.dummy"), FlowSettings())
    assert result.counts == {"rows": 1}


def test_an_object_without_read_does_not_conform() -> None:
    class NoRead:
        name = "broken"

        def sniff(self, path: Path) -> bool:
            return False

    assert not isinstance(NoRead(), Reader)


def test_the_reader_list_is_capture_then_flow_export() -> None:
    assert READERS == (capture, flow_export)
    assert all(isinstance(reader, Reader) for reader in READERS)


@pytest.mark.parametrize(
    "path",
    [
        REAL / "one_http_stream_fin.pcap",
        REAL / "pcapng-example.pcapng",
    ],
    ids=lambda path: path.name,
)
def test_a_capture_is_detected_as_the_capture_reader(path: Path) -> None:
    assert detect(path) is capture


@pytest.mark.parametrize(
    "path",
    [REAL / "cicids2017_excerpt.csv", SAMPLES / "demo_flows.csv"],
    ids=lambda path: path.name,
)
def test_a_cicflowmeter_csv_is_detected_as_the_flow_export_reader(path: Path) -> None:
    assert detect(path) is flow_export


def test_a_new_reader_at_the_end_of_the_list_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "flows.dummy"
    path.write_text("dummy flows\n")
    dummy = DummyReader()
    assert detect(path, (*READERS, dummy)) is dummy


def test_the_earlier_reader_wins_when_two_readers_sniff_true(tmp_path: Path) -> None:
    path = tmp_path / "flows.bin"
    path.write_bytes(b"\x00")
    first, second = SniffsEverything("first"), SniffsEverything("second")
    assert detect(path, (first, second)) is first
    assert detect(path, (second, first)) is second


@pytest.mark.parametrize(
    ("file_name", "content"),
    [
        ("notes.txt", b"plain text, not traffic\n"),
        ("other.csv", b"Flow Duration,Destination Port,Label\n1,80,BENIGN\n"),
        ("empty.pcap", b""),
    ],
    ids=["text file", "csv with the wrong header", "empty file"],
)
def test_unknown_input_raises_an_error_naming_both_formats(
    tmp_path: Path, file_name: str, content: bytes
) -> None:
    path = tmp_path / file_name
    path.write_bytes(content)
    with pytest.raises(InputError) as raised:
        detect(path)
    message = str(raised.value)
    assert str(path) in message
    for part in ("capture", "pcap", "pcapng", "CICFlowMeter CSV flow export"):
        assert part in message
    assert "--input-format" in message


def test_a_missing_path_raises_an_error_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.pcap"
    with pytest.raises(InputError, match="missing.pcap"):
        detect(path)


@pytest.mark.parametrize("name", ["capture", "flow-export"])
def test_a_known_name_returns_its_reader(name: str) -> None:
    reader = reader_named(name)
    assert reader.name == name
    assert reader in READERS


def test_a_named_format_skips_detection() -> None:
    reader = SniffRaises()
    assert reader_named("raises", (DummyReader(), reader)) is reader


def test_an_unknown_name_raises_an_error_listing_the_names() -> None:
    with pytest.raises(InputError) as raised:
        reader_named("zeek")
    message = str(raised.value)
    assert "'zeek'" in message
    assert "capture, flow-export" in message
