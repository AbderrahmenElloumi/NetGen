"""The flow-export reader: CICFlowMeter CSV in the layout of the CIC-IDS2017 data.

A flow export is already flows, so the flow settings are ignored. Header
names are stripped of spaces, and a repeated name keeps its first copy.
CICFlowMeter writes times in microseconds; the reader converts them to
seconds. A negative or infinite number becomes a missing value, and so do a
rate over zero duration and a ratio over zero forward packets.
"""

import csv
import io
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from netgen import record
from netgen.errors import InputError
from netgen.record import FieldType, FlowSettings, Role
from netgen.table import FlowTable, TableResult, check_record

name = "flow-export"

# The header must hold all of these to be recognised (D17).
DISTINCTIVE = (
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packet Length Max",
    "Flow IAT Mean",
)

IDENTITY = {
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Destination Port": "dst_port",
    "Protocol": "protocol",
    "Timestamp": "start_time",
}

MICROSECONDS = 1e-6

# Each feature's CICFlowMeter column and the factor that converts its unit.
FEATURES: Mapping[str, tuple[str, float]] = {
    "duration": ("Flow Duration", MICROSECONDS),
    "fwd_packets": ("Total Fwd Packets", 1),
    "bwd_packets": ("Total Backward Packets", 1),
    "fwd_bytes": ("Total Length of Fwd Packets", 1),
    "bwd_bytes": ("Total Length of Bwd Packets", 1),
    **{
        f"{direction.lower()}_len_{statistic.lower()}": (
            f"{direction} Packet Length {statistic}",
            1,
        )
        for direction in ("Fwd", "Bwd")
        for statistic in ("Min", "Max", "Mean", "Std")
    },
    "bytes_per_s": ("Flow Bytes/s", 1),
    "packets_per_s": ("Flow Packets/s", 1),
    "fwd_packets_per_s": ("Fwd Packets/s", 1),
    "bwd_packets_per_s": ("Bwd Packets/s", 1),
    **{
        f"iat_{statistic.lower()}": (f"Flow IAT {statistic}", MICROSECONDS)
        for statistic in ("Mean", "Std", "Max", "Min")
    },
    "fwd_iat_total": ("Fwd IAT Total", MICROSECONDS),
    "bwd_iat_total": ("Bwd IAT Total", MICROSECONDS),
    "fwd_header_bytes": ("Fwd Header Length", 1),
    "bwd_header_bytes": ("Bwd Header Length", 1),
    "down_up_ratio": ("Down/Up Ratio", 1),
}

REQUIRED = (*IDENTITY, *(column for column, _ in FEATURES.values()))

# Missing over zero duration, as the capture reader gives them (D28).
RATES = ("bytes_per_s", "packets_per_s", "fwd_packets_per_s", "bwd_packets_per_s")

# IANA protocol numbers; ICMPv6 counts as ICMP (D24).
PROTOCOLS = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp"}
INDICATORS = {"is_tcp": "tcp", "is_udp": "udp", "is_icmp": "icmp"}

# Day-first, with or without seconds (D25).
TIMESTAMP_FORMATS = ("%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M")

# Tried in order (D23). utf-8-sig also reads a file that starts with a byte-order mark.
ENCODINGS = (("utf-8-sig", "utf-8"), ("cp1252", "cp1252"))

SNIFF_BYTES = 65536


def sniff(path: Path) -> bool:
    """True when the first line is a header holding every distinctive
    CICFlowMeter name. Never raises."""
    try:
        with path.open("rb") as file:
            line = file.readline(SNIFF_BYTES)
    except OSError:
        return False
    for encoding, _ in ENCODINGS:
        try:
            text = line.decode(encoding)
        except UnicodeDecodeError:
            continue
        # newline="" ends the line at a bare carriage return too, as read does.
        return _recognised(_names(next(csv.reader(io.StringIO(text, newline="")), [])))
    return False


def read(
    path: Path, flow_settings: FlowSettings, label_col: str = "Label"
) -> TableResult:
    """The flows of a CICFlowMeter CSV, in the flow record, checked at the seam.

    `flow_settings` is ignored. The column `label_col` becomes `label`; without
    it the table has no label column. The counts hold `encoding`. Raises
    `InputError` naming the path when the file is missing, is neither UTF-8 nor
    cp1252, is not a CICFlowMeter CSV, lacks a column the record needs, or holds
    a timestamp that does not parse.
    """
    for encoding, reported in ENCODINGS:
        try:
            raw, warnings = _raw(path, encoding, label_col)
        except UnicodeDecodeError:
            continue
        table = FlowTable.from_frame(_record_frame(path, raw, label_col))
        check_record(table)
        return TableResult(table, {"encoding": reported}, warnings)
    raise InputError(
        f"{path}: the file is neither UTF-8 nor cp1252 text. Save it in one of those encodings."
    )


def _names(header: Iterable[str]) -> list[str]:
    return [column.strip() for column in header]


def _recognised(names: Iterable[str]) -> bool:
    return set(DISTINCTIVE) <= set(names)


def _raw(
    path: Path, encoding: str, label_col: str
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """The recognised columns as text, named by their stripped header, and the warnings.

    Raises UnicodeDecodeError when the file is not in `encoding`.
    """
    try:
        with path.open(newline="", encoding=encoding) as file:
            names = _names(next(csv.reader(file), []))
    except OSError as error:
        raise InputError(
            f"{path}: cannot open the flow export ({error.strerror})."
        ) from error
    if not _recognised(names):
        raise InputError(
            f"{path}: the header is not a CICFlowMeter header; it lacks "
            f"{[column for column in DISTINCTIVE if column not in names]}. "
            "NetGen reads CICFlowMeter CSV files and pcap or pcapng captures."
        )
    missing = [column for column in REQUIRED if column not in names]
    if missing:
        raise InputError(
            f"{path}: the CICFlowMeter header lacks the columns {missing}, which the flow record needs."
        )
    first: dict[str, int] = {}
    for index, column in enumerate(names):
        first.setdefault(column, index)
    warnings = [
        f"The header repeats {column!r}; the first copy is kept and the others are dropped."
        for column in first
        if names.count(column) > 1
    ]
    known = {*REQUIRED, label_col}
    dropped = [column for column in first if column not in known]
    if dropped:
        warnings.append(
            f"Dropped {len(dropped)} columns that the flow record does not hold: {', '.join(dropped)}."
        )
    keep = sorted(index for column, index in first.items() if column in known)
    try:
        # Positional names and index_col=False read every row by position: the
        # extra fields of a longer row are ignored, wherever the row is.
        raw = pd.read_csv(
            path,
            encoding=encoding,
            header=0,
            names=range(len(names)),
            index_col=False,
            usecols=keep,
            dtype=str,
        )
    except pd.errors.ParserError as error:
        raise InputError(f"{path}: not a valid CSV ({error}).") from error
    raw.columns = [names[index] for index in keep]
    return raw, tuple(warnings)


def _record_frame(path: Path, raw: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """The identity, label and feature fields of the record, from the raw text columns."""
    protocol = (
        pd.to_numeric(raw["Protocol"], errors="coerce")
        .map(PROTOCOLS)
        .fillna("other")
        .astype(object)
    )
    # Only TCP and UDP have ports; the record leaves them missing otherwise.
    has_ports = protocol.isin(["tcp", "udp"])
    columns: dict[str, pd.Series] = {
        "src_ip": raw["Source IP"].astype(object),
        "dst_ip": raw["Destination IP"].astype(object),
        "src_port": _port(raw["Source Port"], has_ports),
        "dst_port": _port(raw["Destination Port"], has_ports),
        "protocol": protocol,
        "start_time": _start_time(path, raw["Timestamp"]),
    }
    if label_col in raw.columns:
        columns["label"] = raw[label_col].astype(object)
    for field, (column, factor) in FEATURES.items():
        values = _number(raw[column]) * factor
        integer = record.FIELD_BY_NAME[field].type is FieldType.INTEGER
        columns[field] = values.astype("Int64") if integer else values
    # A rate over zero duration and a ratio over a zero denominator are missing (D28, D29).
    for field in RATES:
        columns[field] = columns[field].where(columns["duration"] > 0)
    forward = columns["fwd_packets"].astype("float64")
    columns["down_up_ratio"] = columns["down_up_ratio"].where(forward > 0)
    for field, value in INDICATORS.items():
        columns[field] = (protocol == value).astype("int64")
    names = [
        field.name
        for field in record.FIELDS
        if field.role is not Role.DERIVED and field.name in columns
    ]
    return pd.DataFrame({field: columns[field] for field in names})


def _number(text: pd.Series) -> pd.Series:
    """Numbers as float64; a negative, infinite or non-numeric value is missing (D21)."""
    values = pd.to_numeric(text, errors="coerce").astype("float64")
    return values.where(np.isfinite(values) & (values >= 0))


def _port(text: pd.Series, has_ports: pd.Series) -> pd.Series:
    return pd.to_numeric(text, errors="coerce").astype("Int64").where(has_ports)


def _start_time(path: Path, text: pd.Series) -> pd.Series:
    """Naive timestamps as the file wrote them. Raises InputError at the first value that does not parse."""
    stripped = text.str.strip()
    parsed = pd.to_datetime(stripped, format=TIMESTAMP_FORMATS[0], errors="coerce")
    for timestamp_format in TIMESTAMP_FORMATS[1:]:
        parsed = parsed.fillna(
            pd.to_datetime(stripped, format=timestamp_format, errors="coerce")
        )
    bad = parsed.isna().to_numpy()
    if bad.any():
        row = int(np.argmax(bad))
        raise InputError(
            f"{path}: line {row + 2}: the timestamp {text.iloc[row]!r} is missing or does not parse "
            "as day/month/year hours:minutes, with or without seconds."
        )
    return parsed
