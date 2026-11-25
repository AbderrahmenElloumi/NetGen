"""Flow formation and flow features from packet records. Not a stage.

This module imports neither scapy nor any reader, so a future packet reader
can reuse it.
"""

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from typing import Literal

import pandas as pd

from netgen import record
from netgen.record import FieldType, FlowSettings, Role

PacketProtocol = Literal["tcp", "udp", "icmp", "other"]
Endpoint = tuple[str, int | None]
FlowKey = tuple[PacketProtocol, Endpoint, Endpoint]

FIN, RST, ACK = 0x01, 0x04, 0x10
DEFAULT_SETTINGS = FlowSettings()
EPOCH = datetime(1970, 1, 1)


@dataclass(frozen=True)
class PacketRecord:
    """One IP packet, as flow formation and the flow features need it.

    `timestamp` is epoch seconds. Ports are None for `icmp` and `other`.
    `tcp_flags` is the TCP flag byte, 0 for every other protocol.
    `payload_bytes` is the IP total length minus the IP and transport headers,
    so link-layer padding never counts. `ip_header_bytes` is the IPv4 header
    with its options, or the 40-byte IPv6 header with its extension headers.
    `transport_header_bytes` is the TCP header with its options, the 8-byte
    UDP header, the 8-byte ICMP or ICMPv6 header, or 0 for `other`.
    """

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: PacketProtocol
    tcp_flags: int
    payload_bytes: int
    ip_header_bytes: int
    transport_header_bytes: int


@dataclass(frozen=True)
class Flow:
    """A formed flow: its key, its forward endpoint and its packets per direction.

    The `src_*` and `dst_*` fields are those of the first packet, which is
    forward. Each direction keeps its packets in capture order.
    """

    key: FlowKey
    src_ip: str
    dst_ip: str
    src_port: int | None
    dst_port: int | None
    protocol: PacketProtocol
    forward: tuple[PacketRecord, ...]
    backward: tuple[PacketRecord, ...]

    @property
    def start_time(self) -> datetime:
        """The first packet's time, as a naive UTC timestamp."""
        return EPOCH + timedelta(seconds=self.forward[0].timestamp)

    @property
    def duration(self) -> float:
        """Seconds from the earliest to the latest packet, so never negative."""
        times = [packet.timestamp for packet in self.forward + self.backward]
        return max(times) - min(times)


def flow_key(packet: PacketRecord) -> FlowKey:
    """Both endpoints plus protocol, the same for both directions.

    ICMP and `other` packets carry no ports, so their endpoints are addresses only.
    """
    one: Endpoint = (packet.src_ip, packet.src_port)
    two: Endpoint = (packet.dst_ip, packet.dst_port)
    low, high = sorted((one, two))
    return (packet.protocol, low, high)


@dataclass
class _OpenFlow:
    first: PacketRecord
    forward: list[PacketRecord] = field(default_factory=list)
    backward: list[PacketRecord] = field(default_factory=list)
    latest: float = 0.0
    fin_from_forward: bool | None = None  # the side that sent the first FIN
    both_sides_sent_fin: bool = False

    def __post_init__(self) -> None:
        self.latest = self.first.timestamp

    def expired(self, packet: PacketRecord, settings: FlowSettings) -> bool:
        """Whether `packet` arrives past the idle or the active timeout."""
        if packet.timestamp - self.latest > settings.idle_timeout:
            return True
        active = settings.active_timeout
        return active is not None and packet.timestamp - self.first.timestamp > active

    def add(self, packet: PacketRecord) -> bool:
        """Add `packet` and return whether it closes the flow."""
        is_forward = (packet.src_ip, packet.src_port) == (
            self.first.src_ip,
            self.first.src_port,
        )
        (self.forward if is_forward else self.backward).append(packet)
        self.latest = max(self.latest, packet.timestamp)
        flags = packet.tcp_flags
        if flags & RST:
            return True
        if self.fin_from_forward is None:
            if flags & FIN:
                self.fin_from_forward = is_forward
            return False
        if not self.both_sides_sent_fin:
            self.both_sides_sent_fin = bool(flags & FIN) and (
                is_forward != self.fin_from_forward
            )
            return False
        # The last ACK: sent by the side of the first FIN, after the other FIN.
        return is_forward == self.fin_from_forward and bool(flags & ACK)

    def formed(self) -> Flow:
        first = self.first
        return Flow(
            key=flow_key(first),
            src_ip=first.src_ip,
            dst_ip=first.dst_ip,
            src_port=first.src_port,
            dst_port=first.dst_port,
            protocol=first.protocol,
            forward=tuple(self.forward),
            backward=tuple(self.backward),
        )


def form_flows(
    packets: Iterable[PacketRecord], settings: FlowSettings = DEFAULT_SETTINGS
) -> list[Flow]:
    """Group packet records, in capture order, into flows, in the order they start.

    A flow closes at an RST, or at the last ACK of a two-sided FIN close. A
    packet past the idle or the active timeout of its flow starts a new flow
    with the same key. Every packet ends up in exactly one flow.
    """
    started: list[_OpenFlow] = []
    open_flows: dict[FlowKey, _OpenFlow] = {}
    for packet in packets:
        key = flow_key(packet)
        flow = open_flows.get(key)
        if flow is None or flow.expired(packet, settings):
            flow = _OpenFlow(packet)
            started.append(flow)
            open_flows[key] = flow
        if flow.add(packet):
            del open_flows[key]
    return [flow.formed() for flow in started]


def flow_features(flow: Flow) -> dict[str, float]:
    """The feature fields of the flow record for one flow, by field name.

    Each value follows the definition next to its field in `netgen.record`. A
    rate over zero duration is NaN. A formed flow has at least one forward
    packet, so `down_up_ratio` always has a denominator.
    """
    duration = flow.duration
    gaps = [
        later - earlier
        for earlier, later in pairwise(
            sorted(packet.timestamp for packet in flow.forward + flow.backward)
        )
    ]
    forward, backward = len(flow.forward), len(flow.backward)
    fwd_bytes = sum(packet.payload_bytes for packet in flow.forward)
    bwd_bytes = sum(packet.payload_bytes for packet in flow.backward)
    return {
        "duration": duration,
        "fwd_packets": forward,
        "bwd_packets": backward,
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        **_lengths("fwd", flow.forward),
        **_lengths("bwd", flow.backward),
        "bytes_per_s": _rate(fwd_bytes + bwd_bytes, duration),
        "packets_per_s": _rate(forward + backward, duration),
        "fwd_packets_per_s": _rate(forward, duration),
        "bwd_packets_per_s": _rate(backward, duration),
        "iat_mean": statistics.fmean(gaps) if gaps else 0.0,
        "iat_std": _sample_std(gaps),
        "iat_max": max(gaps, default=0.0),
        "iat_min": min(gaps, default=0.0),
        "fwd_iat_total": _span(flow.forward),
        "bwd_iat_total": _span(flow.backward),
        "fwd_header_bytes": sum(p.transport_header_bytes for p in flow.forward),
        "bwd_header_bytes": sum(p.transport_header_bytes for p in flow.backward),
        "down_up_ratio": backward // forward,
        "is_tcp": int(flow.protocol == "tcp"),
        "is_udp": int(flow.protocol == "udp"),
        "is_icmp": int(flow.protocol == "icmp"),
    }


def _lengths(direction: str, packets: Sequence[PacketRecord]) -> dict[str, float]:
    sizes = [packet.payload_bytes for packet in packets]
    return {
        f"{direction}_len_min": min(sizes, default=0),
        f"{direction}_len_max": max(sizes, default=0),
        f"{direction}_len_mean": statistics.fmean(sizes) if sizes else 0.0,
        f"{direction}_len_std": _sample_std(sizes),
    }


def _sample_std(values: Sequence[float]) -> float:
    """The n - 1 standard deviation; 0 for fewer than two values."""
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _span(packets: Sequence[PacketRecord]) -> float:
    """Seconds from the earliest to the latest packet; 0 for none or one."""
    times = [packet.timestamp for packet in packets]
    return max(times) - min(times) if times else 0.0


def _rate(count: int, duration: float) -> float:
    """`count` per second; NaN at zero duration, never a huge number (D28)."""
    return count / duration if duration > 0 else math.nan


_COLUMNS = tuple(
    column for column in record.FIELDS if column.role in (Role.IDENTITY, Role.FEATURE)
)
_DTYPES = {
    FieldType.TEXT: "object",
    FieldType.NULLABLE_INTEGER: "Int64",
    FieldType.INTEGER: "int64",
    FieldType.DECIMAL: "float64",
    FieldType.INDICATOR: "int64",
}


def flow_frame(flows: Sequence[Flow]) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """One row per flow: the identity and feature fields of the record, in record order.

    No label column: packets carry no label. The second item holds the
    warnings: a start time past the range of a timestamp (after 2262) becomes
    missing, and the warning counts those flows.
    """
    rows = [
        {
            "src_ip": flow.src_ip,
            "dst_ip": flow.dst_ip,
            "src_port": flow.src_port,
            "dst_port": flow.dst_port,
            "protocol": flow.protocol,
            **flow_features(flow),
        }
        for flow in flows
    ]
    names = [column.name for column in _COLUMNS]
    frame = pd.DataFrame(rows, columns=names).astype(
        {
            column.name: _DTYPES[column.type]
            for column in _COLUMNS
            if column.type in _DTYPES
        }
    )
    frame["start_time"] = pd.to_datetime(
        pd.Series([flow.forward[0].timestamp for flow in flows], dtype="float64"),
        unit="s",
        errors="coerce",
    )
    missing = int(frame["start_time"].isna().sum())
    if not missing:
        return frame, ()
    flows_start = "1 flow starts" if missing == 1 else f"{missing} flows start"
    warning = (
        f"{flows_start} after 2262, past the range of a timestamp; start_time is "
        "missing for them. Check the capture's clock."
    )
    return frame, (warning,)
