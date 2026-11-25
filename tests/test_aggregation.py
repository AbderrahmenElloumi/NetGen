"""Tests of netgen.aggregation: packet records become flows, flows get their features."""

import dataclasses
import importlib
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from netgen import record
from netgen.aggregation import (
    Flow,
    PacketProtocol,
    PacketRecord,
    flow_features,
    flow_frame,
    form_flows,
)
from netgen.errors import NetGenError
from netgen.record import FlowSettings, Role
from netgen.table import FlowTable, check_record
from test_capture_reader import (
    FIN_FLOWS,
    PCAPNG_FLOWS,
    REAL,
    RST_FLOWS,
    read,
)

FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10
CLIENT, SERVER, OTHER = "192.0.2.1", "198.51.100.2", "203.0.113.3"
CLIENT_PORT, SERVER_PORT = 40000, 80
EXCERPT = REAL / "cicids2017_excerpt.csv"
FEATURE_NAMES = {field.name for field in record.fields_by_role(Role.FEATURE)}
# A capture is unlabelled: its columns are the identity fields, then the features.
CAPTURE_COLUMNS = [
    field.name for field in record.FIELDS if field.role in (Role.IDENTITY, Role.FEATURE)
]


def packet(
    timestamp: float,
    src: str = CLIENT,
    dst: str = SERVER,
    flags: int = ACK,
    *,
    sport: int | None = CLIENT_PORT,
    dport: int | None = SERVER_PORT,
    protocol: PacketProtocol = "tcp",
    payload: int = 0,
) -> PacketRecord:
    return PacketRecord(
        timestamp=timestamp,
        src_ip=src,
        dst_ip=dst,
        src_port=sport,
        dst_port=dport,
        protocol=protocol,
        tcp_flags=flags,
        payload_bytes=payload,
        ip_header_bytes=20,
        transport_header_bytes=20 if protocol == "tcp" else 8,
    )


def c2s(timestamp: float, flags: int = ACK, payload: int = 0) -> PacketRecord:
    """A TCP packet from the client to the server."""
    return packet(timestamp, CLIENT, SERVER, flags, payload=payload)


def s2c(timestamp: float, flags: int = ACK, payload: int = 0) -> PacketRecord:
    """A TCP packet from the server to the client."""
    return packet(
        timestamp,
        SERVER,
        CLIENT,
        flags,
        sport=SERVER_PORT,
        dport=CLIENT_PORT,
        payload=payload,
    )


def sizes(flows: list[Flow]) -> list[int]:
    """The number of packets of each flow, in flow order."""
    return [len(flow.forward) + len(flow.backward) for flow in flows]


# Named regression tests.


def test_d26_expiring_packet_starts_new_flow() -> None:
    expiring = c2s(1.0 + 120.5)
    flows = form_flows([c2s(0.0, SYN), s2c(1.0, SYN | ACK), expiring])
    assert sizes(flows) == [2, 1]
    assert flows[1].forward == (expiring,)


def test_d27_closing_acks_do_not_start_phantom_flows() -> None:
    close = [c2s(5.0, FIN | ACK), s2c(5.1), s2c(5.2, FIN | ACK), c2s(5.3)]
    later = c2s(6.0, SYN)
    flows = form_flows([c2s(0.0, SYN), s2c(0.1, SYN | ACK), c2s(0.2), *close, later])
    assert sizes(flows) == [7, 1]
    assert flows[0].forward[-1] == close[-1]
    assert flows[1].forward == (later,)


# Closing rules.


def test_handshake_and_two_sided_fin_give_one_flow_with_all_packets() -> None:
    packets = [
        c2s(0.0, SYN),
        s2c(0.1, SYN | ACK),
        c2s(0.2),
        c2s(0.3, PSH | ACK, payload=100),
        s2c(0.4, PSH | ACK, payload=500),
        c2s(0.5, FIN | ACK),
        s2c(0.6),
        s2c(0.7, FIN | ACK),
        c2s(0.8),
    ]
    (flow,) = form_flows(packets)
    assert flow.forward == tuple(p for p in packets if p.src_ip == CLIENT)
    assert flow.backward == tuple(p for p in packets if p.src_ip == SERVER)


def test_a_combined_fin_ack_in_the_middle_of_the_close_closes_at_the_last_ack() -> None:
    flows = form_flows(
        [c2s(0.0, FIN | ACK), s2c(0.1, FIN | ACK), c2s(0.2), c2s(0.3, SYN)]
    )
    assert sizes(flows) == [3, 1]


def test_an_ack_before_the_other_side_sends_fin_does_not_close() -> None:
    flows = form_flows([c2s(0.0, FIN | ACK), c2s(0.1), s2c(0.2), s2c(0.3, ACK)])
    assert sizes(flows) == [4]


def test_a_repeated_fin_from_the_same_side_does_not_close() -> None:
    flows = form_flows([c2s(0.0, FIN | ACK), c2s(0.1, FIN | ACK), c2s(0.2)])
    assert sizes(flows) == [3]


def test_the_ack_of_the_second_fin_side_does_not_close() -> None:
    flows = form_flows([c2s(0.0, FIN | ACK), s2c(0.1, FIN | ACK), s2c(0.2)])
    assert sizes(flows) == [3]


def test_only_an_ack_from_the_first_fin_side_closes_after_both_fins() -> None:
    flows = form_flows(
        [c2s(0.0, FIN | ACK), s2c(0.1, FIN | ACK), c2s(0.2, PSH), c2s(0.3), c2s(0.4)]
    )
    assert sizes(flows) == [4, 1]


def test_rst_closes_at_once_and_a_later_packet_starts_a_new_flow() -> None:
    reset = s2c(0.1, RST | ACK)
    flows = form_flows([c2s(0.0, SYN), reset, c2s(0.2, SYN)])
    assert sizes(flows) == [2, 1]
    assert flows[0].backward == (reset,)


# Timeouts.


def test_the_idle_timeout_defaults_to_120_seconds() -> None:
    assert sizes(form_flows([c2s(0.0), c2s(120.0), c2s(240.0)])) == [3]
    assert sizes(form_flows([c2s(0.0), c2s(120.001)])) == [1, 1]


def test_a_custom_idle_timeout_splits_a_flow_and_keeps_the_expiring_packet() -> None:
    settings = FlowSettings(idle_timeout=10.0)
    packets = [c2s(0.0), s2c(5.0), c2s(15.5), s2c(20.0)]
    flows = form_flows(packets, settings)
    assert sizes(flows) == [2, 2]
    assert flows[1].forward == (packets[2],)
    assert flows[1].backward == (packets[3],)


def test_the_active_timeout_cuts_a_long_lived_flow() -> None:
    packets = [c2s(float(second)) for second in range(13)]
    settings = FlowSettings(active_timeout=5.0)
    assert sizes(form_flows(packets, settings)) == [6, 6, 1]


def test_the_active_timeout_is_off_by_default() -> None:
    packets = [c2s(float(second)) for second in range(1000)]
    assert sizes(form_flows(packets)) == [1000]


# Flow key and direction.


def test_the_reply_direction_joins_the_same_flow_as_backward() -> None:
    request, reply, ack = c2s(0.0, PSH | ACK, 10), s2c(0.1, PSH | ACK, 20), c2s(0.2)
    (flow,) = form_flows([request, reply, ack])
    assert flow.forward == (request, ack)
    assert flow.backward == (reply,)


def test_forward_is_the_direction_of_the_first_packet_even_from_the_server() -> None:
    first, second = s2c(0.0, PSH | ACK, 20), c2s(0.1)
    (flow,) = form_flows([first, second])
    assert (flow.src_ip, flow.src_port, flow.dst_ip, flow.dst_port) == (
        SERVER,
        SERVER_PORT,
        CLIENT,
        CLIENT_PORT,
    )
    assert flow.protocol == "tcp"
    assert flow.forward == (first,)
    assert flow.backward == (second,)


@pytest.mark.parametrize("protocol", ["icmp", "other"])
def test_icmp_and_other_are_keyed_on_addresses_without_ports(
    protocol: PacketProtocol,
) -> None:
    echo = packet(0.0, CLIENT, SERVER, 0, sport=None, dport=None, protocol=protocol)
    reply = packet(0.1, SERVER, CLIENT, 0, sport=None, dport=None, protocol=protocol)
    (flow,) = form_flows([echo, reply])
    assert flow.key == (protocol, (CLIENT, None), (SERVER, None))
    assert (flow.src_port, flow.dst_port) == (None, None)
    assert flow.forward == (echo,)
    assert flow.backward == (reply,)


def test_a_different_protocol_with_the_same_addresses_is_a_different_flow() -> None:
    tcp = c2s(0.0)
    udp = packet(0.1, protocol="udp", flags=0)
    icmp = packet(0.2, flags=0, sport=None, dport=None, protocol="icmp")
    other = packet(0.3, flags=0, sport=None, dport=None, protocol="other")
    flows = form_flows([tcp, udp, icmp, other])
    assert [flow.protocol for flow in flows] == ["tcp", "udp", "icmp", "other"]


def test_different_endpoints_are_different_flows() -> None:
    to_other = packet(0.1, CLIENT, OTHER)
    other_port = packet(0.2, sport=CLIENT_PORT + 1)
    flows = form_flows([c2s(0.0), to_other, other_port, c2s(0.3)])
    assert sizes(flows) == [2, 1, 1]


def test_the_flow_key_is_the_same_for_both_directions() -> None:
    (one,) = form_flows([c2s(0.0)])
    (two,) = form_flows([s2c(0.0)])
    assert one.key == two.key


# Start time and duration.


def test_start_time_is_the_first_packets_time_as_naive_utc() -> None:
    (flow,) = form_flows([c2s(1_700_000_000.25), s2c(1_699_999_999.0)])
    assert flow.start_time == datetime(2023, 11, 14, 22, 13, 20, 250000)
    assert flow.start_time.tzinfo is None


def test_an_out_of_order_packet_cannot_give_a_negative_duration() -> None:
    (flow,) = form_flows([c2s(10.0), s2c(8.0), c2s(12.0)])
    assert flow.duration == 4.0


def test_a_one_packet_flow_has_zero_duration() -> None:
    (flow,) = form_flows([c2s(3.0)])
    assert flow.duration == 0.0


# The committed PcapPlusPlus captures give the hand-worked flow counts.


@pytest.mark.parametrize(
    ("name", "flow_count"),
    [
        ("one_http_stream_fin.pcap", FIN_FLOWS),
        ("one_http_stream_rst.pcap", RST_FLOWS),
        ("pcapng-example.pcapng", PCAPNG_FLOWS),
    ],
)
def test_the_real_captures_give_the_hand_worked_flow_counts(
    name: str, flow_count: int
) -> None:
    records, _ = read(REAL / name)
    flows = form_flows(records)
    assert len(flows) == flow_count
    assert sum(sizes(flows)) == len(records)


# Invariants.


def test_seeded_random_packets_are_conserved_and_durations_are_never_negative() -> None:
    rng = random.Random(28)
    hosts = [CLIENT, SERVER, OTHER]
    packets = []
    clock = 0.0
    for _ in range(3000):
        clock += rng.choice([0.01, 0.5, 3.0, 40.0, 200.0])
        src, dst = rng.sample(hosts, 2)
        protocol: PacketProtocol = rng.choice(["tcp", "tcp", "udp", "icmp", "other"])
        ported = protocol in ("tcp", "udp")
        packets.append(
            packet(
                clock + rng.uniform(-5.0, 5.0),  # some packets out of order
                src,
                dst,
                rng.choice([ACK, FIN | ACK, RST, SYN, PSH | ACK])
                if protocol == "tcp"
                else 0,
                sport=rng.choice([80, 443]) if ported else None,
                dport=rng.choice([80, 443]) if ported else None,
                protocol=protocol,
            )
        )
    for settings in (
        FlowSettings(),
        FlowSettings(idle_timeout=30.0, active_timeout=60.0),
    ):
        flows = form_flows(packets, settings)
        placed = Counter(id(p) for flow in flows for p in flow.forward + flow.backward)
        assert sum(placed.values()) == len(packets)
        assert placed == Counter(id(p) for p in packets)
        assert all(flow.duration >= 0 for flow in flows)


# Settings validation.


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
@pytest.mark.parametrize("name", ["idle_timeout", "active_timeout"])
def test_a_zero_or_negative_timeout_is_refused(name: str, value: float) -> None:
    with pytest.raises(NetGenError, match=name):
        FlowSettings(**{name: value})


# Isolation.


def test_the_module_imports_neither_scapy_nor_any_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in list(sys.modules):
        if name == "scapy" or name.startswith(("scapy.", "netgen.readers")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "scapy", None)
    monkeypatch.setitem(sys.modules, "netgen.readers", None)
    monkeypatch.delitem(sys.modules, "netgen.aggregation")
    module = importlib.import_module("netgen.aggregation")
    assert module.form_flows is not None
    assert Path(module.__file__ or "").name == "aggregation.py"


def test_the_legacy_flow_aggregator_is_deleted() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("netgen.features.flow_aggregator")


# Flow features: hand-worked values.


def features_of(packets: list[PacketRecord]) -> dict[str, float]:
    (flow,) = form_flows(packets)
    return flow_features(flow)


def test_the_features_of_a_small_tcp_flow_match_the_hand_worked_values() -> None:
    # Forward: 0.0 SYN (40-byte header with options), 1.0, 1.5 (100 B), 4.0 (200 B).
    # Backward: 0.5 SYN-ACK, 3.0 (300 B). Every other TCP header is 20 bytes.
    syn = dataclasses.replace(c2s(0.0, SYN), transport_header_bytes=40)
    packets = [
        syn,
        s2c(0.5, SYN | ACK),
        c2s(1.0),
        c2s(1.5, PSH | ACK, payload=100),
        s2c(3.0, PSH | ACK, payload=300),
        c2s(4.0, PSH | ACK, payload=200),
    ]
    expected = {
        "duration": 4.0,
        "fwd_packets": 4,
        "bwd_packets": 2,
        "fwd_bytes": 300,
        "bwd_bytes": 300,
        "fwd_len_min": 0.0,
        "fwd_len_max": 200.0,
        "fwd_len_mean": 75.0,
        # Deviations -75, -75, 25, 125: the squares sum to 27500, over n - 1 = 3.
        "fwd_len_std": math.sqrt(27500 / 3),
        "bwd_len_min": 0.0,
        "bwd_len_max": 300.0,
        "bwd_len_mean": 150.0,
        "bwd_len_std": math.sqrt(2 * 150**2 / 1),
        "bytes_per_s": 600 / 4.0,
        "packets_per_s": 6 / 4.0,
        "fwd_packets_per_s": 4 / 4.0,
        "bwd_packets_per_s": 2 / 4.0,
        # Times 0, 0.5, 1.0, 1.5, 3.0, 4.0: gaps 0.5, 0.5, 0.5, 1.5, 1.0.
        "iat_mean": 0.8,
        "iat_std": math.sqrt((3 * 0.3**2 + 0.7**2 + 0.2**2) / 4),
        "iat_max": 1.5,
        "iat_min": 0.5,
        "fwd_iat_total": 4.0,
        "bwd_iat_total": 2.5,
        "fwd_header_bytes": 40 + 3 * 20,
        "bwd_header_bytes": 2 * 20,
        "down_up_ratio": 0,
        "is_tcp": 1,
        "is_udp": 0,
        "is_icmp": 0,
    }
    values = features_of(packets)
    assert set(values) == FEATURE_NAMES
    assert values == pytest.approx(expected)


def test_the_gaps_are_taken_in_time_order_so_none_is_negative() -> None:
    values = features_of([c2s(10.0), s2c(8.0), c2s(12.0)])
    assert values["duration"] == 4.0
    assert (values["iat_min"], values["iat_max"], values["iat_mean"]) == (2.0, 2.0, 2.0)
    assert values["fwd_iat_total"] == 2.0


def test_down_up_ratio_is_the_integer_part_of_backward_over_forward_packets() -> None:
    replies = [s2c(0.1 * i) for i in range(1, 6)]
    assert features_of([c2s(0.0), *replies])["down_up_ratio"] == 5
    assert features_of([c2s(0.0), c2s(0.05), *replies])["down_up_ratio"] == 2


def test_a_udp_flow_has_ports_the_udp_indicator_and_eight_byte_headers() -> None:
    query = packet(0.0, flags=0, protocol="udp", payload=40)
    answer = packet(
        0.2,
        SERVER,
        CLIENT,
        0,
        sport=SERVER_PORT,
        dport=CLIENT_PORT,
        protocol="udp",
        payload=120,
    )
    flows = form_flows([query, answer])
    values = flow_features(flows[0])
    assert (values["is_tcp"], values["is_udp"], values["is_icmp"]) == (0, 1, 0)
    assert (values["fwd_header_bytes"], values["bwd_header_bytes"]) == (8, 8)
    assert (values["fwd_bytes"], values["bwd_bytes"]) == (40, 120)
    frame, _ = flow_frame(flows)
    assert (frame["src_port"][0], frame["dst_port"][0]) == (CLIENT_PORT, SERVER_PORT)


@pytest.mark.parametrize(
    ("protocol", "indicators", "header"),
    [("icmp", (0, 0, 1), 8), ("other", (0, 0, 0), 0)],
)
def test_icmp_and_other_flows_have_missing_ports_and_their_indicators(
    protocol: PacketProtocol, indicators: tuple[int, int, int], header: int
) -> None:
    echo = packet(0.0, flags=0, sport=None, dport=None, protocol=protocol)
    flows = form_flows([dataclasses.replace(echo, transport_header_bytes=header)])
    values = flow_features(flows[0])
    assert (values["is_tcp"], values["is_udp"], values["is_icmp"]) == indicators
    assert values["fwd_header_bytes"] == header
    frame, _ = flow_frame(flows)
    assert frame["src_port"].isna().all() and frame["dst_port"].isna().all()
    assert frame["protocol"][0] == protocol


def test_a_flow_with_only_forward_packets_gives_zero_for_the_empty_direction() -> None:
    values = features_of([c2s(0.0, SYN), c2s(1.0, payload=10), c2s(2.0, payload=30)])
    for name in (
        "bwd_packets",
        "bwd_bytes",
        "bwd_len_min",
        "bwd_len_max",
        "bwd_len_mean",
        "bwd_len_std",
        "bwd_iat_total",
        "bwd_header_bytes",
        "bwd_packets_per_s",
    ):
        assert values[name] == 0, name
    assert values["fwd_bytes"] == 40
    # Lengths 0, 10, 30 around the mean 40 / 3, over n - 1 = 2.
    deviations = (-40 / 3, -10 / 3, 50 / 3)
    assert values["fwd_len_std"] == pytest.approx(
        math.sqrt(sum(d**2 for d in deviations) / 2)
    )


# Named regression tests for the numerics.

RATES = ("bytes_per_s", "packets_per_s", "fwd_packets_per_s", "bwd_packets_per_s")


def test_d28_a_one_packet_flow_has_zero_duration_and_missing_rates() -> None:
    values = features_of([c2s(5.0, PSH | ACK, payload=1000)])
    assert values["duration"] == 0.0
    for name in RATES:
        assert math.isnan(values[name]), name
    for name in ("iat_mean", "iat_std", "iat_max", "iat_min", "fwd_iat_total"):
        assert values[name] == 0.0, name
    assert values["fwd_len_std"] == 0.0


def test_d28_packets_at_the_same_instant_have_missing_rates() -> None:
    values = features_of([c2s(5.0, payload=10), s2c(5.0, payload=10)])
    assert values["duration"] == 0.0
    for name in RATES:
        assert math.isnan(values[name]), name


def test_d29_a_ratio_without_reverse_traffic_is_never_the_forward_value() -> None:
    values = features_of([c2s(float(second)) for second in range(3)])
    assert values["fwd_packets"] == 3
    assert values["down_up_ratio"] == 0


# The flow frame.


def test_the_flow_frame_has_the_record_columns_and_passes_the_seam_check() -> None:
    flows = form_flows(
        [c2s(0.0, SYN), s2c(0.1, SYN | ACK), packet(0.2, protocol="udp", flags=0)]
    )
    frame, warnings = flow_frame(flows)
    assert list(frame.columns) == CAPTURE_COLUMNS
    assert len(frame) == 2 and warnings == ()
    check_record(FlowTable.from_frame(frame))


def test_an_empty_flow_list_gives_an_empty_frame_that_passes_the_seam_check() -> None:
    frame, warnings = flow_frame([])
    assert list(frame.columns) == CAPTURE_COLUMNS
    assert len(frame) == 0 and warnings == ()
    check_record(FlowTable.from_frame(frame))


def test_start_time_is_a_naive_utc_timestamp_in_the_frame() -> None:
    frame, _ = flow_frame(form_flows([c2s(1_700_000_000.25)]))
    assert frame["start_time"][0] == pd.Timestamp("2023-11-14 22:13:20.25")
    assert frame["start_time"].dt.tz is None


def test_a_start_time_past_the_timestamp_range_is_missing_with_a_warning() -> None:
    flows = form_flows([c2s(0.0), packet(1.8e13, OTHER, SERVER)])
    frame, warnings = flow_frame(flows)
    assert frame["start_time"][0] == pd.Timestamp("1970-01-01")
    assert pd.isna(frame["start_time"][1])
    assert len(warnings) == 1 and "1 flow " in warnings[0]
    check_record(FlowTable.from_frame(frame))


# Exactness: the capture-side definitions reproduce the excerpt's arithmetic.
#
# Each entry rebuilds one excerpt row as packets, per direction: (time in
# microseconds, payload bytes, transport header bytes). Only the rows whose
# packets the excerpt fixes are rebuilt: two forward packets, or three forward
# packets and one backward packet whose times follow from the IAT minimum,
# maximum and total. The header split among packets is free; its sum is not.
EXCERPT_FLOWS: dict[
    int, tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]
] = {
    0: ([(0, 6, 20), (3, 6, 20)], []),
    1: ([(0, 6, 20), (0, 6, 20)], []),
    2: ([(0, 0, 32), (0, 0, 32)], []),
    8: ([(0, 6, 20), (4, 6, 20)], []),
    11: ([(0, 0, 40), (690, 0, 32), (5_638_432, 0, 32)], [(72, 0, 40)]),
    13: ([(0, 0, 40), (1245, 0, 32), (5_216_127, 0, 32)], [(124, 0, 40)]),
    14: ([(0, 0, 32), (404, 0, 32)], []),
}

MICROSECONDS = 1e6
# Each feature, the excerpt column that holds it, and the factor that turns the
# excerpt's unit into the record's.
EXCERPT_COLUMNS: dict[str, tuple[str, float]] = {
    "duration": ("Flow Duration", 1 / MICROSECONDS),
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
        f"iat_{s.lower()}": (f"Flow IAT {s}", 1 / MICROSECONDS)
        for s in ("Mean", "Std", "Max", "Min")
    },
    "fwd_iat_total": ("Fwd IAT Total", 1 / MICROSECONDS),
    "bwd_iat_total": ("Bwd IAT Total", 1 / MICROSECONDS),
    "fwd_header_bytes": ("Fwd Header Length", 1),
    "bwd_header_bytes": ("Bwd Header Length", 1),
    "down_up_ratio": ("Down/Up Ratio", 1),
}
PROTOCOL_NUMBERS = {"is_tcp": 6, "is_udp": 17, "is_icmp": 1}


@pytest.fixture(scope="module")
def excerpt() -> pd.DataFrame:
    frame = pd.read_csv(EXCERPT, encoding="cp1252")
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def excerpt_flow(row: int) -> Flow:
    forward, backward = EXCERPT_FLOWS[row]
    packets = [
        dataclasses.replace(
            make(time / MICROSECONDS, payload=payload), transport_header_bytes=header
        )
        for make, direction in ((c2s, forward), (s2c, backward))
        for time, payload, header in direction
    ]
    packets.sort(key=lambda p: p.timestamp)
    (flow,) = form_flows(packets)
    return flow


def test_every_feature_is_compared_with_the_excerpt() -> None:
    assert set(EXCERPT_COLUMNS) | set(PROTOCOL_NUMBERS) == FEATURE_NAMES


@pytest.mark.parametrize("row", sorted(EXCERPT_FLOWS))
def test_the_capture_definitions_reproduce_the_excerpt_row(
    excerpt: pd.DataFrame, row: int
) -> None:
    line = excerpt.loc[row]
    flow = excerpt_flow(row)
    assert len(flow.forward) == line["Total Fwd Packets"]
    assert len(flow.backward) == line["Total Backward Packets"]
    values = flow_features(flow)
    for name, (column, factor) in EXCERPT_COLUMNS.items():
        if name in RATES and line["Flow Duration"] == 0:
            # The flow export writes Infinity, NaN or 0 for a rate over zero
            # duration; the record makes it missing (D28).
            assert math.isnan(values[name]), (row, name)
        else:
            assert values[name] == pytest.approx(
                float(line[column]) * factor, rel=1e-6, abs=1e-12
            ), (row, name)
    for name, number in PROTOCOL_NUMBERS.items():
        assert values[name] == int(line["Protocol"] == number), (row, name)
