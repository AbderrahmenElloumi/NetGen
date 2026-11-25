"""Tests of the capture reader (netgen.readers.capture): `sniff` and packet reading."""

import struct
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest
from scapy.all import (
    ARP,
    ICMP,
    IP,
    TCP,
    UDP,
    Ether,
    ICMPv6EchoRequest,
    ICMPv6MLReport,
    ICMPv6ND_NS,
    IPv6,
    IPv6ExtHdrFragment,
    IPv6ExtHdrHopByHop,
    Packet,
    PcapNgWriter,
    PcapWriter,
    Raw,
    RouterAlert,
    wrpcap,
)

from netgen import record
from netgen.aggregation import PacketRecord
from netgen.errors import InputError
from netgen.readers import Reader, capture
from netgen.record import FlowSettings, Role
from netgen.table import TableResult, check_record

REAL = Path(__file__).parent.parent / "data" / "samples" / "real"

# Counted once from the record headers of each file (pcap packet records,
# pcapng enhanced packet blocks), not through scapy. Every packet is TCP over
# IP: no packet of these captures is dropped. The pcapng holds two interfaces
# (Ethernet on loopback, Linux cooked capture); four of its packets are IPv6.
FIN_PACKETS = 15
RST_PACKETS = 10
PCAPNG_PACKETS = 159
PCAPNG_IPV6_PACKETS = 4

# Worked out from the packet listing of each file, with the closing rules and
# the 120-second idle timeout of flow formation; no active timeout. Every
# packet is TCP, so a flow key is the two endpoints.
# - one_http_stream_fin.pcap: one key, no gap above 1 second. The server sends
#   the first FIN, the client its FIN, and the server's last ACK (the last
#   packet) closes the flow: 1 flow.
# - one_http_stream_rst.pcap: one key. The client's RST is the last packet: 1
#   flow.
# - pcapng-example.pcapng: 15 keys. On the loopback interface, the MySQL
#   stream's second packet (the SYN-ACK) carries a timestamp 1,472,003,999
#   seconds after the SYN, as published. It arrives past the idle timeout and
#   starts a second flow. The later packets lie before it in time, so no later
#   gap exceeds 120 seconds, and the two-sided FIN closes that flow at the last
#   packet. The other 14 keys are on the Linux cooked interface, whose
#   timestamps lie near 2^64 microseconds (about 1.8e13 seconds, past the
#   range of a datetime) with gaps under 1 second. They hold no FIN, and their
#   one RST is the last packet of its key: 1 flow each. 2 + 14 = 16 flows.
FIN_FLOWS = 1
RST_FLOWS = 1
PCAPNG_FLOWS = 16

ETHER = Ether(src="00:00:5e:00:53:01", dst="00:00:5e:00:53:02")
A4, B4 = "192.0.2.1", "198.51.100.2"
A6, B6 = "2001:db8::1", "2001:db8::2"

# The four pcap magic numbers as the first four bytes of a file, and pcapng's.
PCAP_MAGICS = (
    bytes.fromhex("d4c3b2a1"),  # microsecond, little-endian
    bytes.fromhex("a1b2c3d4"),  # microsecond, big-endian
    bytes.fromhex("4d3cb2a1"),  # nanosecond, little-endian
    bytes.fromhex("a1b23c4d"),  # nanosecond, big-endian
)
PCAPNG_MAGIC = bytes.fromhex("0a0d0d0a")


def write(path: Path, packets: Iterable[Packet]) -> Path:
    wrpcap(str(path), list(packets))
    return path


def read(path: Path) -> tuple[list[PacketRecord], dict[str, int]]:
    """All packet records of a capture, and the counts the generator returns."""
    generator = capture.read_packets(path)
    records: list[PacketRecord] = []
    while True:
        try:
            records.append(next(generator))
        except StopIteration as stop:
            return records, stop.value


def only(path: Path, packet: Packet) -> PacketRecord:
    records, _ = read(write(path, [packet]))
    assert len(records) == 1
    return records[0]


# sniff (D16)


@pytest.mark.parametrize("magic", [*PCAP_MAGICS, PCAPNG_MAGIC])
def test_d16_every_pcap_magic_and_pcapng_is_sniffed(
    tmp_path: Path, magic: bytes
) -> None:
    path = tmp_path / "capture.bin"
    path.write_bytes(magic + bytes(28))
    assert capture.sniff(path)


@pytest.mark.parametrize(
    "content",
    [b"Flow ID,Source IP\n1,2\n", bytes(range(32)), b"", b"\xd4\xc3"],
    ids=["csv", "random", "empty", "short"],
)
def test_d16_any_other_file_is_not_sniffed(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "other.bin"
    path.write_bytes(content)
    assert not capture.sniff(path)


def test_sniff_is_false_for_a_missing_file_or_a_folder(tmp_path: Path) -> None:
    assert not capture.sniff(tmp_path / "missing.pcap")
    assert not capture.sniff(tmp_path)


def test_the_real_captures_are_sniffed() -> None:
    for name in (
        "one_http_stream_fin.pcap",
        "one_http_stream_rst.pcap",
        "pcapng-example.pcapng",
    ):
        assert capture.sniff(REAL / name)


# Named regression tests


def test_d10_non_ip_packets_are_dropped_and_counted(tmp_path: Path) -> None:
    packets = [
        ETHER / ARP(psrc=A4, pdst=B4),
        ETHER / IP(src=A4, dst=B4) / UDP(sport=5353, dport=53) / Raw(b"q"),
        ETHER / ARP(psrc=B4, pdst=A4, op=2),
    ]
    records, counts = read(write(tmp_path / "arp.pcap", packets))
    assert [r.protocol for r in records] == ["udp"]
    assert counts == {"packets_read": 3, "non_ip_dropped": 2}


def test_d11_payload_bytes_exclude_headers(tmp_path: Path) -> None:
    packet = (
        ETHER
        / IP(src=A4, dst=B4)
        / TCP(sport=40000, dport=80, flags="PA")
        / Raw(bytes(100))
    )
    record = only(tmp_path / "data.pcap", packet)
    assert record.payload_bytes == 100
    assert record.ip_header_bytes == 20
    assert record.transport_header_bytes == 20


# Carried defects


def test_d12_a_pure_ack_padded_to_the_ethernet_minimum_has_no_payload(
    tmp_path: Path,
) -> None:
    packet = ETHER / IP(src=A4, dst=B4) / TCP(sport=40000, dport=80, flags="A")
    frame = Ether(bytes(packet) + bytes(6))  # padded to the 60-byte minimum
    assert len(frame) == 60
    record = only(tmp_path / "ack.pcap", frame)
    assert record.payload_bytes == 0
    assert record.tcp_flags == 0x10


def test_d13_an_icmp_error_embedding_tcp_stays_icmp(tmp_path: Path) -> None:
    embedded = IP(src=B4, dst=A4) / TCP(sport=443, dport=40000)
    packet = ETHER / IP(src=A4, dst=B4) / ICMP(type=3, code=3) / embedded
    record = only(tmp_path / "unreachable.pcap", packet)
    assert record.protocol == "icmp"
    assert (record.src_port, record.dst_port) == (None, None)
    assert (record.src_ip, record.dst_ip) == (A4, B4)
    assert record.payload_bytes == len(embedded)


@pytest.mark.parametrize(
    "message",
    [ICMPv6ND_NS(tgt=B6), ICMPv6EchoRequest(data=b"ping")],
    ids=["neighbour-solicitation", "echo"],
)
def test_d14_icmpv6_counts_as_icmp(tmp_path: Path, message: Packet) -> None:
    record = only(tmp_path / "icmpv6.pcap", ETHER / IPv6(src=A6, dst=B6) / message)
    assert record.protocol == "icmp"
    assert (record.src_port, record.dst_port) == (None, None)
    assert record.ip_header_bytes == 40


# Scenarios


def test_ipv6_tcp_is_read_like_ipv4(tmp_path: Path) -> None:
    packet = (
        ETHER
        / IPv6(src=A6, dst=B6)
        / TCP(sport=40000, dport=443, flags="S")
        / Raw(b"x" * 7)
    )
    record = only(tmp_path / "v6.pcap", packet)
    assert record == PacketRecord(
        timestamp=record.timestamp,
        src_ip=A6,
        dst_ip=B6,
        src_port=40000,
        dst_port=443,
        protocol="tcp",
        tcp_flags=0x02,
        payload_bytes=7,
        ip_header_bytes=40,
        transport_header_bytes=20,
    )


def test_ipv6_extension_headers_count_as_ip_header(tmp_path: Path) -> None:
    hop_by_hop = IPv6ExtHdrHopByHop(
        options=[RouterAlert()]
    )  # 8 bytes, as in an MLD report
    mld = only(
        tmp_path / "mld.pcap",
        ETHER / IPv6(src=A6, dst=B6) / hop_by_hop / ICMPv6MLReport(),
    )
    assert (mld.protocol, mld.ip_header_bytes) == ("icmp", 48)
    fragment = (
        ETHER
        / IPv6(src=A6, dst=B6)
        / IPv6ExtHdrFragment(m=0)
        / UDP(sport=1, dport=2)
        / Raw(bytes(3))
    )
    record = only(tmp_path / "fragment.pcap", fragment)
    assert (record.protocol, record.ip_header_bytes, record.payload_bytes) == (
        "udp",
        48,
        3,
    )


def test_udp_has_ports_and_an_eight_byte_header(tmp_path: Path) -> None:
    packet = ETHER / IP(src=A4, dst=B4) / UDP(sport=5353, dport=53) / Raw(bytes(30))
    record = only(tmp_path / "udp.pcap", packet)
    assert (record.protocol, record.src_port, record.dst_port) == ("udp", 5353, 53)
    assert (record.transport_header_bytes, record.payload_bytes, record.tcp_flags) == (
        8,
        30,
        0,
    )


def test_tcp_options_count_as_transport_header(tmp_path: Path) -> None:
    tcp = TCP(
        sport=40000,
        dport=80,
        flags="S",
        options=[("MSS", 1460), ("NOP", None), ("NOP", None)],
    )
    record = only(tmp_path / "options.pcap", ETHER / IP(src=A4, dst=B4) / tcp)
    assert (record.transport_header_bytes, record.payload_bytes) == (28, 0)


def test_an_ip_packet_of_another_protocol_is_kept_as_other(tmp_path: Path) -> None:
    packet = ETHER / IP(src=A4, dst="224.0.0.22", proto=2) / Raw(bytes(8))  # IGMP
    record = only(tmp_path / "igmp.pcap", packet)
    assert record.protocol == "other"
    assert (record.src_port, record.dst_port) == (None, None)
    assert (
        record.ip_header_bytes,
        record.transport_header_bytes,
        record.payload_bytes,
    ) == (20, 0, 8)


def test_the_timestamp_is_epoch_seconds(tmp_path: Path) -> None:
    packet = ETHER / IP(src=A4, dst=B4) / UDP(sport=1, dport=2)
    packet.time = 1_700_000_000.25
    assert only(tmp_path / "time.pcap", packet).timestamp == 1_700_000_000.25


def test_a_big_endian_nanosecond_pcap_is_read(tmp_path: Path) -> None:
    path = tmp_path / "big.pcap"
    packet = ETHER / IP(src=A4, dst=B4) / UDP(sport=1, dport=2) / Raw(bytes(4))
    with PcapWriter(str(path), endianness=">", nano=True) as writer:
        writer.write([packet, packet])
    assert path.read_bytes()[:4] == bytes.fromhex("a1b23c4d")
    records, counts = read(path)
    assert [r.payload_bytes for r in records] == [4, 4]
    assert counts == {"packets_read": 2, "non_ip_dropped": 0}


def test_a_pcapng_capture_is_read(tmp_path: Path) -> None:
    path = tmp_path / "capture.pcapng"
    with PcapNgWriter(str(path)) as writer:
        writer.write(ETHER / IP(src=A4, dst=B4) / TCP(sport=1, dport=2) / Raw(bytes(5)))
        writer.write(ETHER / ARP(psrc=A4, pdst=B4))
    records, counts = read(path)
    assert [(r.protocol, r.payload_bytes) for r in records] == [("tcp", 5)]
    assert counts == {"packets_read": 2, "non_ip_dropped": 1}


def pcapng_section(order: str, packets: Iterable[Packet]) -> bytes:
    """A pcapng section in byte order `order` ("<" or ">"): a section header, one
    Ethernet interface, and one enhanced packet block per packet. scapy's pcapng
    writer writes only the host byte order."""
    section = struct.pack(order + "IIIHHqI", 0x0A0D0D0A, 28, 0x1A2B3C4D, 1, 0, -1, 28)
    section += struct.pack(order + "IIHHII", 1, 20, 1, 0, 0, 20)
    for packet in packets:
        data = bytes(packet)
        padding = bytes(-len(data) % 4)
        length = 32 + len(data) + len(padding)
        section += struct.pack(
            order + "IIIIIII", 6, length, 0, 0, 1, len(data), len(data)
        )
        section += data + padding + struct.pack(order + "I", length)
    return section


def test_a_big_endian_pcapng_is_read(tmp_path: Path) -> None:
    path = tmp_path / "big.pcapng"
    packet = ETHER / IP(src=A4, dst=B4) / TCP(sport=1, dport=2) / Raw(bytes(5))
    path.write_bytes(pcapng_section(">", [packet, packet]))
    records, counts = read(path)
    assert [(r.protocol, r.payload_bytes) for r in records] == [("tcp", 5), ("tcp", 5)]
    assert counts == {"packets_read": 2, "non_ip_dropped": 0}


def test_a_pcapng_with_sections_of_both_byte_orders_is_read(tmp_path: Path) -> None:
    path = tmp_path / "mixed.pcapng"
    first = ETHER / IP(src=A4, dst=B4) / UDP(sport=1, dport=2) / Raw(bytes(3))
    second = ETHER / IP(src=A4, dst=B4) / TCP(sport=1, dport=2) / Raw(bytes(5))
    path.write_bytes(
        pcapng_section("<", [first])
        + pcapng_section(">", [second, second])
        + pcapng_section("<", [first])
    )
    records, counts = read(path)
    assert [(r.protocol, r.payload_bytes) for r in records] == [
        ("udp", 3),
        ("tcp", 5),
        ("tcp", 5),
        ("udp", 3),
    ]
    assert counts == {"packets_read": 4, "non_ip_dropped": 0}


@pytest.mark.parametrize(
    ("name", "packets"),
    [
        ("one_http_stream_fin.pcap", FIN_PACKETS),
        ("one_http_stream_rst.pcap", RST_PACKETS),
        ("pcapng-example.pcapng", PCAPNG_PACKETS),
    ],
)
def test_the_real_captures_give_the_hand_counted_packets(
    name: str, packets: int
) -> None:
    records, counts = read(REAL / name)
    assert counts == {"packets_read": packets, "non_ip_dropped": 0}
    assert len(records) == packets
    assert {r.protocol for r in records} == {"tcp"}


def test_the_real_pcapng_holds_ipv6_and_ip_inside_a_vlan_and_pppoe_is_found() -> None:
    v6 = [r for r in read(REAL / "pcapng-example.pcapng")[0] if ":" in r.src_ip]
    assert len(v6) == PCAPNG_IPV6_PACKETS
    # one_http_stream_rst.pcap carries IP inside 802.1Q and PPPoE: none is dropped.
    assert read(REAL / "one_http_stream_rst.pcap")[1]["non_ip_dropped"] == 0


# Rejections


def test_a_missing_file_raises_an_input_error_naming_the_path(tmp_path: Path) -> None:
    path = tmp_path / "missing.pcap"
    with pytest.raises(InputError, match="missing.pcap"):
        read(path)


def test_a_file_that_is_not_a_capture_raises_an_input_error_naming_the_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.pcap"
    path.write_text("not a capture\n")
    with pytest.raises(InputError, match="notes.pcap.*not a pcap or pcapng"):
        read(path)


def test_a_capture_scapy_cannot_read_raises_an_input_error_naming_the_path(
    tmp_path: Path,
) -> None:
    # A pcapng section header of 16 bytes: whole, but too short for its fields.
    path = tmp_path / "header.pcapng"
    path.write_bytes(
        struct.pack("<II4sI", 0x0A0D0D0A, 16, bytes.fromhex("4d3c2b1a"), 16)
    )
    with pytest.raises(InputError, match="header.pcapng.*scapy cannot read"):
        read(path)


@pytest.mark.parametrize(
    ("name", "cut"),
    [
        ("one_http_stream_fin.pcap", 10),  # inside the last packet's data
        ("one_http_stream_fin.pcap", 1500),  # several packets short
        ("pcapng-example.pcapng", 30),  # inside the last block
    ],
)
def test_a_truncated_capture_raises_an_input_error(
    tmp_path: Path, name: str, cut: int
) -> None:
    data = (REAL / name).read_bytes()
    path = tmp_path / name
    path.write_bytes(data[:-cut])
    with pytest.raises(InputError, match=f"{name}.*truncated"):
        read(path)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("header-only.pcap", bytes.fromhex("d4c3b2a1") + bytes(10)),
        (
            "zero-length.pcapng",
            (REAL / "pcapng-example.pcapng").read_bytes()[:28] + bytes(12),
        ),
    ],
)
def test_a_cut_file_header_or_a_zero_block_length_raises_an_input_error(
    tmp_path: Path, name: str, content: bytes
) -> None:
    path = tmp_path / name
    path.write_bytes(content)
    with pytest.raises(InputError, match=f"{name}.*truncated or corrupt"):
        read(path)


# read: a capture becomes a FlowTable with counts and warnings.

IDENTITY_AND_FEATURES = [
    field.name for field in record.FIELDS if field.role in (Role.IDENTITY, Role.FEATURE)
]


def test_the_capture_module_is_a_reader_named_capture() -> None:
    assert isinstance(capture, Reader)
    assert capture.name == "capture"


def test_read_returns_a_checked_table_with_counts_and_no_label(tmp_path: Path) -> None:
    path = write(
        tmp_path / "flows.pcap",
        [
            ETHER / IP(src=A4, dst=B4) / TCP(sport=40000, dport=80, flags="S"),
            ETHER / ARP(psrc=A4, pdst=B4),
            ETHER / IP(src=B4, dst=A4) / TCP(sport=80, dport=40000, flags="SA"),
            ETHER / IP(src=A4, dst=B4) / ICMP(),
        ],
    )
    result = capture.read(path, FlowSettings(), label_col="Attack")
    assert isinstance(result, TableResult)
    check_record(result.table)
    frame = result.table.to_frame()
    assert list(frame.columns) == IDENTITY_AND_FEATURES
    assert "label" not in result.table.roles
    assert result.counts == {"packets_read": 4, "non_ip_dropped": 1}
    assert result.warnings == ()
    assert list(frame["protocol"]) == ["tcp", "icmp"]
    assert list(frame["fwd_packets"]) == [1, 1]
    assert list(frame["bwd_packets"]) == [1, 0]
    assert pd.isna(frame["src_port"][1]) and pd.isna(frame["dst_port"][1])


def test_read_applies_the_flow_settings(tmp_path: Path) -> None:
    packets = []
    for second in (0, 5, 20):
        packet = ETHER / IP(src=A4, dst=B4) / UDP(sport=1, dport=2)
        packet.time = second
        packets.append(packet)
    path = write(tmp_path / "idle.pcap", packets)
    assert len(capture.read(path, FlowSettings()).table.to_frame()) == 1
    assert (
        len(capture.read(path, FlowSettings(idle_timeout=10.0)).table.to_frame()) == 2
    )


def test_a_capture_of_only_non_ip_packets_gives_an_empty_checked_table(
    tmp_path: Path,
) -> None:
    path = write(tmp_path / "arp.pcap", [ETHER / ARP(psrc=A4, pdst=B4)])
    result = capture.read(path, FlowSettings())
    check_record(result.table)
    assert len(result.table.to_frame()) == 0
    assert result.counts == {"packets_read": 1, "non_ip_dropped": 1}


@pytest.mark.parametrize(
    ("name", "packets", "flows", "out_of_range"),
    [
        ("one_http_stream_fin.pcap", FIN_PACKETS, FIN_FLOWS, 0),
        ("one_http_stream_rst.pcap", RST_PACKETS, RST_FLOWS, 0),
        # The 14 flows of the Linux cooked interface start near 1.8e13 seconds.
        ("pcapng-example.pcapng", PCAPNG_PACKETS, PCAPNG_FLOWS, 14),
    ],
)
def test_read_gives_the_hand_worked_flow_counts_of_the_real_captures(
    name: str, packets: int, flows: int, out_of_range: int
) -> None:
    result = capture.read(REAL / name, FlowSettings())
    frame = result.table.to_frame()
    assert len(frame) == flows
    assert result.counts == {"packets_read": packets, "non_ip_dropped": 0}
    assert int(frame["start_time"].isna().sum()) == out_of_range
    assert len(result.warnings) == (1 if out_of_range else 0)
    assert int((frame["fwd_packets"] + frame["bwd_packets"]).sum()) == packets


def test_read_of_a_missing_file_raises_an_input_error_naming_the_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(InputError, match="missing.pcap"):
        capture.read(tmp_path / "missing.pcap", FlowSettings())
