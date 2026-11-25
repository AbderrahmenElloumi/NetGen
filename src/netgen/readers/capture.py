"""The capture reader: pcap and pcapng, read by scapy into packet records.

This module is the only place that imports scapy. IPv4 and IPv6 are read
alike. TCP, UDP and ICMP (ICMPv6 counts as ICMP) are kept; any other IP
protocol is kept as `other`; non-IP packets are dropped and counted.
"""

import struct
from collections.abc import Generator
from pathlib import Path
from typing import BinaryIO

# scapy.all loads every layer binding, so IP is found inside 802.1Q, PPPoE and
# the other link layers scapy knows.
import scapy.all  # noqa: F401
from scapy.error import Scapy_Exception
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import (
    IPv6,
    IPv6ExtHdrDestOpt,
    IPv6ExtHdrFragment,
    IPv6ExtHdrHopByHop,
    IPv6ExtHdrRouting,
)
from scapy.packet import NoPayload, Packet
from scapy.utils import PcapReader

from netgen.aggregation import PacketProtocol, PacketRecord, flow_frame, form_flows
from netgen.errors import InputError
from netgen.record import FlowSettings
from netgen.table import FlowTable, TableResult, check_record

name = "capture"

# The first four bytes of a pcap file give its byte order; microsecond and
# nanosecond files differ only in the magic.
PCAP_BYTE_ORDER = {
    bytes.fromhex("d4c3b2a1"): "<",
    bytes.fromhex("a1b2c3d4"): ">",
    bytes.fromhex("4d3cb2a1"): "<",
    bytes.fromhex("a1b23c4d"): ">",
}
PCAPNG_MAGIC = bytes.fromhex("0a0d0d0a")
PCAPNG_LITTLE_ENDIAN = bytes.fromhex("4d3c2b1a")
PCAP_FILE_HEADER_BYTES = 24
PCAP_RECORD_HEADER_BYTES = 16
PCAPNG_BLOCK_MIN_BYTES = 12

IPV6_HEADER_BYTES = 40
UDP_HEADER_BYTES = 8
ICMP_HEADER_BYTES = 8
ICMP_NUMBERS = (1, 58)  # ICMP, ICMPv6
IPV6_EXTENSION_HEADERS = (
    IPv6ExtHdrHopByHop,
    IPv6ExtHdrRouting,
    IPv6ExtHdrDestOpt,
    IPv6ExtHdrFragment,
)
IPV6_FRAGMENT_HEADER_BYTES = 8

Counts = dict[str, int]


def sniff(path: Path) -> bool:
    """True when the file starts with a pcap magic (either byte order,
    microsecond or nanosecond) or the pcapng section header. Never raises."""
    try:
        with path.open("rb") as file:
            magic = file.read(4)
    except OSError:
        return False
    return magic in PCAP_BYTE_ORDER or magic == PCAPNG_MAGIC


def read(
    path: Path, flow_settings: FlowSettings, label_col: str = "Label"
) -> TableResult:
    """The flows of a capture, in the flow record, checked at the seam.

    `label_col` is ignored: a capture is unlabelled, so the table has no label
    column. The counts are `packets_read` and `non_ip_dropped`. Raises
    `InputError` as `read_packets` does.
    """
    counts: Counts = {}

    def packets() -> Generator[PacketRecord, None, None]:
        counts.update((yield from read_packets(path)))

    flows = form_flows(packets(), flow_settings)
    frame, warnings = flow_frame(flows)
    table = FlowTable.from_frame(frame)
    check_record(table)
    return TableResult(table, counts, warnings)


def read_packets(path: Path) -> Generator[PacketRecord, None, Counts]:
    """Yield one packet record per IP packet, streaming the capture.

    Returns the counts `packets_read` and `non_ip_dropped`. Raises
    `InputError` naming the path when the file is missing, is not a capture
    scapy can read, or is truncated.
    """
    _check_complete(path)
    counts = {"packets_read": 0, "non_ip_dropped": 0}
    try:
        with PcapReader(str(path)) as reader:
            for packet in reader:
                counts["packets_read"] += 1
                record = _record(packet)
                if record is None:
                    counts["non_ip_dropped"] += 1
                else:
                    yield record
    except Scapy_Exception as error:
        raise InputError(
            f"{path}: scapy cannot read this capture ({error})."
        ) from error
    return counts


def _check_complete(path: Path) -> None:
    """Raise `InputError` unless every record or block ends inside the file.

    scapy logs a warning and stops early on a truncated capture, so the
    reader walks the record lengths itself before reading.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            magic = file.read(4)
            if magic in PCAP_BYTE_ORDER:
                complete = _pcap_complete(file, PCAP_BYTE_ORDER[magic], size)
            elif magic == PCAPNG_MAGIC:
                complete = _pcapng_complete(file, size)
            else:
                raise InputError(
                    f"{path}: not a pcap or pcapng capture. The capture reader reads only those two formats."
                )
    except OSError as error:
        raise InputError(
            f"{path}: cannot open the capture ({error.strerror})."
        ) from error
    if not complete:
        raise InputError(
            f"{path}: the capture is truncated or corrupt: a record length runs past the end "
            "of the file. Use a complete capture."
        )


def _pcap_complete(file: BinaryIO, order: str, size: int) -> bool:
    position = PCAP_FILE_HEADER_BYTES
    while position + PCAP_RECORD_HEADER_BYTES <= size:
        file.seek(position + 8)  # the captured length follows two 4-byte time fields
        captured: int = struct.unpack(order + "I", file.read(4))[0]
        position += PCAP_RECORD_HEADER_BYTES + captured
    return position == size


def _pcapng_complete(file: BinaryIO, size: int) -> bool:
    position, order = 0, "<"
    while position + PCAPNG_BLOCK_MIN_BYTES <= size:
        file.seek(position)
        head = file.read(PCAPNG_BLOCK_MIN_BYTES)
        if head[:4] == PCAPNG_MAGIC:  # a section header sets the byte order
            order = "<" if head[8:12] == PCAPNG_LITTLE_ENDIAN else ">"
        length: int = struct.unpack(order + "I", head[4:8])[0]
        if length < PCAPNG_BLOCK_MIN_BYTES:  # a corrupt length; 0 would never advance
            return False
        position += length
    return position == size


def _record(packet: Packet) -> PacketRecord | None:
    """The packet record of the outermost IP layer, or None for a non-IP packet."""
    ip = packet
    while not isinstance(ip, IP | IPv6):
        if isinstance(ip, NoPayload):
            return None
        ip = ip.payload
    if isinstance(ip, IP):
        total, number, ip_header, transport = ip.len, ip.proto, ip.ihl * 4, ip.payload
    else:
        total, number, ip_header, transport = (
            IPV6_HEADER_BYTES + ip.plen,
            ip.nh,
            IPV6_HEADER_BYTES,
            ip.payload,
        )
        while isinstance(transport, IPV6_EXTENSION_HEADERS):
            number = transport.nh
            if isinstance(transport, IPv6ExtHdrFragment):
                ip_header += IPV6_FRAGMENT_HEADER_BYTES
            else:
                ip_header += 8 * (
                    transport.len + 1
                )  # the length field counts 8-byte units after the first
            transport = transport.payload
    # The outer transport decides the protocol: an ICMP error that embeds a
    # TCP header is ICMP.
    protocol: PacketProtocol
    src_port: int | None = None
    dst_port: int | None = None
    flags = 0
    if isinstance(transport, TCP):
        protocol, src_port, dst_port = "tcp", transport.sport, transport.dport
        flags, transport_header = int(transport.flags), transport.dataofs * 4
    elif isinstance(transport, UDP):
        protocol, src_port, dst_port, transport_header = (
            "udp",
            transport.sport,
            transport.dport,
            UDP_HEADER_BYTES,
        )
    elif number in ICMP_NUMBERS:
        protocol, transport_header = "icmp", ICMP_HEADER_BYTES
    else:
        protocol, transport_header = "other", 0
    return PacketRecord(
        timestamp=float(packet.time),
        src_ip=ip.src,
        dst_ip=ip.dst,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        tcp_flags=flags,
        payload_bytes=max(0, total - ip_header - transport_header),
        ip_header_bytes=ip_header,
        transport_header_bytes=transport_header,
    )
