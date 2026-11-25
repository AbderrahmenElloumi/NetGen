"""Write the scrubbed PcapPlusPlus captures into real/.

Run it once, in the dev container, against a folder that holds the three
captures from `Tests/Pcap++Test/PcapExamples/` of the PcapPlusPlus repository:

    docker compose run --rm dev python data/samples/make_pcpp_captures.py <folder>

Each packet keeps its link layer, timestamp, lengths, ports, TCP flags,
sequence numbers and options. The scrub changes three things:

1. Addresses take documentation values in order of first appearance, with one
   mapping across the three files: IPv4 from 192.0.2.0/24, IPv6 from
   2001:db8::/32, MAC addresses from 00:00:5e:00:53:00/24.
2. Every payload byte and every padding byte becomes zero; lengths stay.
3. Checksums are recomputed, and the pcapng is written fresh, so its section
   and interface options (host name, a TLS key comment, capture filter) and
   its per-packet comments are gone.
"""

import argparse
import ipaddress
from collections.abc import Sequence
from pathlib import Path

# scapy.all loads every layer binding; one capture carries IP inside 802.1Q and PPPoE.
import scapy.all  # noqa: F401
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import CookedLinux, Ether
from scapy.packet import Packet, Padding, Raw
from scapy.utils import PcapNgWriter, PcapReader, PcapWriter

FILES = (
    "one_http_stream_fin.pcap",
    "one_http_stream_rst.pcap",
    "pcapng-example.pcapng",
)
V4_RANGE = ipaddress.ip_network("192.0.2.0/24")
V6_RANGE = ipaddress.ip_network("2001:db8::/32")
MAC_PREFIX = "00:00:5e:00:53:"
IPV6_HEADER_BYTES = 40


class Mapping:
    """One documentation value per real value, in order of first appearance."""

    def __init__(self) -> None:
        self.v4: dict[str, str] = {}
        self.v6: dict[str, str] = {}
        self.macs: dict[str, str] = {}

    def address(self, real: str) -> str:
        v4 = ipaddress.ip_address(real).version == 4
        table, base = (self.v4, V4_RANGE) if v4 else (self.v6, V6_RANGE)
        if real not in table:
            table[real] = str(base.network_address + len(table) + 1)
        return table[real]

    def mac(self, real: str) -> str:
        key = real.lower()
        if key not in self.macs:
            self.macs[key] = f"{MAC_PREFIX}{len(self.macs):02x}"
        return self.macs[key]


def scrub(packet: Packet, mapping: Mapping) -> Packet:
    if Ether in packet:
        packet[Ether].src = mapping.mac(packet[Ether].src)
        packet[Ether].dst = mapping.mac(packet[Ether].dst)
    if CookedLinux in packet:
        size = packet[CookedLinux].lladdrlen
        real = ":".join(f"{b:02x}" for b in bytes(packet[CookedLinux].src)[:size])
        packet[CookedLinux].src = bytes.fromhex(
            mapping.mac(real).replace(":", "")
        ).ljust(8, b"\0")
    ip = packet.getlayer(IP) or packet.getlayer(IPv6)
    transport = packet.getlayer(TCP) or packet.getlayer(UDP)
    if ip is None or transport is None:
        raise SystemExit(f"not a TCP or UDP packet over IP: {packet.summary()}")
    # A snapped packet holds fewer bytes than its IP total length.
    captured = len(bytes(ip))
    total = ip.len if isinstance(ip, IP) else IPV6_HEADER_BYTES + ip.plen
    headers = captured - len(bytes(transport.payload))
    ip.src, ip.dst = mapping.address(ip.src), mapping.address(ip.dst)
    if isinstance(ip, IP):
        del ip.chksum
    del transport.chksum
    transport.remove_payload()
    zeros = Raw(bytes(min(total, captured) - headers)) / Padding(
        bytes(max(0, captured - total))
    )
    transport.add_payload(zeros)
    # The pcapng writer starts one interface block per name, with that name's link type.
    packet.sniffed_on = type(packet).__name__
    packet.comments = None
    return packet


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write the scrubbed PcapPlusPlus captures."
    )
    parser.add_argument("source", type=Path, help="folder holding the three captures")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "real")
    args = parser.parse_args(argv)
    mapping = Mapping()
    for name in FILES:
        target = args.out / name
        writer = (
            PcapNgWriter(str(target))
            if name.endswith(".pcapng")
            else PcapWriter(str(target))
        )
        with PcapReader(str(args.source / name)) as reader, writer:
            for packet in reader:
                # Field writes reset `wirelen`, and the pcapng writer would round the
                # time through a float: both are taken before the scrub. The pcap writer
                # takes whole seconds from `sec` and microseconds from `usec`.
                wirelen, time = packet.wirelen, packet.time
                scrubbed = scrub(packet, mapping)
                if not writer.header_present:
                    writer.write_header(scrubbed)
                usec = int(time % 1 * 1_000_000)
                writer.write_packet(scrubbed, sec=time, usec=usec, wirelen=wirelen)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
