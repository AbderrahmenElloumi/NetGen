"""Write demo_capture.pcap, a synthetic Ethernet capture for the demo and the tests.

The traffic is made up and every address is a documentation address. The mix:
TCP sessions closed by a two-sided FIN (some over IPv6), TCP sessions closed by
an RST, DNS queries over UDP, pings, a few ARP frames, one conversation with an
idle gap longer than the 120 s idle timeout, and a port-scan burst. The design
counts below give the number of flows by construction.

The output is deterministic: one seeded generator, no clock, and an explicit
value for every field scapy would otherwise pick (IP identification, sequence
numbers, ports, MAC addresses, timestamps, link type). Two runs write
identical bytes.

    python data/samples/make_demo_capture.py data/samples/demo_capture.pcap
"""

import argparse
import random
from collections.abc import Callable, Sequence
from pathlib import Path

from scapy.data import DLT_EN10MB
from scapy.layers.dns import DNS, DNSQR, DNSRR
from scapy.layers.inet import ICMP, IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import ARP, Ether
from scapy.packet import Packet, Raw
from scapy.utils import PcapWriter

SEED = 2026
START_US = 1_767_225_600_000_000  # 2026-01-01T00:00:00Z
SPAN_US = 600_000_000  # sessions start within the first ten minutes

# The design counts. Each TCP client port carries SESSIONS_PER_PORT sessions
# one after the other, well inside the idle timeout, so a FIN or an RST that
# did not end its flow would merge them into one flow.
SESSIONS_PER_PORT = 2
FIN_PORTS_V4 = 130
FIN_PORTS_V6 = 10
RST_PORTS = 15
DNS_QUERIES = 40
PINGS = 8
ARP_FRAMES = 4
SCAN_PORTS = 40
IDLE_FLOWS = 2  # one conversation, split by the idle gap
IDLE_GAP_US = 150_000_000
EXPECTED_FLOWS = (
    SESSIONS_PER_PORT * (FIN_PORTS_V4 + FIN_PORTS_V6 + RST_PORTS)
    + DNS_QUERIES
    + PINGS
    + IDLE_FLOWS
    + SCAN_PORTS
)

WEB_PORT = 443
RESET_PORT = 8080
DNS_PORT = 53
IDLE_PORT = 22

# Addresses from RFC 5737 (IPv4), RFC 3849 (IPv6) and RFC 7042 (MAC).
CLIENTS_V4 = [f"192.0.2.{n}" for n in range(10, 50)]
SERVERS_V4 = [f"198.51.100.{n}" for n in range(10, 20)]
CLIENTS_V6 = [f"2001:db8:1::{n:x}" for n in range(16, 24)]
SERVERS_V6 = ["2001:db8:2::10", "2001:db8:2::11"]
GATEWAY = "192.0.2.1"
RESOLVER = "198.51.100.53"
PING_CLIENTS = [f"192.0.2.{n}" for n in range(200, 200 + PINGS)]
PING_TARGET = "198.51.100.1"
IDLE_CLIENT, IDLE_SERVER, IDLE_CLIENT_PORT = "192.0.2.100", "198.51.100.22", 50022
SCANNER, SCAN_TARGET, SCANNER_PORT = "203.0.113.66", "203.0.113.10", 54321
IDLE_START_US = 100_000_000
SCAN_START_US = 300_000_000
HOST_MAC = "00:00:5e:00:53:01"
GATEWAY_MAC = "00:00:5e:00:53:fe"
BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"
WINDOW = 64240


class Capture:
    """The frames written so far, each with its time in microseconds from the start."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.frames: list[tuple[int, int, Packet]] = []
        self.ip_id = 0

    def add(self, time_us: int, frame: Packet) -> None:
        self.frames.append((time_us, len(self.frames), frame))

    def ip(self, src: str, dst: str, from_host: bool) -> Packet:
        """An Ethernet frame with the IP header from `src` to `dst`."""
        ether = Ether(
            src=HOST_MAC if from_host else GATEWAY_MAC,
            dst=GATEWAY_MAC if from_host else HOST_MAC,
        )
        if ":" in src:
            return ether / IPv6(src=src, dst=dst, hlim=64)
        self.ip_id = (self.ip_id + 1) % 65536
        return ether / IP(src=src, dst=dst, id=self.ip_id, flags="DF", ttl=64)

    def packets(self) -> list[Packet]:
        """Every frame in time order, stamped with its capture time."""
        ordered = []
        for time_us, _, frame in sorted(self.frames, key=lambda item: item[:2]):
            frame.time = (START_US + time_us) / 1_000_000
            ordered.append(frame)
        return ordered


class Connection:
    """One TCP connection; `send` advances its clock and both sequence numbers."""

    def __init__(
        self,
        capture: Capture,
        client: tuple[str, int],
        server: tuple[str, int],
        start_us: int,
    ) -> None:
        self.capture, self.client, self.server = capture, client, server
        self.clock_us = start_us
        rng = capture.rng
        self.seq = {True: rng.randrange(2**32), False: rng.randrange(2**32)}

    def send(self, from_client: bool, flags: str, payload: int = 0) -> None:
        src, dst = (
            (self.client, self.server) if from_client else (self.server, self.client)
        )
        ack = self.seq[not from_client] if "A" in flags else 0
        segment = TCP(
            sport=src[1],
            dport=dst[1],
            seq=self.seq[from_client],
            ack=ack,
            flags=flags,
            window=WINDOW,
        )
        frame = self.capture.ip(src[0], dst[0], from_client) / segment
        if payload:
            frame = frame / Raw(bytes(payload))
        self.capture.add(self.clock_us, frame)
        advance = payload + ("S" in flags) + ("F" in flags)
        self.seq[from_client] = (self.seq[from_client] + advance) % 2**32
        self.clock_us += self.capture.rng.randint(1_000, 40_000)

    def open(self) -> None:
        self.send(True, "S")
        self.send(False, "SA")
        self.send(True, "A")

    def exchange(self) -> None:
        rng = self.capture.rng
        self.send(True, "PA", rng.randint(40, 300))
        self.send(False, "PA", rng.randint(100, 800))

    def close(self) -> None:
        self.send(True, "FA")
        self.send(False, "FA")
        self.send(True, "A")


def fin_session(connection: Connection) -> None:
    """8 packets: the handshake, one request and response, the FIN close."""
    connection.open()
    connection.exchange()
    connection.close()


def rst_session(connection: Connection) -> None:
    """5 packets: the handshake, one request, the server's RST."""
    connection.open()
    connection.send(True, "PA", connection.capture.rng.randint(40, 300))
    connection.send(False, "RA")


def sessions_on_one_port(
    capture: Capture,
    client: tuple[str, int],
    server: tuple[str, int],
    session: Callable[[Connection], None],
) -> None:
    rng = capture.rng
    start_us = rng.randrange(SPAN_US)
    for _ in range(SESSIONS_PER_PORT):
        connection = Connection(capture, client, server, start_us)
        session(connection)
        start_us = connection.clock_us + rng.randint(500_000, 30_000_000)


def dns_query(capture: Capture, client: str, port: int, number: int) -> None:
    rng = capture.rng
    time_us = rng.randrange(SPAN_US)
    name = f"host{number}.example."
    query = DNS(id=rng.randrange(65536), rd=1, qd=DNSQR(qname=name))
    answer = DNSRR(rrname=name, type="A", ttl=300, rdata=rng.choice(SERVERS_V4))
    reply = DNS(id=query.id, qr=1, rd=1, ra=1, qd=DNSQR(qname=name), an=answer)
    udp = UDP(sport=port, dport=DNS_PORT)
    capture.add(time_us, capture.ip(client, RESOLVER, True) / udp / query)
    udp = UDP(sport=DNS_PORT, dport=port)
    capture.add(
        time_us + rng.randint(1_000, 30_000),
        capture.ip(RESOLVER, client, False) / udp / reply,
    )


def ping(capture: Capture, client: str) -> None:
    rng = capture.rng
    time_us, ident = rng.randrange(SPAN_US), rng.randrange(65536)
    request = ICMP(type="echo-request", id=ident, seq=1) / Raw(bytes(32))
    reply = ICMP(type="echo-reply", id=ident, seq=1) / Raw(bytes(32))
    capture.add(time_us, capture.ip(client, PING_TARGET, True) / request)
    capture.add(
        time_us + rng.randint(1_000, 30_000),
        capture.ip(PING_TARGET, client, False) / reply,
    )


def idle_conversation(capture: Capture) -> None:
    """One SSH-like connection that goes quiet for IDLE_GAP_US, then closes.

    The first flow is the handshake and one exchange (5 packets); the second
    is one exchange and the FIN close (5 packets).
    """
    connection = Connection(
        capture,
        (IDLE_CLIENT, IDLE_CLIENT_PORT),
        (IDLE_SERVER, IDLE_PORT),
        IDLE_START_US,
    )
    connection.open()
    connection.send(True, "PA", 48)
    last_us = connection.clock_us
    connection.send(False, "PA", 48)
    connection.clock_us = last_us + IDLE_GAP_US
    connection.exchange()
    connection.close()


def port_scan(capture: Capture) -> None:
    """One SYN per port; a closed port answers with an RST, a filtered one not at all."""
    rng = capture.rng
    time_us = SCAN_START_US
    for port in sorted(rng.sample(range(1, 1025), SCAN_PORTS)):
        probe = Connection(
            capture, (SCANNER, SCANNER_PORT), (SCAN_TARGET, port), time_us
        )
        probe.send(True, "S")
        if rng.random() < 0.8:
            probe.send(False, "RA")
        time_us += rng.randint(200, 2_000)


def arp_request(capture: Capture, client: str) -> None:
    frame = Ether(src=HOST_MAC, dst=BROADCAST_MAC) / ARP(
        op="who-has", hwsrc=HOST_MAC, psrc=client, pdst=GATEWAY
    )
    capture.add(capture.rng.randrange(SPAN_US), frame)


def demo_packets() -> list[Packet]:
    """The packets of the demo capture, in time order."""
    rng = random.Random(SEED)
    capture = Capture(rng)
    ports = iter(
        rng.sample(
            range(49152, 65536), FIN_PORTS_V4 + FIN_PORTS_V6 + RST_PORTS + DNS_QUERIES
        )
    )
    for clients, servers, count in (
        (CLIENTS_V4, SERVERS_V4, FIN_PORTS_V4),
        (CLIENTS_V6, SERVERS_V6, FIN_PORTS_V6),
    ):
        for _ in range(count):
            client = (rng.choice(clients), next(ports))
            server = (rng.choice(servers), WEB_PORT)
            sessions_on_one_port(capture, client, server, fin_session)
    for _ in range(RST_PORTS):
        client = (rng.choice(CLIENTS_V4), next(ports))
        server = (rng.choice(SERVERS_V4), RESET_PORT)
        sessions_on_one_port(capture, client, server, rst_session)
    for number in range(DNS_QUERIES):
        dns_query(capture, rng.choice(CLIENTS_V4), next(ports), number)
    for address in PING_CLIENTS:
        ping(capture, address)
    idle_conversation(capture)
    port_scan(capture)
    for _ in range(ARP_FRAMES):
        arp_request(capture, rng.choice(CLIENTS_V4))
    return capture.packets()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument("out_path", type=Path, help="where to write the capture")
    args = parser.parse_args(argv)
    with PcapWriter(str(args.out_path), linktype=DLT_EN10MB, endianness="<") as writer:
        writer.write(demo_packets())


if __name__ == "__main__":
    main()
