"""Write demo_flows.csv, a synthetic flow export in the CIC-IDS2017 layout.

The flows are made up. Each flow is simulated packet by packet and every column
is computed from its packets, so totals, rates and inter-arrival times agree
within a row. Times are integer microseconds, as in the real files. The output
is deterministic: two runs write identical bytes.

    python data/samples/make_demo_flows.py --out data/samples/demo_flows.csv
"""

import argparse
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path

SEED = 2017
ACTIVITY_TIMEOUT_US = 5_000_000
TCP, UDP = 6, 17

# The header of the CIC-IDS2017 files in TrafficLabelling, quirks included:
# leading spaces, missing spaces after some commas, two "Fwd Header Length".
HEADER = (
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

Number = int | float


@dataclass(frozen=True)
class Packet:
    time_us: int
    forward: bool
    payload: int
    header: int
    flags: frozenset[str]
    window: int


@dataclass
class Flow:
    src: str
    sport: int
    dst: str
    dport: int
    protocol: int
    label: str
    packets: list[Packet] = field(default_factory=list)
    clock_us: int = 0

    def send(
        self,
        forward: bool,
        gap_us: int,
        flags: str = "",
        payload: int = 0,
        header: int = 20,
        window: int = -1,
    ) -> None:
        self.clock_us += gap_us
        self.packets.append(
            Packet(
                self.clock_us,
                forward,
                payload,
                header,
                frozenset(flags.split()),
                window,
            )
        )


def stats(values: Sequence[int]) -> tuple[Number, Number, Number, Number, Number]:
    """Max, min, mean, sample standard deviation and sample variance; zeros if empty."""
    if not values:
        return 0, 0, 0, 0, 0
    mean = sum(values) / len(values)
    variance = 0.0
    if len(values) > 1:
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return max(values), min(values), mean, math.sqrt(variance), variance


def gaps(times: Sequence[int]) -> list[int]:
    return [b - a for a, b in pairwise(times)]


def active_idle(times: Sequence[int]) -> tuple[list[int], list[int]]:
    """Active and idle periods, split at gaps longer than the activity timeout."""
    active: list[int] = []
    idle: list[int] = []
    start = times[0]
    for previous, current in pairwise(times):
        if current - previous > ACTIVITY_TIMEOUT_US:
            active.append(previous - start)
            idle.append(current - previous)
            start = current
    if idle:
        active.append(times[-1] - start)
    return active, idle


def rate(count: int, duration_us: int) -> float:
    if duration_us == 0:
        return math.nan if count == 0 else math.inf
    return count / (duration_us / 1e6)


def text(value: Number) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity"
    if value == int(value):
        return str(int(value))
    return repr(value)


def row(flow: Flow, start: datetime, seconds: bool) -> list[str]:
    packets = flow.packets
    fwd = [p for p in packets if p.forward]
    bwd = [p for p in packets if not p.forward]
    times = [p.time_us for p in packets]
    duration = times[-1] - times[0]
    fwd_lengths = [p.payload for p in fwd]
    bwd_lengths = [p.payload for p in bwd]
    lengths = [p.payload for p in packets]
    fwd_gaps = gaps([p.time_us for p in fwd])
    bwd_gaps = gaps([p.time_us for p in bwd])
    flow_gaps = gaps(times)
    active, idle = active_idle(times)
    fwd_max, fwd_min, fwd_mean, fwd_std, _ = stats(fwd_lengths)
    bwd_max, bwd_min, bwd_mean, bwd_std, _ = stats(bwd_lengths)
    len_max, len_min, len_mean, len_std, len_var = stats(lengths)
    fiat_max, fiat_min, fiat_mean, fiat_std, _ = stats(flow_gaps)
    fwd_iat_max, fwd_iat_min, fwd_iat_mean, fwd_iat_std, _ = stats(fwd_gaps)
    bwd_iat_max, bwd_iat_min, bwd_iat_mean, bwd_iat_std, _ = stats(bwd_gaps)
    act_max, act_min, act_mean, act_std, _ = stats(active)
    idle_max, idle_min, idle_mean, idle_std, _ = stats(idle)
    fwd_header = sum(p.header for p in fwd)

    def flags(name: str, group: Sequence[Packet] = packets) -> int:
        return sum(1 for p in group if name in p.flags)

    def window(group: Sequence[Packet]) -> int:
        return group[0].window if group and flow.protocol == TCP else -1

    if seconds:
        stamp = (
            f"{start.day:02d}/{start.month:02d}/{start.year} "
            f"{start.hour:02d}:{start.minute:02d}:{start.second:02d}"
        )
    else:
        stamp = (
            f"{start.day}/{start.month}/{start.year} {start.hour}:{start.minute:02d}"
        )

    values: list[Number | str] = [
        f"{flow.src}-{flow.dst}-{flow.sport}-{flow.dport}-{flow.protocol}",
        flow.src,
        flow.sport,
        flow.dst,
        flow.dport,
        flow.protocol,
        stamp,
        duration,
        len(fwd),
        len(bwd),
        sum(fwd_lengths),
        sum(bwd_lengths),
        fwd_max,
        fwd_min,
        fwd_mean,
        fwd_std,
        bwd_max,
        bwd_min,
        bwd_mean,
        bwd_std,
        rate(sum(lengths), duration),
        rate(len(packets), duration),
        fiat_mean,
        fiat_std,
        fiat_max,
        fiat_min,
        sum(fwd_gaps),
        fwd_iat_mean,
        fwd_iat_std,
        fwd_iat_max,
        fwd_iat_min,
        sum(bwd_gaps),
        bwd_iat_mean,
        bwd_iat_std,
        bwd_iat_max,
        bwd_iat_min,
        flags("PSH", fwd),
        flags("PSH", bwd),
        flags("URG", fwd),
        flags("URG", bwd),
        fwd_header,
        sum(p.header for p in bwd),
        rate(len(fwd), duration) if duration else 0,
        rate(len(bwd), duration) if duration else 0,
        len_min,
        len_max,
        len_mean,
        len_std,
        len_var,
        flags("FIN"),
        flags("SYN"),
        flags("RST"),
        flags("PSH"),
        flags("ACK"),
        flags("URG"),
        flags("CWE"),
        flags("ECE"),
        len(bwd) // len(fwd),
        sum(lengths) / len(packets),
        fwd_mean,
        bwd_mean,
        fwd_header,
        *[0] * 6,
        len(fwd),
        sum(fwd_lengths),
        len(bwd),
        sum(bwd_lengths),
        window(fwd),
        window(bwd),
        sum(1 for p in fwd if p.payload > 0),
        min(p.header for p in fwd),
        act_mean,
        act_std,
        act_max,
        act_min,
        idle_mean,
        idle_std,
        idle_max,
        idle_min,
        flow.label,
    ]
    return [v if isinstance(v, str) else text(v) for v in values]


def client_v4(rng: random.Random) -> str:
    return f"192.0.2.{rng.randint(10, 99)}"


def server_v4(rng: random.Random) -> str:
    return f"198.51.100.{rng.randint(1, 50)}"


def ephemeral(rng: random.Random) -> int:
    return rng.randint(32768, 60999)


def web(rng: random.Random, v6: bool = False) -> Flow:
    """A TCP session: handshake, request and response exchanges, close."""
    if v6:
        src, dst = (
            f"2001:db8:0:1::{rng.randint(16, 255):x}",
            f"2001:db8:0:2::{rng.randint(1, 15):x}",
        )
    else:
        src, dst = client_v4(rng), server_v4(rng)
    flow = Flow(src, ephemeral(rng), dst, rng.choice([80, 443, 389]), TCP, "BENIGN")
    rtt = rng.randint(200, 5000)
    flow.send(True, 0, "SYN", header=32, window=rng.choice([8192, 29200, 65535]))
    flow.send(
        False, rtt, "SYN ACK", header=32, window=rng.choice([14600, 28960, 65160])
    )
    flow.send(True, rng.randint(10, 300), "ACK")
    for _ in range(rng.randint(1, 4)):
        think = (
            rng.randint(6_000_000, 9_000_000)
            if rng.random() < 0.1
            else rng.randint(1, 900_000)
        )
        flow.send(True, think, "PSH ACK", payload=rng.randint(80, 600))
        for _ in range(rng.randint(1, 3)):
            flow.send(
                False, rng.randint(50, rtt), "PSH ACK", payload=rng.randint(100, 1460)
            )
        flow.send(True, rng.randint(10, 300), "ACK")
    flow.send(True, rng.randint(1000, 500_000), "FIN ACK")
    flow.send(False, rtt, "FIN ACK")
    flow.send(True, rng.randint(10, 300), "ACK")
    return flow


def dns(rng: random.Random) -> Flow:
    """A UDP query, answered unless the answer was lost."""
    flow = Flow(client_v4(rng), ephemeral(rng), server_v4(rng), 53, UDP, "BENIGN")
    flow.send(True, 0, payload=rng.randint(28, 60), header=8)
    if rng.random() < 0.8:
        flow.send(
            False, rng.randint(500, 50_000), payload=rng.randint(60, 300), header=8
        )
    return flow


def lone_ack(rng: random.Random) -> Flow:
    """One empty TCP segment: zero duration and zero bytes."""
    flow = Flow(client_v4(rng), ephemeral(rng), server_v4(rng), 443, TCP, "BENIGN")
    flow.send(True, 0, "ACK", window=rng.randint(200, 2000))
    return flow


def ddos(rng: random.Random) -> Flow:
    """A short HTTP flood connection against one victim."""
    flow = Flow(
        f"192.0.2.{rng.randint(200, 230)}",
        ephemeral(rng),
        "198.51.100.80",
        80,
        TCP,
        "DDoS",
    )
    flow.send(True, 0, "SYN", header=40, window=8192)
    if rng.random() < 0.85:
        flow.send(False, rng.randint(100, 3000), "SYN ACK", header=32, window=29200)
        flow.send(True, rng.randint(10, 200), "ACK")
        flow.send(True, rng.randint(1, 500), "PSH ACK", payload=rng.randint(6, 20))
        flow.send(False, rng.randint(100, 3000), "ACK")
    flow.send(True, rng.randint(1000, 90_000), "RST ACK")
    return flow


def portscan(rng: random.Random) -> Flow:
    """A SYN probe, refused by a reset or left unanswered."""
    flow = Flow(
        "192.0.2.250",
        rng.randint(40000, 40100),
        "198.51.100.30",
        rng.randint(1, 1024),
        TCP,
        "PortScan",
    )
    flow.send(True, 0, "SYN", header=24, window=1024)
    if rng.random() < 0.9:
        flow.send(False, rng.randint(20, 200), "RST ACK", window=0)
    return flow


MIX: list[tuple[Callable[[random.Random], Flow], int]] = [
    (web, 200),
    (lambda rng: web(rng, v6=True), 20),
    (dns, 60),
    (lone_ack, 20),
    (ddos, 60),
    (portscan, 40),
]


def build() -> bytes:
    rng = random.Random(SEED)
    rows: list[tuple[datetime, list[str]]] = []
    for make, count in MIX:
        for _ in range(count):
            flow = make(rng)
            day = rng.choice([13, 14])
            start = datetime(2017, 7, day, 9) + timedelta(
                microseconds=rng.randint(0, 8 * 3600 * 10**6)
            )
            rows.append((start, row(flow, start, seconds=day == 13)))
    rows.sort(key=lambda item: item[0])
    lines = [HEADER] + [",".join(values) for _, values in rows]
    return "".join(f"{line}\r\n" for line in lines).encode("ascii")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("demo_flows.csv")
    )
    args = parser.parse_args(argv)
    args.out.write_bytes(build())


if __name__ == "__main__":
    main()
