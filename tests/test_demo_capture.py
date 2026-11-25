"""Tests of the demo capture (data/samples/demo_capture.pcap) and its generator."""

import importlib.util
import ipaddress
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from scapy.all import ARP, IP, Ether, IPv6, rdpcap

from netgen.readers import capture
from netgen.record import FlowSettings
from netgen.table import TableResult, stratified_split

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
DEMO_CAPTURE = SAMPLES / "demo_capture.pcap"

DOCUMENTATION_RANGES = [
    ipaddress.ip_network("192.0.2.0/24"),
    ipaddress.ip_network("198.51.100.0/24"),
    ipaddress.ip_network("203.0.113.0/24"),
    ipaddress.ip_network("2001:db8::/32"),
]
DOCUMENTATION_MAC_PREFIX = "00:00:5e:00:53:"  # RFC 7042
BROADCAST_MAC = "ff:ff:ff:ff:ff:ff"

# Hand-worked from the design. A FIN-closed session is SYN, SYN-ACK, ACK, the
# request, the response, the client's FIN, the server's FIN and the last ACK.
# An RST-closed session is SYN, SYN-ACK, ACK, the request and the server's RST.
FIN_SESSION = {"fwd_packets": 5, "bwd_packets": 3}
RST_SESSION = {"fwd_packets": 3, "bwd_packets": 2}


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "make_demo_capture", SAMPLES / "make_demo_capture.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo() -> TableResult:
    return capture.read(DEMO_CAPTURE, FlowSettings())


@pytest.fixture(scope="module")
def flows(demo: TableResult) -> pd.DataFrame:
    return demo.table.to_frame()


def key(row: pd.Series) -> tuple[object, ...]:
    return (row.protocol, row.src_ip, row.src_port, row.dst_ip, row.dst_port)


# The generator


def test_generator_reproduces_committed_capture(
    tmp_path: Path, generator: ModuleType
) -> None:
    out = tmp_path / "demo_capture.pcap"
    generator.main([str(out)])
    assert out.read_bytes() == DEMO_CAPTURE.read_bytes()


def test_generator_writes_only_the_output_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generator: ModuleType
) -> None:
    monkeypatch.chdir(tmp_path)
    generator.main(["out.pcap"])
    assert [path.name for path in tmp_path.iterdir()] == ["out.pcap"]


def test_generator_refuses_a_missing_output_path(
    generator: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        generator.main([])
    assert raised.value.code == 2
    assert "out_path" in capsys.readouterr().err


# The capture


def test_capture_is_under_500_kb() -> None:
    assert DEMO_CAPTURE.stat().st_size < 500_000


def test_capture_forms_the_designed_number_of_flows(
    flows: pd.DataFrame, generator: ModuleType
) -> None:
    assert len(flows) == generator.EXPECTED_FLOWS
    assert len(flows) > 320


def test_capture_mixes_tcp_udp_icmp_and_ipv6(
    flows: pd.DataFrame, generator: ModuleType
) -> None:
    assert {"tcp", "udp", "icmp"} <= set(flows.protocol)
    ipv6 = flows[flows.src_ip.str.contains(":")]
    assert len(ipv6) == generator.SESSIONS_PER_PORT * generator.FIN_PORTS_V6 >= 1


def test_fin_ends_each_session_on_a_reused_client_port(
    flows: pd.DataFrame, generator: ModuleType
) -> None:
    web = flows[
        (flows.protocol == "tcp")
        & (flows.dst_port == generator.WEB_PORT)
        & (flows.src_ip != generator.SCANNER)
    ]
    ports = generator.FIN_PORTS_V4 + generator.FIN_PORTS_V6
    assert len(web) == generator.SESSIONS_PER_PORT * ports
    assert (web.fwd_packets == FIN_SESSION["fwd_packets"]).all()
    assert (web.bwd_packets == FIN_SESSION["bwd_packets"]).all()
    # Without the FIN close the sessions of one port would form one flow.
    counts = web.apply(key, axis=1).value_counts()
    assert len(counts) == ports
    assert (counts == generator.SESSIONS_PER_PORT).all()


def test_rst_ends_each_session_on_a_reused_client_port(
    flows: pd.DataFrame, generator: ModuleType
) -> None:
    reset = flows[
        (flows.protocol == "tcp")
        & (flows.dst_port == generator.RESET_PORT)
        & (flows.src_ip != generator.SCANNER)
    ]
    assert len(reset) == generator.SESSIONS_PER_PORT * generator.RST_PORTS
    assert (reset.fwd_packets == RST_SESSION["fwd_packets"]).all()
    assert (reset.bwd_packets == RST_SESSION["bwd_packets"]).all()
    counts = reset.apply(key, axis=1).value_counts()
    assert len(counts) == generator.RST_PORTS
    assert (counts == generator.SESSIONS_PER_PORT).all()


def test_dns_queries_and_pings_form_one_flow_each(
    flows: pd.DataFrame, generator: ModuleType
) -> None:
    dns = flows[(flows.protocol == "udp") & (flows.dst_port == generator.DNS_PORT)]
    pings = flows[flows.protocol == "icmp"]
    assert len(dns) == generator.DNS_QUERIES
    assert len(pings) == generator.PINGS
    for part in (dns, pings):
        assert (part.fwd_packets == 1).all()
        assert (part.bwd_packets == 1).all()


def test_arp_frames_are_dropped_and_counted(
    demo: TableResult, flows: pd.DataFrame, generator: ModuleType
) -> None:
    assert demo.counts["non_ip_dropped"] == generator.ARP_FRAMES >= 1
    ip_packets = int((flows.fwd_packets + flows.bwd_packets).sum())
    assert demo.counts["packets_read"] == ip_packets + generator.ARP_FRAMES


def test_idle_gap_splits_one_conversation_into_two_flows(
    flows: pd.DataFrame, generator: ModuleType
) -> None:
    idle = flows[flows.src_ip == generator.IDLE_CLIENT].sort_values("start_time")
    assert len(idle) == 2
    first, second = idle.iloc[0], idle.iloc[1]
    assert key(first) == key(second)
    apart = (second.start_time - first.start_time).total_seconds()
    assert apart > 120
    # The second flow starts one gap after the last packet of the first.
    assert apart == pytest.approx(
        generator.IDLE_GAP_US / 1_000_000 + first.duration, abs=1e-5
    )


def test_port_scan_burst_probes_distinct_ports(
    flows: pd.DataFrame, generator: ModuleType
) -> None:
    scan = flows[flows.src_ip == generator.SCANNER]
    assert len(scan) == generator.SCAN_PORTS
    assert scan.dst_port.nunique() == generator.SCAN_PORTS
    assert ((scan.fwd_packets + scan.bwd_packets) <= 2).all()


def test_training_split_holds_at_least_256_flows(demo: TableResult) -> None:
    train, _ = stratified_split(demo.table, seed=42)
    assert len(train.to_frame()) >= 256


def test_every_address_is_a_documentation_address() -> None:
    def documented(text: str) -> bool:
        address = ipaddress.ip_address(text)
        return any(address in network for network in DOCUMENTATION_RANGES)

    def documented_mac(text: str) -> bool:
        return text.startswith(DOCUMENTATION_MAC_PREFIX) or text == BROADCAST_MAC

    for packet in rdpcap(str(DEMO_CAPTURE)):
        assert documented_mac(packet[Ether].src)
        assert documented_mac(packet[Ether].dst)
        for layer, fields in ((IP, ("src", "dst")), (IPv6, ("src", "dst"))):
            if layer in packet:
                assert all(documented(packet[layer].getfieldval(f)) for f in fields)
        if ARP in packet:
            assert documented(packet[ARP].psrc)
            assert documented(packet[ARP].pdst)
