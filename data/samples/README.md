# Sample inputs

The demo and the tests share these files. Each maker sits beside its file.

## CIC-IDS2017

`real/cicids2017_excerpt.csv` holds 20 real rows of the CIC-IDS2017 flow
exports (the `TrafficLabelling` CSV files), made by `make_cic_excerpt.py`.

- Dataset page: <https://www.unb.ca/cic/datasets/ids-2017.html>
- Paper: Iman Sharafaldin, Arash Habibi Lashkari and Ali A. Ghorbani, "Toward
  Generating a New Intrusion Detection Dataset and Intrusion Traffic
  Characterization", 4th International Conference on Information Systems
  Security and Privacy (ICISSP), 2018.
- Condition: the Canadian Institute for Cybersecurity requires that any use
  or redistribution of the dataset includes a citation to the dataset and to
  the paper above.

The excerpt is scrubbed. Every address in `Source IP`, `Destination IP` and
`Flow ID` is replaced by a documentation address (`192.0.2.0/24`,
`198.51.100.0/24`, `2001:db8::/32`), with one consistent mapping across the
file. All other values, the header and the cp1252 encoding are kept as
published, including the non-UTF-8 dash in the web-attack labels. Line
endings are CRLF throughout. The maker's docstring states the selection rule.

The excerpt was made once, in the `dev` container, with:

```sh
docker compose run --rm dev python data/samples/make_cic_excerpt.py data/external/TrafficLabelling
```

No test runs the maker; it needs the dataset, which this repository does not
hold.

## Generated samples

- `demo_capture.pcap` (made by `make_demo_capture.py`): synthetic, not real
  traffic. An Ethernet pcap of 400 flows: TCP sessions closed by a two-sided
  FIN (some over IPv6), TCP sessions closed by an RST, DNS queries over UDP,
  pings, four ARP frames (dropped and counted by the capture reader), one
  conversation with a 150 s idle gap that the 120 s idle timeout splits into
  two flows, and a port-scan burst of 40 probes. The flow counts per category
  are constants in the script. Addresses are documentation addresses (IPv4,
  IPv6 and MAC). The generator has a fixed seed and writes identical bytes on
  every run:
  `docker compose run --rm dev python data/samples/make_demo_capture.py data/samples/demo_capture.pcap`.
- `demo_flows.csv` (made by `make_demo_flows.py`): synthetic, not real data.
  About 400 flows in the CIC-IDS2017 layout, with its quirks: the real
  header with its leading spaces and duplicated `Fwd Header Length`,
  microsecond times, day-first timestamps with and without seconds,
  `Infinity` and `NaN` rates, and `-1` initial windows. Labels are `BENIGN`,
  `DDoS` and `PortScan`. Each flow is simulated packet by packet, so its
  columns agree with each other. Addresses are documentation addresses. The
  generator has a fixed seed and writes identical bytes on every run:
  `docker compose run --rm dev python data/samples/make_demo_flows.py --out data/samples/demo_flows.csv`.

## PcapPlusPlus

`real/one_http_stream_fin.pcap`, `real/one_http_stream_rst.pcap` and
`real/pcapng-example.pcapng` come from the PcapPlusPlus test captures, made
by `make_pcpp_captures.py`.

- Project: PcapPlusPlus, <https://github.com/seladb/PcapPlusPlus>, released
  under the Unlicense.
- Folder: `Tests/Pcap++Test/PcapExamples/`, files of the same names.

The captures are scrubbed. The originals hold public addresses, an HTTP
request with a software licence key, and pcapng options with a host name and
a TLS key comment. Every IPv4 address is replaced by a `192.0.2.0/24`
address, every IPv6 address by a `2001:db8::/32` address and every MAC
address by a `00:00:5e:00:53:00/24` address, with one mapping across the
three files. Every payload and padding byte is zero. Timestamps (to the
microsecond), lengths,
ports, TCP flags, sequence numbers and options are kept, and checksums are
recomputed. The pcapng keeps its two interfaces (Ethernet on loopback, Linux
cooked capture), its snapped packets and its timestamps as published; those
of the cooked interface lie near 2^64 microseconds.

The captures were made once, in the `dev` container, from a host download of
the three files into the ignored `out/pcpp/` folder:

```sh
docker compose run --rm dev python data/samples/make_pcpp_captures.py out/pcpp
```

No test runs the maker; it needs the original captures, which this
repository does not hold.
