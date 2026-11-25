"""Write real/cicids2017_excerpt.csv: 20 scrubbed rows of the CIC-IDS2017 flow exports.

Run it once, in the dev container, against the read-only dataset folder:

    docker compose run --rm dev python data/samples/make_cic_excerpt.py \\
        data/external/TrafficLabelling

Selection, reading the CSV files in name order and taking each row once:

1. the first row of each attack label;
2. the first row that holds `Infinity`, then the first that holds `NaN`;
3. the first `BENIGN` row of each file, until 20 rows are chosen.

The rows keep the real header and values and are written in file and line
order. Addresses are replaced by documentation addresses in order of first
appearance: the testbed's 192.168.0.0/16 by 192.0.2.0/24, other IPv4 by
198.51.100.0/24, IPv6 by 2001:db8::/32. The addresses inside `Flow ID` take the
same mapping. The output is cp1252, the source encoding, with CRLF line endings
(two source files end their lines with LF only).
"""

import argparse
import ipaddress
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

ENCODING = "cp1252"
ROW_COUNT = 20
FLOW_ID, SOURCE_IP, DESTINATION_IP = 0, 1, 3
TESTBED = ipaddress.ip_network("192.168.0.0/16")
TESTBED_RANGE = ipaddress.ip_network("192.0.2.0/24")
OTHER_V4_RANGE = ipaddress.ip_network("198.51.100.0/24")
V6_RANGE = ipaddress.ip_network("2001:db8::/32")


@dataclass(frozen=True)
class Line:
    file: int
    number: int
    fields: list[str]

    @property
    def key(self) -> tuple[int, int]:
        return self.file, self.number


def read_lines(path: Path, file: int) -> Iterator[Line]:
    with path.open(encoding=ENCODING, newline="") as handle:
        for number, text in enumerate(handle):
            yield Line(file, number, text.rstrip("\r\n").split(","))


def select(files: Sequence[Path]) -> tuple[str, list[Line]]:
    """The common header and the chosen rows, in file and line order."""
    header = ""
    attacks: dict[str, Line] = {}
    with_value: dict[str, Line] = {}
    benign: list[Line] = []
    for index, path in enumerate(files):
        lines = read_lines(path, index)
        file_header = ",".join(next(lines).fields)
        if header and file_header != header:
            raise SystemExit(f"{path.name}: header differs from {files[0].name}")
        header = file_header
        first_benign = True
        for line in lines:
            label = line.fields[-1]
            if not label:
                continue
            if label == "BENIGN" and first_benign:
                benign.append(line)
                first_benign = False
            elif label != "BENIGN":
                attacks.setdefault(label, line)
            for value in ("Infinity", "NaN"):
                if value in line.fields:
                    with_value.setdefault(value, line)
    chosen: dict[tuple[int, int], Line] = {}
    candidates = [*attacks.values(), with_value["Infinity"], with_value["NaN"], *benign]
    for line in candidates:
        if len(chosen) == ROW_COUNT:
            break
        chosen.setdefault(line.key, line)
    if len(chosen) < ROW_COUNT:
        raise SystemExit(f"only {len(chosen)} rows match the selection rule")
    return header, [chosen[key] for key in sorted(chosen)]


class Scrubber:
    """Maps each real address to a documentation address, the same one every time."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}
        self.used: dict[ipaddress.IPv4Network | ipaddress.IPv6Network, int] = {}

    def __call__(self, address: str) -> str:
        if address not in self.mapping:
            real = ipaddress.ip_address(address)
            if real.version == 6:
                pool = V6_RANGE
            elif real in TESTBED:
                pool = TESTBED_RANGE
            else:
                pool = OTHER_V4_RANGE
            self.used[pool] = self.used.get(pool, 0) + 1
            self.mapping[address] = str(pool[self.used[pool]])
        return self.mapping[address]

    def flow_id(self, value: str) -> str:
        pair, source_port, destination_port, protocol = value.rsplit("-", 3)
        first, second = pair.split("-")
        return "-".join(
            [self(first), self(second), source_port, destination_port, protocol]
        )


def scrub(lines: Sequence[Line]) -> list[str]:
    scrubber = Scrubber()
    rows = []
    for line in lines:
        fields = list(line.fields)
        fields[SOURCE_IP] = scrubber(fields[SOURCE_IP])
        fields[DESTINATION_IP] = scrubber(fields[DESTINATION_IP])
        fields[FLOW_ID] = scrubber.flow_id(fields[FLOW_ID])
        rows.append(",".join(fields))
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write the scrubbed CIC-IDS2017 excerpt."
    )
    parser.add_argument(
        "dataset", type=Path, help="folder of the CIC-IDS2017 CSV files"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "real" / "cicids2017_excerpt.csv",
    )
    args = parser.parse_args(argv)
    files = sorted(args.dataset.glob("*.csv"))
    if not files:
        raise SystemExit(f"{args.dataset}: no CSV files")
    header, lines = select(files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding=ENCODING, newline="") as handle:
        handle.writelines(f"{text}\r\n" for text in [header, *scrub(lines)])


if __name__ == "__main__":
    main()
