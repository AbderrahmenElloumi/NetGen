"""Tests of the flow record in netgen.record.

The excerpt tests apply the exactness rule: every feature definition in the
record is checked against what the 20-row CIC-IDS2017 excerpt lets us
recompute from its own raw columns. CICFlowMeter writes times in microseconds.
"""

import math
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from netgen import record
from netgen.record import Role, Scaling

EXCERPT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "samples"
    / "real"
    / "cicids2017_excerpt.csv"
)

# CICFlowMeter writes values of 10^7 and more with three significant figures
# (for example 1.13E+08), so a check that reads such a column allows 0.5 %.
ROUNDED = 5e-3
EXACT = 1e-6


@pytest.fixture(scope="module")
def excerpt() -> pd.DataFrame:
    frame = pd.read_csv(EXCERPT, encoding="cp1252")
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def features() -> tuple[record.Field, ...]:
    return record.fields_by_role(Role.FEATURE)


# Scenario tests on the declaration


def test_every_feature_declares_log_or_linear_scaling() -> None:
    assert features()
    for field in features():
        assert field.scaling in (Scaling.LOG, Scaling.LINEAR), field.name


def test_no_identity_label_or_derived_field_has_a_scaling() -> None:
    for role in (Role.IDENTITY, Role.LABEL, Role.DERIVED):
        for field in record.fields_by_role(role):
            assert field.scaling is None, field.name


def test_field_names_are_snake_case_and_unique() -> None:
    names = [field.name for field in record.FIELDS]
    assert len(names) == len(set(names))
    for name in names:
        assert re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", name), name


def test_every_feature_states_its_definition_and_unit() -> None:
    for field in features():
        assert field.definition, field.name
        assert field.unit, field.name


def test_record_has_six_identity_fields_and_one_label() -> None:
    identity = [field.name for field in record.fields_by_role(Role.IDENTITY)]
    assert identity == [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "start_time",
    ]
    assert [field.name for field in record.fields_by_role(Role.LABEL)] == ["label"]


def test_heavy_tailed_fields_are_log() -> None:
    heavy = [
        field
        for field in features()
        if field.name.startswith(
            ("fwd_", "bwd_", "iat_", "duration", "bytes_", "packets_")
        )
    ]
    assert heavy
    for field in heavy:
        assert field.scaling is Scaling.LOG, field.name


def test_only_the_ratio_and_protocol_indicators_are_linear() -> None:
    linear = {
        field.name
        for field in features()
        if record.feature_scaling(field.name) is Scaling.LINEAR
    }
    assert linear == {"down_up_ratio", "is_tcp", "is_udp", "is_icmp"}


def test_promoted_dst_port_is_linear() -> None:
    assert record.feature_scaling("dst_port") is Scaling.LINEAR


@pytest.mark.parametrize(
    "name", ["src_port", "label", "anomaly_score", "latent_0", "no_such_field"]
)
def test_scaling_of_a_non_feature_is_refused(name: str) -> None:
    with pytest.raises(KeyError):
        record.feature_scaling(name)


def test_derived_names() -> None:
    assert record.latent_names(3) == ("latent_0", "latent_1", "latent_2")
    assert record.is_derived_name("anomaly_score")
    assert record.is_derived_name("latent_12")
    assert not record.is_derived_name("latent_")
    assert not record.is_derived_name("duration")


def test_dataset_column_order_is_identity_label_features_derived() -> None:
    roles = record.default_roles(
        [field.name for field in record.FIELDS] + ["latent_1", "latent_0"]
    )
    shuffled = dict(reversed(list(roles.items())))
    order = record.dataset_order(shuffled)
    identity = [field.name for field in record.fields_by_role(Role.IDENTITY)]
    feature = [field.name for field in features()]
    assert order == [
        *identity,
        "label",
        *feature,
        "anomaly_score",
        "latent_0",
        "latent_1",
    ]


def test_promoted_dst_port_is_ordered_first_among_features() -> None:
    names = [field.name for field in record.FIELDS if field.role is not Role.DERIVED]
    roles = record.default_roles(names, dst_port_as_feature=True)
    order = record.dataset_order(roles)
    assert roles["dst_port"] is Role.FEATURE
    assert order.index("dst_port") == order.index("label") + 1


def test_default_roles_come_from_the_record() -> None:
    roles = record.default_roles(
        [
            "src_ip",
            "dst_port",
            "label",
            "duration",
            "anomaly_score",
            "latent_4",
            "extra",
        ]
    )
    assert roles == {
        "src_ip": Role.IDENTITY,
        "dst_port": Role.IDENTITY,
        "label": Role.LABEL,
        "duration": Role.FEATURE,
        "anomaly_score": Role.DERIVED,
        "latent_4": Role.DERIVED,
    }


def test_flow_settings_defaults() -> None:
    settings = record.FlowSettings()
    assert settings.idle_timeout == 120.0
    assert settings.active_timeout is None


# The exactness rule: each feature definition against the excerpt


def close(actual: float, expected: float, rel: float = EXACT) -> bool:
    return math.isclose(actual, expected, rel_tol=rel, abs_tol=1e-6)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def check_packets(direction: str) -> Callable[[pd.Series], None]:
    column = "Total Fwd Packets" if direction == "Fwd" else "Total Backward Packets"

    def check(row: pd.Series) -> None:
        assert float(row[column]).is_integer() and row[column] >= 0

    return check


def check_bytes(direction: str) -> Callable[[pd.Series], None]:
    packets = "Total Fwd Packets" if direction == "Fwd" else "Total Backward Packets"

    def check(row: pd.Series) -> None:
        total = row[f"Total Length of {direction} Packets"]
        assert close(total, row[f"{direction} Packet Length Mean"] * row[packets])

    return check


def check_length(direction: str, stat: str) -> Callable[[pd.Series], None]:
    packets = "Total Fwd Packets" if direction == "Fwd" else "Total Backward Packets"

    def check(row: pd.Series) -> None:
        n = int(row[packets])
        low, high = (
            row[f"{direction} Packet Length Min"],
            row[f"{direction} Packet Length Max"],
        )
        mean, std = (
            row[f"{direction} Packet Length Mean"],
            row[f"{direction} Packet Length Std"],
        )
        value = row[f"{direction} Packet Length {stat}"]
        if n == 0:
            assert value == 0
            return
        assert low <= mean <= high
        if n == 1:
            assert low == mean == high and std == 0
        if stat == "Std" and 2 <= n <= 3:
            lengths = (
                [low, high]
                if n == 2
                else [
                    low,
                    high,
                    row[f"Total Length of {direction} Packets"] - low - high,
                ]
            )
            assert close(std, sample_std(lengths[:n]))

    return check


def duration_seconds(row: pd.Series) -> float:
    return float(row["Flow Duration"]) / 1e6


def all_packets(row: pd.Series) -> int:
    return int(row["Total Fwd Packets"] + row["Total Backward Packets"])


def check_duration(row: pd.Series) -> None:
    assert close(row["Flow Duration"], row["Flow IAT Mean"] * (all_packets(row) - 1))


def check_rate(
    numerator: Callable[[pd.Series], float], column: str
) -> Callable[[pd.Series], None]:
    def check(row: pd.Series) -> None:
        if row["Flow Duration"] == 0:
            return
        assert close(row[column], numerator(row) / duration_seconds(row))

    return check


def check_flow_iat(row: pd.Series) -> None:
    n = all_packets(row)
    low, high, mean, std = (
        row[f"Flow IAT {stat}"] for stat in ("Min", "Max", "Mean", "Std")
    )
    duration = row["Flow Duration"]
    assert (
        low <= mean * (1 + EXACT)
        and mean <= high * (1 + ROUNDED)
        and high <= duration * (1 + ROUNDED)
    )
    if n == 2:
        assert low == high == mean == duration and std == 0
    if n == 4:
        assert close(std, sample_std([low, high, duration - low - high]), ROUNDED)


def check_iat_total(direction: str) -> Callable[[pd.Series], None]:
    packets = "Total Fwd Packets" if direction == "Fwd" else "Total Backward Packets"

    def check(row: pd.Series) -> None:
        total = row[f"{direction} IAT Total"]
        gaps = max(int(row[packets]) - 1, 0)
        assert close(total, row[f"{direction} IAT Mean"] * gaps, ROUNDED)
        assert total <= row["Flow Duration"] * (1 + ROUNDED)

    return check


def check_header(direction: str) -> Callable[[pd.Series], None]:
    packets = "Total Fwd Packets" if direction == "Fwd" else "Total Backward Packets"

    def check(row: pd.Series) -> None:
        n, header = int(row[packets]), int(row[f"{direction} Header Length"])
        # TCP header with options, IP header excluded: 20 to 60 bytes, a multiple of 4.
        assert header % 4 == 0
        assert 20 * n <= header <= 60 * n
        if direction == "Fwd":
            assert header == row["Fwd Header Length.1"]
            assert header >= row["min_seg_size_forward"] * n

    return check


def check_ratio(row: pd.Series) -> None:
    assert (
        row["Down/Up Ratio"]
        == row["Total Backward Packets"] // row["Total Fwd Packets"]
    )


def check_protocol(row: pd.Series) -> None:
    # The indicators derive from the IANA protocol number (1 ICMP, 6 TCP, 17 UDP).
    assert row["Protocol"] in (0, 1, 6, 17)


EXCERPT_CHECKS: dict[str, Callable[[pd.Series], None]] = {
    "duration": check_duration,
    "fwd_packets": check_packets("Fwd"),
    "bwd_packets": check_packets("Bwd"),
    "fwd_bytes": check_bytes("Fwd"),
    "bwd_bytes": check_bytes("Bwd"),
    **{
        f"{d.lower()}_len_{s.lower()}": check_length(d, s)
        for d in ("Fwd", "Bwd")
        for s in ("Min", "Max", "Mean", "Std")
    },
    "bytes_per_s": check_rate(
        lambda row: (
            row["Total Length of Fwd Packets"] + row["Total Length of Bwd Packets"]
        ),
        "Flow Bytes/s",
    ),
    "packets_per_s": check_rate(lambda row: all_packets(row), "Flow Packets/s"),
    "fwd_packets_per_s": check_rate(
        lambda row: row["Total Fwd Packets"], "Fwd Packets/s"
    ),
    "bwd_packets_per_s": check_rate(
        lambda row: row["Total Backward Packets"], "Bwd Packets/s"
    ),
    **{f"iat_{stat}": check_flow_iat for stat in ("mean", "std", "max", "min")},
    "fwd_iat_total": check_iat_total("Fwd"),
    "bwd_iat_total": check_iat_total("Bwd"),
    "fwd_header_bytes": check_header("Fwd"),
    "bwd_header_bytes": check_header("Bwd"),
    "down_up_ratio": check_ratio,
    "is_tcp": check_protocol,
    "is_udp": check_protocol,
    "is_icmp": check_protocol,
}


def test_every_feature_has_an_excerpt_check() -> None:
    assert set(EXCERPT_CHECKS) == {field.name for field in features()}


@pytest.mark.parametrize("name", sorted(EXCERPT_CHECKS))
def test_feature_definition_holds_on_the_excerpt(
    excerpt: pd.DataFrame, name: str
) -> None:
    assert len(excerpt) == 20
    for index, row in excerpt.iterrows():
        try:
            EXCERPT_CHECKS[name](row)
        except AssertionError as error:
            raise AssertionError(f"{name}: excerpt row {index}") from error


def test_flag_columns_of_the_excerpt_are_not_per_flow_counts(
    excerpt: pd.DataFrame,
) -> None:
    # Why the record has no TCP flag counts: a long TCP flow holds many ACK
    # packets, yet the excerpt writes at most 1 in every flag-count column.
    flags = ["FIN", "SYN", "RST", "PSH", "ACK", "URG"]
    counts = excerpt[[f"{flag} Flag Count" for flag in flags]]
    assert counts.isin([0, 1]).all().all()
    longest = excerpt.loc[excerpt["Total Fwd Packets"].idxmax()]
    assert longest["Total Fwd Packets"] > 1000 and longest["ACK Flag Count"] == 0
    for flag in flags:
        assert record.FIELD_BY_NAME.get(f"{flag.lower()}_count") is None
