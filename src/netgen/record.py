"""The flow record: every field a reader fills, declared once.

Each field declares its name, type, unit, default role and scaling. Times are
decimal seconds and sizes are payload bytes. A feature field enters the record
only if both a capture and a flow export can fill it exactly; each feature's
definition below was checked against the CIC-IDS2017 excerpt (see
`tests/test_record.py`). Nothing about scaling is detected from data.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from netgen.errors import NetGenError


class Role(StrEnum):
    """What a column is for."""

    IDENTITY = "identity"
    LABEL = "label"
    FEATURE = "feature"
    DERIVED = "derived"


class Scaling(StrEnum):
    """How the scaler treats a feature: `log` takes `log1p` before the Z-score."""

    LOG = "log"
    LINEAR = "linear"


class FieldType(StrEnum):
    TEXT = "text"
    NULLABLE_INTEGER = "nullable integer"
    TIMESTAMP = "naive timestamp"
    INTEGER = "integer"
    DECIMAL = "decimal"
    INDICATOR = "0 or 1"


@dataclass(frozen=True)
class Field:
    """One field of the flow record.

    `scaling` is None for a field that is never scaled (identity, label and
    derived fields). `promoted_scaling` is the scaling of an identity field
    that an option promotes to a feature.
    """

    name: str
    type: FieldType
    unit: str
    role: Role
    scaling: Scaling | None
    definition: str
    promoted_scaling: Scaling | None = None


@dataclass(frozen=True)
class FlowSettings:
    """How packets are grouped into flows. Seconds; `active_timeout` None is off."""

    idle_timeout: float = 120.0
    active_timeout: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("idle_timeout", self.idle_timeout),
            ("active_timeout", self.active_timeout),
        ):
            if value is not None and not value > 0:  # `not >` also refuses NaN
                raise NetGenError(
                    f"{name} is {value} seconds; a flow timeout must be greater than 0."
                )


_T, _NI, _I, _D = (
    FieldType.TEXT,
    FieldType.NULLABLE_INTEGER,
    FieldType.INTEGER,
    FieldType.DECIMAL,
)
_ID, _LAB, _FEAT, _DER = Role.IDENTITY, Role.LABEL, Role.FEATURE, Role.DERIVED
_LOG, _LIN = Scaling.LOG, Scaling.LINEAR

_ZERO_DIRECTION = "A direction with no packets gives 0."
_RATE = "Missing when the duration is 0."


def _length(direction: str, statistic: str) -> Field:
    return Field(
        f"{direction}_len_{statistic}",
        _D,
        "payload bytes",
        _FEAT,
        _LOG,
        f"The {statistic} of the payload length of the {direction} packets"
        + (
            " (sample standard deviation, n - 1 denominator; 0 for one packet)."
            if statistic == "std"
            else "."
        )
        + f" {_ZERO_DIRECTION}",
    )


def _lengths(direction: str) -> tuple[Field, ...]:
    return tuple(
        _length(direction, statistic) for statistic in ("min", "max", "mean", "std")
    )


_GAPS = "The gaps between consecutive packets of the flow, both directions together."

FIELDS: tuple[Field, ...] = (
    Field("src_ip", _T, "none", _ID, None, "The source address of the first packet."),
    Field(
        "dst_ip", _T, "none", _ID, None, "The destination address of the first packet."
    ),
    Field(
        "src_port",
        _NI,
        "none",
        _ID,
        None,
        "The source port of the first packet; missing for ICMP.",
    ),
    Field(
        "dst_port",
        _NI,
        "none",
        _ID,
        None,
        "The destination port of the first packet; missing for ICMP. `--dst-port-as-feature` promotes it.",
        promoted_scaling=_LIN,
    ),
    Field("protocol", _T, "none", _ID, None, "One of `tcp`, `udp`, `icmp`, `other`."),
    Field(
        "start_time",
        FieldType.TIMESTAMP,
        "UTC wall time for captures; local wall time as written for flow exports",
        _ID,
        None,
        "When the first packet of the flow was seen.",
    ),
    Field(
        "label",
        _T,
        "none",
        _LAB,
        None,
        "What the flow is; present only when the input has labels.",
    ),
    Field(
        "duration",
        _D,
        "seconds",
        _FEAT,
        _LOG,
        "Time from the first to the last packet of the flow; 0 for one packet.",
    ),
    Field("fwd_packets", _I, "packets", _FEAT, _LOG, "The number of forward packets."),
    Field(
        "bwd_packets",
        _I,
        "packets",
        _FEAT,
        _LOG,
        f"The number of backward packets. {_ZERO_DIRECTION}",
    ),
    Field(
        "fwd_bytes",
        _I,
        "payload bytes",
        _FEAT,
        _LOG,
        "The sum of the payload lengths of the forward packets.",
    ),
    Field(
        "bwd_bytes",
        _I,
        "payload bytes",
        _FEAT,
        _LOG,
        f"The sum of the payload lengths of the backward packets. {_ZERO_DIRECTION}",
    ),
    *_lengths("fwd"),
    *_lengths("bwd"),
    Field(
        "bytes_per_s",
        _D,
        "payload bytes per second",
        _FEAT,
        _LOG,
        f"(fwd_bytes + bwd_bytes) / duration. {_RATE}",
    ),
    Field(
        "packets_per_s",
        _D,
        "packets per second",
        _FEAT,
        _LOG,
        f"(fwd_packets + bwd_packets) / duration. {_RATE}",
    ),
    Field(
        "fwd_packets_per_s",
        _D,
        "packets per second",
        _FEAT,
        _LOG,
        f"fwd_packets / duration. {_RATE}",
    ),
    Field(
        "bwd_packets_per_s",
        _D,
        "packets per second",
        _FEAT,
        _LOG,
        f"bwd_packets / duration. {_RATE}",
    ),
    Field(
        "iat_mean",
        _D,
        "seconds",
        _FEAT,
        _LOG,
        f"The mean of the gaps: duration / (packets - 1). {_GAPS} 0 for one packet.",
    ),
    Field(
        "iat_std",
        _D,
        "seconds",
        _FEAT,
        _LOG,
        f"The sample standard deviation (n - 1 denominator) of the gaps. {_GAPS} 0 for fewer than two gaps.",
    ),
    Field(
        "iat_max",
        _D,
        "seconds",
        _FEAT,
        _LOG,
        f"The largest gap. {_GAPS} 0 for one packet.",
    ),
    Field(
        "iat_min",
        _D,
        "seconds",
        _FEAT,
        _LOG,
        f"The smallest gap. {_GAPS} 0 for one packet.",
    ),
    Field(
        "fwd_iat_total",
        _D,
        "seconds",
        _FEAT,
        _LOG,
        "Time from the first to the last forward packet (the sum of the gaps between forward packets); 0 for one.",
    ),
    Field(
        "bwd_iat_total",
        _D,
        "seconds",
        _FEAT,
        _LOG,
        f"Time from the first to the last backward packet; 0 for one. {_ZERO_DIRECTION}",
    ),
    Field(
        "fwd_header_bytes",
        _I,
        "bytes",
        _FEAT,
        _LOG,
        "The sum of the transport header lengths of the forward packets: the TCP header with its options"
        " (data offset times 4), the 8-byte UDP header, the 8-byte ICMP header, or 0 for `other`."
        " The IP header is not counted.",
    ),
    Field(
        "bwd_header_bytes",
        _I,
        "bytes",
        _FEAT,
        _LOG,
        f"As fwd_header_bytes, for the backward packets. {_ZERO_DIRECTION}",
    ),
    Field(
        "down_up_ratio",
        _D,
        "none",
        _FEAT,
        _LIN,
        "The integer part of bwd_packets / fwd_packets. Missing when fwd_packets is 0.",
    ),
    Field(
        "is_tcp",
        FieldType.INDICATOR,
        "none",
        _FEAT,
        _LIN,
        "1 when the protocol is `tcp`, else 0.",
    ),
    Field(
        "is_udp",
        FieldType.INDICATOR,
        "none",
        _FEAT,
        _LIN,
        "1 when the protocol is `udp`, else 0.",
    ),
    Field(
        "is_icmp",
        FieldType.INDICATOR,
        "none",
        _FEAT,
        _LIN,
        "1 when the protocol is `icmp`, else 0.",
    ),
    Field(
        "anomaly_score",
        _D,
        "none",
        _DER,
        None,
        "How poorly the autoencoder rebuilds the flow; only when the autoencoder ran.",
    ),
)

FIELD_BY_NAME: Mapping[str, Field] = {field.name: field for field in FIELDS}

ANOMALY_SCORE = "anomaly_score"
_LATENT = re.compile(r"latent_(0|[1-9][0-9]*)")
_ROLE_ORDER = (Role.IDENTITY, Role.LABEL, Role.FEATURE, Role.DERIVED)


def fields_by_role(role: Role) -> tuple[Field, ...]:
    """The fields whose default role is `role`, in table order."""
    return tuple(field for field in FIELDS if field.role is role)


def feature_scaling(name: str) -> Scaling:
    """The declared scaling of a feature, or of a promoted identity field.

    Raises KeyError for a name that is never a feature.
    """
    field = FIELD_BY_NAME.get(name)
    if field is not None:
        scaling = (
            field.scaling if field.role is Role.FEATURE else field.promoted_scaling
        )
        if scaling is not None:
            return scaling
    raise KeyError(f"{name!r} is not a feature of the flow record")


def latent_names(count: int) -> tuple[str, ...]:
    """The names of `count` latent columns: `latent_0` to `latent_<count-1>`."""
    return tuple(f"latent_{i}" for i in range(count))


def is_derived_name(name: str) -> bool:
    """Whether `name` is a derived column: the anomaly score or a latent column."""
    return name == ANOMALY_SCORE or _LATENT.fullmatch(name) is not None


def default_roles(
    columns: Iterable[str], *, dst_port_as_feature: bool = False
) -> dict[str, Role]:
    """The role of each column that the record knows, from the record's defaults and the options.

    A column the record does not know gets no role; FlowTable refuses it.
    """
    roles: dict[str, Role] = {}
    for name in columns:
        if name in FIELD_BY_NAME:
            roles[name] = FIELD_BY_NAME[name].role
        elif is_derived_name(name):
            roles[name] = Role.DERIVED
    if dst_port_as_feature and "dst_port" in roles:
        roles["dst_port"] = Role.FEATURE
    return roles


def _position(name: str) -> tuple[int, int]:
    if name in FIELD_BY_NAME:
        return (FIELDS.index(FIELD_BY_NAME[name]), 0)
    match = _LATENT.fullmatch(name)
    if match is not None:
        return (len(FIELDS), int(match.group(1)))
    return (len(FIELDS) + 1, 0)


def dataset_order(roles: Mapping[str, Role]) -> list[str]:
    """The dataset column order: identity, label, features, derived, each group in table order.

    A column the record does not know follows the record's columns of its group,
    in the order given.
    """
    names = list(roles)
    return sorted(
        names,
        key=lambda name: (
            _ROLE_ORDER.index(roles[name]),
            _position(name),
            names.index(name),
        ),
    )
