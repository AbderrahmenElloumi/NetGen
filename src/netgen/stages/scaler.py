"""The scaler: the learning stage after the fill step.

One method for every feature: `log1p` on the fields the flow record declares
`log`, then a Z-score with the training mean and the population standard
deviation (`ddof=0`). A feature whose training standard deviation is below
`CONSTANT_STD` is constant and written as 0. No value is clipped; values
outside the training range are counted.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from netgen import record
from netgen.record import Role, Scaling
from netgen.stages import (
    FLAG,
    NUMBER,
    as_float,
    check_columns,
    json_field,
    json_object,
    read_json,
)
from netgen.table import FlowTable, TableResult

FILE = "scaler.json"
CONSTANT_STD = 1e-12


@dataclass(frozen=True)
class ScaleValues:
    """What the scaler learned for one feature. Every value is taken after the log step."""

    log: bool
    mean: float
    std: float
    minimum: float
    maximum: float
    constant: bool

    def logged(self, column: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The column after the log step, if this feature has one."""
        return np.log1p(column) if self.log else column


@dataclass(frozen=True)
class FittedScaler:
    """A fitted scaler. `warnings` are known only at fit: `scaler.json` does not hold them."""

    values: Mapping[str, ScaleValues]
    warnings: tuple[str, ...] = ()

    def transform(self, table: FlowTable) -> TableResult:
        """The table with each feature scaled, out-of-range and constant-mismatch values counted."""
        features = table.features()
        check_columns(features.columns, self.values, FILE)
        out_of_range: dict[str, int] = {}
        constant_mismatch: dict[str, int] = {}
        for name, learned in self.values.items():
            column = learned.logged(as_float(features[name]))
            out_of_range[name] = int(
                ((column < learned.minimum) | (column > learned.maximum)).sum()
            )
            if learned.constant:
                # The floating mean of equal values may differ from them by a few ulps.
                differs = ~np.isclose(
                    column, learned.mean, rtol=1e-9, atol=CONSTANT_STD
                )
                constant_mismatch[name] = int(differs.sum())
                features[name] = np.zeros(len(column))
            else:
                features[name] = (column - learned.mean) / learned.std
        constant = tuple(
            name for name, learned in self.values.items() if learned.constant
        )
        return TableResult(
            table.with_features(features),
            {
                "out_of_range": out_of_range,
                "constant_features": constant,
                "constant_mismatch": constant_mismatch,
            },
            self.warnings,
        )

    def save(self, directory: Path) -> None:
        """Write `scaler.json`: feature name to its log flag and its statistics after the log step."""
        state = {
            name: {
                "log": learned.log,
                "mean": learned.mean,
                "std": learned.std,
                "min": learned.minimum,
                "max": learned.maximum,
                "constant": learned.constant,
            }
            for name, learned in self.values.items()
        }
        (directory / FILE).write_text(
            json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def fit(table: FlowTable) -> FittedScaler:
    """Learn, per feature of the training table, its log flag and its statistics after the log step.

    The log flag is the flow record's declared scaling; nothing is detected from data.
    """
    values: dict[str, ScaleValues] = {}
    warnings: list[str] = []
    features = table.features()
    for name in features.columns:
        log = record.feature_scaling(name) is Scaling.LOG
        column = np.log1p(as_float(features[name])) if log else as_float(features[name])
        std = float(column.std())
        values[name] = ScaleValues(
            log=log,
            mean=float(column.mean()),
            std=std,
            minimum=float(column.min()),
            maximum=float(column.max()),
            constant=std < CONSTANT_STD,
        )
        if record.FIELD_BY_NAME[name].role is Role.IDENTITY:
            warnings.append(
                f"The identity field {name!r} is a feature and is scaled linearly. "
                "This is crude: port numbers are categories, not quantities."
            )
    return FittedScaler(MappingProxyType(values), tuple(warnings))


def load(directory: Path, manifest: Mapping[str, Any]) -> FittedScaler:
    """Read `scaler.json` from an artefacts folder. The scaler reads nothing from the manifest."""
    state = json_object(read_json(directory, FILE), FILE, "the file")
    values: dict[str, ScaleValues] = {}
    for name, entry in state.items():
        where = f"the entry {name!r}"
        entry = json_object(entry, FILE, where)
        values[name] = ScaleValues(
            log=json_field(entry, "log", FLAG, FILE, where),
            mean=float(json_field(entry, "mean", NUMBER, FILE, where)),
            std=float(json_field(entry, "std", NUMBER, FILE, where)),
            minimum=float(json_field(entry, "min", NUMBER, FILE, where)),
            maximum=float(json_field(entry, "max", NUMBER, FILE, where)),
            constant=json_field(entry, "constant", FLAG, FILE, where),
        )
    return FittedScaler(MappingProxyType(values))
