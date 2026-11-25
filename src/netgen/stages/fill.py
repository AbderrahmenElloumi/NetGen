"""The fill step: the first learning stage.

It learns per feature, from the training table, the largest finite value and
the median of the finite values. `inf` and `-inf` become the largest finite
value; a missing value becomes the median. A feature with no finite training
value is filled with 0 and named in a warning.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from netgen.stages import (
    NUMBER,
    as_float,
    check_columns,
    json_field,
    json_object,
    read_json,
)
from netgen.table import FlowTable, TableResult

FILE = "fill.json"


@dataclass(frozen=True)
class FillValues:
    """The two values learned for one feature."""

    maximum: float
    median: float


@dataclass(frozen=True)
class FittedFill:
    """A fitted fill step. `no_finite_value` is known only at fit: `fill.json` does not hold it."""

    values: Mapping[str, FillValues]
    no_finite_value: tuple[str, ...] = ()

    def transform(self, table: FlowTable) -> TableResult:
        """The table with infinite and missing feature values replaced, and the replacements counted."""
        features = table.features()
        check_columns(features.columns, self.values, FILE)
        filled: dict[str, int] = {}
        for name, learned in self.values.items():
            column = as_float(features[name])
            infinite = np.isinf(column)
            missing = np.isnan(column)
            column[infinite] = learned.maximum
            column[missing] = learned.median
            features[name] = column
            filled[name] = int(infinite.sum() + missing.sum())
        warnings = tuple(
            f"The feature {name!r} has no finite value in the training table. "
            "It is filled with 0 and is constant."
            for name in self.no_finite_value
        )
        return TableResult(
            table.with_features(features),
            {"filled": filled, "no_finite_value": self.no_finite_value},
            warnings,
        )

    def save(self, directory: Path) -> None:
        """Write `fill.json`: feature name to `{"max": x, "median": y}`."""
        state = {
            name: {"max": learned.maximum, "median": learned.median}
            for name, learned in self.values.items()
        }
        (directory / FILE).write_text(
            json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def fit(table: FlowTable) -> FittedFill:
    """Learn the largest finite value and the median of each feature of the training table."""
    values: dict[str, FillValues] = {}
    no_finite_value: list[str] = []
    features = table.features()
    for name in features.columns:
        column = as_float(features[name])
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            values[name] = FillValues(0.0, 0.0)
            no_finite_value.append(name)
        else:
            values[name] = FillValues(float(finite.max()), float(np.median(finite)))
    return FittedFill(MappingProxyType(values), tuple(no_finite_value))


def load(directory: Path, manifest: Mapping[str, Any]) -> FittedFill:
    """Read `fill.json` from an artefacts folder. The fill step reads nothing from the manifest."""
    state = json_object(read_json(directory, FILE), FILE, "the file")
    values: dict[str, FillValues] = {}
    for name, entry in state.items():
        where = f"the entry {name!r}"
        entry = json_object(entry, FILE, where)
        values[name] = FillValues(
            float(json_field(entry, "max", NUMBER, FILE, where)),
            float(json_field(entry, "median", NUMBER, FILE, where)),
        )
    return FittedFill(MappingProxyType(values))
