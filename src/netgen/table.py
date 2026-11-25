"""The FlowTable: a frozen pair of a DataFrame and its column roles, and the seam check.

The rules "never scale or learn from identity, label or derived columns" live
here: learning stages read only `features()`, and derived columns enter only
through `with_derived`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Self

import numpy as np
import pandas as pd

from netgen import record
from netgen.errors import InputError, NetGenError
from netgen.record import FieldType, Role


class FlowTable:
    """A DataFrame plus the role of each of its columns. Never changed after construction."""

    __slots__ = ("_frame", "_roles")
    _frame: pd.DataFrame
    _roles: Mapping[str, Role]

    def __init__(self, frame: pd.DataFrame, roles: Mapping[str, Role]) -> None:
        derived = [name for name, role in roles.items() if role is Role.DERIVED]
        if derived:
            raise NetGenError(
                f"Derived columns {derived} were given to FlowTable directly. "
                "Add a derived column with FlowTable.with_derived."
            )
        self._set(frame.copy(), roles)

    @classmethod
    def from_frame(
        cls, frame: pd.DataFrame, *, dst_port_as_feature: bool = False
    ) -> Self:
        """A table whose roles come from the record's defaults and the options."""
        return cls(
            frame,
            record.default_roles(
                frame.columns, dst_port_as_feature=dst_port_as_feature
            ),
        )

    def _set(self, frame: pd.DataFrame, roles: Mapping[str, Role]) -> None:
        _check_roles(frame, roles)
        object.__setattr__(self, "_frame", frame)
        object.__setattr__(self, "_roles", MappingProxyType(dict(roles)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"FlowTable is frozen: cannot set {name!r}")

    @property
    def roles(self) -> Mapping[str, Role]:
        """The role of each column, read-only."""
        return self._roles

    def features(self) -> pd.DataFrame:
        """A copy of the feature columns only, in dataset column order."""
        return self._frame[self._names(Role.FEATURE)].copy()

    def to_frame(self) -> pd.DataFrame:
        """A copy of the whole table in dataset column order."""
        return self._frame[record.dataset_order(self._roles)].copy()

    def with_derived(self, name: str, values: Any) -> "FlowTable":
        """A new table with one more column, of the derived role. This table is unchanged."""
        if name in self._roles:
            raise NetGenError(
                f"The derived column {name!r} already exists in the table. Choose another name."
            )
        array = np.asarray(values)
        if array.shape != (len(self._frame),):
            raise NetGenError(
                f"The derived column {name!r} has {array.size} values; the table has {len(self._frame)} rows. "
                "Give one value per row."
            )
        frame = self._frame.copy()
        frame[name] = array
        table = object.__new__(FlowTable)
        table._set(frame, {**self._roles, name: Role.DERIVED})
        return table

    def with_features(self, values: pd.DataFrame) -> "FlowTable":
        """A new table whose feature columns hold `values`. The other columns and this table are unchanged.

        `values` has exactly the feature columns, one row per table row, in row order.
        """
        names = self._names(Role.FEATURE)
        if sorted(str(column) for column in values.columns) != sorted(names):
            raise NetGenError(
                f"The new feature values have columns {list(values.columns)}; the table's features are {names}. "
                "Give exactly the feature columns."
            )
        if len(values) != len(self._frame):
            raise NetGenError(
                f"The new feature values have {len(values)} rows; the table has {len(self._frame)} rows. "
                "Give one row per table row."
            )
        frame = self._frame.copy()
        for name in names:
            frame[name] = values[name].to_numpy()
        table = object.__new__(FlowTable)
        table._set(frame, self._roles)
        return table

    def _names(self, role: Role) -> list[str]:
        return [
            name
            for name in record.dataset_order(self._roles)
            if self._roles[name] is role
        ]


type Count = int | str | Mapping[str, int] | tuple[str, ...]
"""One entry of the report's counts: a number, a name, a number per column, or a list of names."""


@dataclass(frozen=True)
class TableResult:
    """What a reader or a stage returns: the table, its counts and its warnings, for the report."""

    table: FlowTable
    counts: Mapping[str, Count]
    warnings: tuple[str, ...]


def _check_roles(frame: pd.DataFrame, roles: Mapping[str, Role]) -> None:
    columns = [str(column) for column in frame.columns]
    unassigned = [name for name in columns if name not in roles]
    if unassigned:
        raise NetGenError(
            f"Columns {unassigned} have no role. Give each column a role, or drop the columns the flow record lacks."
        )
    absent = [name for name in roles if name not in columns]
    if absent:
        raise NetGenError(
            f"Roles were given for missing columns {absent}. Remove those roles or add the columns."
        )
    labels = [name for name, role in roles.items() if role is Role.LABEL]
    if len(labels) > 1:
        raise NetGenError(
            f"More than one label column: {labels}. Keep exactly one column with the label role."
        )
    text = [
        name
        for name, role in roles.items()
        if role is Role.FEATURE and not _is_number(frame[name])
    ]
    if text:
        raise NetGenError(
            f"Feature columns {text} are not numeric. A feature must hold numbers."
        )


def _is_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _is_naive_timestamp(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_dtype(series) and not isinstance(
        series.dtype, pd.DatetimeTZDtype
    )


_TYPE_CHECKS: Mapping[FieldType, Callable[[pd.Series], bool]] = {
    FieldType.TEXT: lambda series: (
        pd.api.types.is_string_dtype(series) and not _is_number(series)
    ),
    FieldType.NULLABLE_INTEGER: pd.api.types.is_integer_dtype,
    FieldType.TIMESTAMP: _is_naive_timestamp,
    FieldType.INTEGER: _is_number,
    FieldType.DECIMAL: _is_number,
    FieldType.INDICATOR: _is_number,
}


def check_record(table: FlowTable) -> None:
    """Check a table against the flow record: required fields, types and roles.

    Raises InputError naming the first problem. Readers call it on their output;
    `Artefacts.transform` calls it on its input.
    """
    roles = table.roles
    required = [
        field.name
        for field in record.FIELDS
        if field.role in (Role.IDENTITY, Role.FEATURE)
    ]
    missing = [name for name in required if name not in roles]
    if missing:
        raise InputError(
            f"The table lacks the flow record fields {missing}. The reader must fill every field."
        )
    frame = table.to_frame()
    for name, role in roles.items():
        field = record.FIELD_BY_NAME.get(name)
        if field is None:
            if role is Role.DERIVED and record.is_derived_name(name):
                continue
            raise InputError(
                f"The column {name!r} is not in the flow record. The reader must drop it."
            )
        allowed = {field.role} | (
            {Role.FEATURE} if field.promoted_scaling is not None else set()
        )
        if role not in allowed:
            raise InputError(
                f"The column {name!r} has the role {role}; the flow record allows {sorted(allowed)}."
            )
        if not _TYPE_CHECKS[field.type](frame[name]):
            raise InputError(
                f"The column {name!r} has dtype {frame[name].dtype}; the flow record wants {field.type}."
            )


def stratified_split(table: FlowTable, *, seed: int) -> tuple[FlowTable, FlowTable]:
    """The Training split and the Test split of `table`: 80/20 per label value, with `seed`.

    Rows without a label form one group of their own. A group of `n` rows sends
    `(n + 2) // 5` rows to the test split (20%, rounded half up), at least 1 when
    `n` is 2 or more. Each split keeps the input row order, row index and roles.
    """
    frame = table._frame
    if frame.empty:
        raise InputError(
            "There are no flows to split: the table has no rows. "
            "Give an input that forms at least one flow."
        )
    labels = [name for name, role in table.roles.items() if role is Role.LABEL]
    if labels:
        codes, uniques = pd.factorize(frame[labels[0]], sort=True)
        order = [*range(len(uniques)), -1]
    else:
        codes, order = np.full(len(frame), -1), [-1]
    generator = np.random.default_rng(seed)
    test = np.zeros(len(frame), dtype=bool)
    for code in order:
        positions = generator.permutation(np.flatnonzero(codes == code))
        count = (len(positions) + 2) // 5
        if len(positions) >= 2:
            count = max(count, 1)
        test[positions[:count]] = True
    return _take(table, ~test), _take(table, test)


def _take(table: FlowTable, rows: np.ndarray) -> FlowTable:
    part = object.__new__(FlowTable)
    part._set(table._frame[rows].copy(), table.roles)
    return part
