"""NetGen's learning-stage protocol (ADR 0002).

A stage's `fit(table)` returns a frozen `Fitted` object. The `Fitted` object
has no `fit`, so a second fit cannot append to the first. Stages read only
`table.features()` and change only feature columns.
"""

import json
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd

from netgen.errors import ArtefactsError
from netgen.table import FlowTable, TableResult


@runtime_checkable
class Fitted(Protocol):
    """A fitted stage. Its state never changes."""

    def transform(self, table: FlowTable) -> TableResult: ...

    def save(self, directory: Path) -> None: ...


@runtime_checkable
class Stage(Protocol):
    """A learning stage, as a module: `fit` learns from the training table, `load` reads saved state.

    `manifest` is the manifest mapping of the artefacts folder; a stage reads
    from it only the settings it needs.
    """

    def fit(self, table: FlowTable) -> Fitted: ...

    def load(self, directory: Path, manifest: Mapping[str, Any]) -> Fitted: ...


def as_float(series: pd.Series) -> npt.NDArray[np.float64]:
    """A writable float copy of a feature column, with a missing value as NaN."""
    # A nullable integer column holds pd.NA, which numpy reads as NaN only through `na_value`.
    # The copy is writable: under copy-on-write, `to_numpy` may return a read-only view.
    return np.array(series.to_numpy(dtype=float, na_value=np.nan), dtype=np.float64)


@dataclass(frozen=True)
class JsonType:
    """The Python types a JSON value may have for one key, and how a message names them."""

    types: tuple[type, ...]
    name: str

    def accepts(self, value: object) -> bool:
        """Whether `value` has one of the types. A boolean is never a number."""
        if isinstance(value, bool) and bool not in self.types:
            return False
        return isinstance(value, self.types)

    def or_null(self) -> "JsonType":
        """The same type, or JSON `null`."""
        return JsonType((*self.types, type(None)), f"{self.name} or null")


NUMBER = JsonType((int, float), "a number")
INTEGER = JsonType((int,), "an integer")
FLAG = JsonType((bool,), "true or false")
TEXT = JsonType((str,), "a string")
OBJECT = JsonType((dict,), "a JSON object")


def malformed(file: str, problem: str) -> ArtefactsError:
    """The error for an artefacts file whose content is wrong."""
    return ArtefactsError(
        f"{file} is malformed: {problem}. Run `netgen fit` to write the artefacts again."
    )


def _refuse_constant(name: str) -> None:
    raise ValueError(f"{name} is not a JSON number")


class _OutOfRange(ValueError):
    """A JSON number that no finite float holds."""


def _finite_float(text: str) -> float:
    # float() reads an overflowing literal such as 1e999 as inf and does not raise.
    number = float(text)
    if not math.isfinite(number):
        raise _OutOfRange(f"the number {text} is out of the float range")
    return number


def _float_sized_int(text: str) -> int:
    # The stages read each number with float(), which raises OverflowError past the float range.
    number = int(text)
    if abs(number) > sys.float_info.max:
        raise _OutOfRange(f"the number {text} is out of the float range")
    return number


def read_json(directory: Path, file: str) -> Any:
    """The JSON content of `file` in an artefacts folder.

    Refuses a missing or unreadable file and content that is not UTF-8 JSON.
    `NaN`, `Infinity` and a number out of the float range are refused: the
    stages write finite numbers only.
    """
    path = directory / file
    if not path.is_file():
        raise ArtefactsError(
            f"The artefacts folder {directory} has no {file}. Run `netgen fit` to write the artefacts again."
        )
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ArtefactsError(
            f"NetGen cannot read {file}: {error.strerror}. "
            f"Give read access to the artefacts folder {directory}, or run `netgen fit` to write it again."
        ) from error
    # UnicodeDecodeError is a ValueError; read_text raises it.
    except ValueError as error:
        raise malformed(file, "it is not valid JSON") from error
    try:
        return json.loads(
            content,
            parse_constant=_refuse_constant,
            parse_float=_finite_float,
            parse_int=_float_sized_int,
        )
    except _OutOfRange as error:
        raise malformed(file, str(error)) from error
    # json.JSONDecodeError is a ValueError.
    except ValueError as error:
        raise malformed(file, "it is not valid JSON") from error


def json_object(value: Any, file: str, where: str) -> Mapping[str, Any]:
    """`value` as a JSON object; `where` names it in the message."""
    if not isinstance(value, dict):
        raise malformed(file, f"{where} is not a JSON object")
    return value


def json_field(
    entry: Mapping[str, Any], key: str, kind: JsonType, file: str, where: str
) -> Any:
    """The value of `key` in the JSON object `entry`, refused when it is absent or of another type."""
    if key not in entry:
        raise malformed(file, f"{where} lacks the key {key!r}")
    value = entry[key]
    if not kind.accepts(value):
        raise malformed(file, f"{where}: {key!r} is {value!r}, not {kind.name}")
    return value


def check_columns(columns: Iterable[str], learned: Iterable[str], file: str) -> None:
    """Refuse a table whose feature columns differ from those a stage saved in `file`."""
    present, saved = list(columns), list(learned)
    missing = [name for name in saved if name not in present]
    extra = [name for name in present if name not in saved]
    if missing or extra:
        raise ArtefactsError(
            f"The table's feature columns differ from those in {file}: missing {missing}, extra {extra}. "
            "Transform a table with the features the artefacts were fitted on."
        )
