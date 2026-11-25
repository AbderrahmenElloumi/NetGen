"""The readers: each turns one input format into flows in the flow record.

`READERS` is the explicit reader list. A new reader is one new module in this
package plus one line in the list.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from netgen.errors import InputError
from netgen.readers import capture, flow_export
from netgen.record import FlowSettings
from netgen.table import TableResult


@runtime_checkable
class Reader(Protocol):
    """One input format. A reader learns nothing.

    `read` returns the table with its counts and warnings, checked with
    `netgen.table.check_record`. `label_col` names the label column of a flow
    export; a capture ignores it.
    """

    name: str

    def sniff(self, path: Path) -> bool: ...

    def read(
        self, path: Path, flow_settings: FlowSettings, label_col: str = "Label"
    ) -> TableResult: ...


# Detection asks each reader in this order. A capture is told apart by its
# magic bytes and a flow export by its CSV header, so the order does not
# change the result for the two built-in readers.
READERS: tuple[Reader, ...] = (capture, flow_export)


def detect(path: Path, readers: Sequence[Reader] = READERS) -> Reader:
    """The first reader whose `sniff` is true for `path`.

    Raises `InputError` naming the path when it does not exist, or when no
    reader recognises it.
    """
    if not path.exists():
        raise InputError(f"{path}: no such file. Check the input path.")
    for reader in readers:
        if reader.sniff(path):
            return reader
    raise InputError(
        f"{path} is neither a capture (pcap or pcapng) nor a CICFlowMeter CSV "
        "flow export. Give a file in one of these two formats, or name its "
        "format with --input-format to see that reader's own error."
    )


def reader_named(name: str, readers: Sequence[Reader] = READERS) -> Reader:
    """The reader called `name`, without detection: its `sniff` is not called.

    Raises `InputError` listing the valid names when no reader has this name.
    """
    for reader in readers:
        if reader.name == name:
            return reader
    names = ", ".join(reader.name for reader in readers)
    raise InputError(f"Unknown input format {name!r}. Valid names: {names}.")
