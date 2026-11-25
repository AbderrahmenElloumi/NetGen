"""Tests of the error classes in netgen.errors."""

import pytest

from netgen.errors import (
    ArtefactsError,
    AutoencoderUnavailable,
    BenchmarkUnavailable,
    InputError,
    NetGenError,
)


@pytest.mark.parametrize(
    "error", [InputError, ArtefactsError, AutoencoderUnavailable, BenchmarkUnavailable]
)
def test_every_error_derives_from_netgen_error(error: type[Exception]) -> None:
    assert issubclass(error, NetGenError)
    assert issubclass(NetGenError, Exception)
