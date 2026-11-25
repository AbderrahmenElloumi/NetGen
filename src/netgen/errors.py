"""The errors NetGen raises. A message names the cause and the fix."""


class NetGenError(Exception):
    """Any error NetGen reports to the user."""


class InputError(NetGenError):
    """An input file, or a table at a seam, does not meet the flow record."""


class ArtefactsError(NetGenError):
    """An artefacts folder is missing, unreadable or does not match the input."""


class AutoencoderUnavailable(NetGenError):
    """The autoencoder was asked for, but the optional torch dependency is missing."""


class BenchmarkUnavailable(NetGenError):
    """The benchmark was asked for, but the optional scikit-learn dependency is missing."""
