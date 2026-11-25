"""The autoencoder: the optional learning stage after the scaler, with its model and training.

The stage trains on the scaler's output, restricted to the feature columns
and, with `fit_on_label`, to the rows carrying that label. Its `transform`
adds the derived column `anomaly_score` and, with `latents`, the derived
columns `latent_0` to `latent_<n-1>`. `autoencoder.pt` holds the network's
`state_dict` only and is read back with `weights_only=True`.

The network maps the input to 64, 32 and the latent size, then mirrors back
to the input size. Hidden layers use BatchNorm, LeakyReLU (slope 0.1) and
dropout 0.2. The output layer is linear, because the scaler's output is
unbounded. The anomaly score of a row is its reconstruction error: the mean
squared error over the features.

Training is seeded, runs on the CPU only with deterministic torch algorithms,
drops the incomplete last batch and needs at least `MIN_ROWS` rows. torch is
the optional extra `netgen[autoencoder]` and is imported only inside the
functions that need it, so the rest of NetGen works without it.
"""

import importlib
import pickle
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from netgen import record
from netgen.errors import ArtefactsError, AutoencoderUnavailable
from netgen.record import Role
from netgen.stages import as_float, check_columns
from netgen.table import FlowTable, TableResult

if TYPE_CHECKING:
    import torch

FILE = "autoencoder.pt"
HIDDEN_SIZES = (64, 32)
NEGATIVE_SLOPE = 0.1
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 1.0
BATCH_SIZE = 256
MIN_ROWS = BATCH_SIZE


@dataclass(frozen=True)
class Settings:
    """The exposed settings. Every other hyperparameter is a fixed module constant.

    `latents` and `fit_on_label` belong to the stage; training ignores them.
    """

    latent_dim: int = 8
    epochs: int = 50
    seed: int = 42
    latents: bool = False
    fit_on_label: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Self:
        """The settings `mapping()` recorded in the manifest."""
        return cls(
            latent_dim=int(mapping["latent_dim"]),
            epochs=int(mapping["epochs"]),
            seed=int(mapping["seed"]),
            latents=bool(mapping["latents"]),
            fit_on_label=mapping["fit_on_label"],
        )

    def mapping(self) -> dict[str, Any]:
        """Every hyperparameter, the fixed ones included, for the manifest."""
        return {
            "latent_dim": self.latent_dim,
            "epochs": self.epochs,
            "seed": self.seed,
            "hidden_sizes": list(HIDDEN_SIZES),
            "activation": "leaky_relu",
            "negative_slope": NEGATIVE_SLOPE,
            "dropout": DROPOUT,
            "batch_norm": True,
            "optimiser": "adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "schedule": "cosine",
            "gradient_clip": GRADIENT_CLIP,
            "batch_size": BATCH_SIZE,
            "drop_last": True,
            "min_rows": MIN_ROWS,
            "latents": self.latents,
            "fit_on_label": self.fit_on_label,
        }


@dataclass(frozen=True)
class Model:
    """A trained network: `network[0]` is the encoder, `network[1]` the decoder."""

    network: "torch.nn.Sequential"
    settings: Settings

    @property
    def input_size(self) -> int:
        """The number of features the network takes."""
        # The first parameter is the weight of the first linear layer: one column per feature.
        return next(self.network.parameters()).shape[1]

    def scores(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The reconstruction error of each row, in evaluation mode and without gradients."""
        inputs = _tensor(matrix)
        with _evaluating(self.network):
            errors = ((self.network(inputs) - inputs) ** 2).mean(dim=1)
        return np.asarray(errors.numpy(), dtype=np.float64)

    def encode(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The latent encoding of each row, in evaluation mode and without gradients."""
        encoder = self.network[0]
        with _evaluating(self.network):
            latent = encoder(_tensor(matrix))
        return np.asarray(latent.numpy(), dtype=np.float64)


def build_network(input_size: int, latent_dim: int) -> "torch.nn.Sequential":
    """An untrained network. The input size is the length of the feature list; it has no default."""
    from torch import nn

    def block(size_in: int, size_out: int) -> list[nn.Module]:
        return [
            nn.Linear(size_in, size_out),
            nn.BatchNorm1d(size_out),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Dropout(DROPOUT),
        ]

    encoder_sizes = [input_size, *HIDDEN_SIZES]
    decoder_sizes = [latent_dim, *reversed(HIDDEN_SIZES)]
    encoder: list[nn.Module] = []
    decoder: list[nn.Module] = []
    for size_in, size_out in pairwise(encoder_sizes):
        encoder += block(size_in, size_out)
    for size_in, size_out in pairwise(decoder_sizes):
        decoder += block(size_in, size_out)
    encoder.append(nn.Linear(HIDDEN_SIZES[-1], latent_dim))
    decoder.append(nn.Linear(HIDDEN_SIZES[0], input_size))
    return nn.Sequential(nn.Sequential(*encoder), nn.Sequential(*decoder))


def train(matrix: npt.NDArray[np.float64], settings: Settings = Settings()) -> Model:
    """A network trained on every row of `matrix`, each epoch, with no validation split.

    Raises `AutoencoderUnavailable` before anything is returned when torch is
    missing, when `matrix` has fewer than `MIN_ROWS` rows, or when the
    training loss is not finite.
    """
    _require_torch()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    rows = len(matrix)
    if rows < MIN_ROWS:
        raise AutoencoderUnavailable(
            f"The autoencoder needs at least {MIN_ROWS} training rows (one full batch); "
            f"it was given {rows}. Fit on more flows, or turn the autoencoder off."
        )

    deterministic = torch.are_deterministic_algorithms_enabled()
    # fork_rng restores the global random state on exit, so the seed changes nothing outside.
    with torch.random.fork_rng(devices=[]):
        torch.use_deterministic_algorithms(True)
        try:
            torch.manual_seed(settings.seed)
            inputs = _tensor(matrix)
            network = build_network(inputs.shape[1], settings.latent_dim)
            shuffle = torch.Generator(device="cpu").manual_seed(settings.seed)
            loader = DataLoader(
                TensorDataset(inputs),
                batch_size=BATCH_SIZE,
                shuffle=True,
                drop_last=True,
                generator=shuffle,
            )
            optimiser = torch.optim.Adam(
                network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
            )
            schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=settings.epochs
            )
            loss_of = torch.nn.MSELoss()
            network.train()
            for epoch in range(1, settings.epochs + 1):
                for (batch,) in loader:
                    optimiser.zero_grad()
                    loss = loss_of(network(batch), batch)
                    if not torch.isfinite(loss):
                        raise AutoencoderUnavailable(
                            f"The autoencoder's training loss is not finite at epoch {epoch}. "
                            "Check that the training features hold no infinite or missing values."
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), GRADIENT_CLIP)
                    optimiser.step()
                schedule.step()
        finally:
            torch.use_deterministic_algorithms(deterministic)
    network.eval()
    return Model(network, settings)


@dataclass(frozen=True)
class FittedAutoencoder:
    """A fitted autoencoder stage: the trained model and the feature list it takes, in order."""

    model: Model
    features: tuple[str, ...]
    latents: bool

    def transform(self, table: FlowTable) -> TableResult:
        """The table with `anomaly_score` and, with `latents`, the latent columns added as derived columns."""
        features = table.features()
        check_columns(features.columns, self.features, "manifest.json")
        matrix = _matrix(features, self.features)
        output = table.with_derived(record.ANOMALY_SCORE, self.model.scores(matrix))
        if self.latents:
            for index, values in enumerate(self.model.encode(matrix).T):
                output = output.with_derived(f"latent_{index}", values)
        return TableResult(output, {}, ())

    def save(self, directory: Path) -> None:
        """Write `autoencoder.pt`: the network's `state_dict` only."""
        import torch

        torch.save(self.model.network.state_dict(), directory / FILE)


def fit(table: FlowTable, settings: Settings = Settings()) -> FittedAutoencoder:
    """Train on the feature columns of `table`, or of its rows labelled `settings.fit_on_label`.

    Raises `AutoencoderUnavailable` before anything is returned when torch is
    missing, when `fit_on_label` is given for a table without a label or
    matches no row, when fewer than `MIN_ROWS` rows remain, or when the
    training loss is not finite. The label is never learned from.
    """
    _require_torch()
    features = table.features()
    names = tuple(str(name) for name in features.columns)
    if settings.fit_on_label is not None:
        features = features[_labelled(table, settings.fit_on_label)]
    model = train(_matrix(features, names), settings)
    return FittedAutoencoder(model, names, settings.latents)


def load(directory: Path, manifest: Mapping[str, Any]) -> FittedAutoencoder:
    """Read `autoencoder.pt` with `weights_only=True`; the settings and feature list come from the manifest."""
    path = directory / FILE
    if not path.is_file():
        raise ArtefactsError(
            f"The artefacts folder {directory} has no {FILE}. Run `netgen fit` to write the artefacts again."
        )
    _require_torch()
    import torch

    settings = Settings.from_mapping(manifest["autoencoder"])
    features = tuple(str(entry["name"]) for entry in manifest["features"])
    network = build_network(len(features), settings.latent_dim)
    refused = ArtefactsError(
        f"{path} is not a weights-only state dict of an autoencoder with {len(features)} features "
        f"and latent size {settings.latent_dim}. Run `netgen fit` to write the artefacts again."
    )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    # A truncated zip makes torch's reader raise OSError, an empty file EOFError.
    except (pickle.UnpicklingError, RuntimeError, EOFError, OSError) as error:
        raise refused from error
    # load_state_dict raises AttributeError, not RuntimeError, on a key that is not a string.
    # It refuses a value that is not a tensor with RuntimeError itself.
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) for name in state
    ):
        raise refused
    try:
        network.load_state_dict(state)
    except RuntimeError as error:
        raise refused from error
    network.eval()
    return FittedAutoencoder(Model(network, settings), features, settings.latents)


def _require_torch() -> None:
    """Raise `AutoencoderUnavailable` naming the extra when torch is not installed."""
    try:
        importlib.import_module("torch")
    except ImportError as error:
        raise AutoencoderUnavailable(
            "The autoencoder needs torch, which is not installed. "
            "Install the extra: pip install 'netgen[autoencoder]'."
        ) from error


def _labelled(table: FlowTable, label: str) -> npt.NDArray[np.bool_]:
    """Which rows of `table` carry `label`. An unlabelled row never matches."""
    columns = [name for name, role in table.roles.items() if role is Role.LABEL]
    if not columns:
        raise AutoencoderUnavailable(
            f"The autoencoder was asked to fit on the label {label!r}, but the table has no label column. "
            "Fit on a labelled input, or leave out --fit-on-label."
        )
    labels = table.to_frame()[columns[0]].astype("string")
    rows = np.asarray((labels == label).fillna(False).to_numpy(dtype=bool))
    if not rows.any():
        raise AutoencoderUnavailable(
            f"The autoencoder was asked to fit on the label {label!r}, which no training row carries. "
            "Give a label the training data holds."
        )
    return rows


def _matrix(features: pd.DataFrame, names: Iterable[str]) -> npt.NDArray[np.float64]:
    """The feature columns `names`, in that order, as one float matrix with a row per table row."""
    return np.column_stack([as_float(features[name]) for name in names])


def _tensor(matrix: npt.NDArray[np.float64]) -> "torch.Tensor":
    """A float32 CPU tensor of `matrix`."""
    import torch

    return torch.as_tensor(np.asarray(matrix), dtype=torch.float32, device="cpu")


@contextmanager
def _evaluating(network: "torch.nn.Module") -> Iterator[None]:
    """Evaluation mode (dropout off, BatchNorm running statistics) and no gradients."""
    import torch

    network.eval()
    with torch.no_grad():
        yield
