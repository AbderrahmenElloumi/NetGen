"""Tests of the autoencoder (netgen.stages.autoencoder): the model, its training and the stage.

Training uses 1 or 2 epochs on tiny matrices of 256 to 300 rows, so each
test stays fast. Determinism is checked by exact equality between two runs
in one process: no golden values are committed.
"""

import dataclasses
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import torch

from netgen import record
from netgen.errors import ArtefactsError, AutoencoderUnavailable
from netgen.record import Role
from netgen.stages import Fitted, Stage, autoencoder, scaler
from netgen.stages.autoencoder import FittedAutoencoder, Model, Settings
from netgen.table import FlowTable

QUICK = Settings(epochs=1)
FEATURES = [field.name for field in record.fields_by_role(Role.FEATURE)[:5]]


def matrix(rows: int, columns: int = 5, seed: int = 0) -> npt.NDArray[np.float64]:
    return np.random.default_rng(seed).standard_normal((rows, columns))


def weights(model: Model) -> dict[str, torch.Tensor]:
    return model.network.state_dict()


# Carried defects (spec section 16), written first.


@pytest.mark.parametrize("columns", [5, 40])
def test_d40_input_size_is_the_column_count(columns: int) -> None:
    model = autoencoder.train(matrix(256, columns), QUICK)
    assert model.input_size == columns
    assert model.network[0][0].in_features == columns
    assert model.network[1][-1].out_features == columns


def test_d40_no_default_input_size_exists() -> None:
    size = inspect.signature(autoencoder.build_network).parameters["input_size"]
    assert size.default is inspect.Parameter.empty
    fields = {field.name for field in dataclasses.fields(Settings)}
    assert not any("input" in name for name in fields)


@pytest.mark.parametrize("rows", [256, 257])
def test_d41_training_drops_the_incomplete_last_batch(rows: int) -> None:
    # 257 rows leave a last batch of one row, which BatchNorm refuses in training mode.
    model = autoencoder.train(matrix(rows), Settings(epochs=2))
    assert len(model.scores(matrix(rows))) == rows


def test_d41_fewer_than_256_rows_is_refused() -> None:
    with pytest.raises(AutoencoderUnavailable, match=r"256.*255"):
        autoencoder.train(matrix(255), QUICK)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_d43_a_non_finite_loss_fails_loudly(value: float) -> None:
    rows = matrix(256)
    rows[3, 2] = value
    with pytest.raises(AutoencoderUnavailable, match="epoch 1"):
        autoencoder.train(rows, QUICK)


# Settled defect D39: the output layer is linear.


def test_d39_the_output_layer_has_no_activation() -> None:
    network = autoencoder.build_network(input_size=5, latent_dim=3)
    decoder = network[1]
    assert isinstance(decoder[-1], torch.nn.Linear)
    assert decoder[-1].out_features == 5


def test_d39_the_module_docstring_does_not_say_sigmoid() -> None:
    assert "sigmoid" not in (autoencoder.__doc__ or "").lower()


# Defect D42 (the part this module owns): seeded shuffle, CPU only.


def test_d42_the_model_parameters_are_on_the_cpu() -> None:
    model = autoencoder.train(matrix(256), QUICK)
    assert {parameter.device.type for parameter in model.network.parameters()} == {
        "cpu"
    }


# Architecture and settings.


def test_the_network_mirrors_the_hidden_sizes_around_the_latent_layer() -> None:
    network = autoencoder.build_network(input_size=5, latent_dim=3)
    linear = [
        (layer.in_features, layer.out_features)
        for layer in network.modules()
        if isinstance(layer, torch.nn.Linear)
    ]
    assert linear == [(5, 64), (64, 32), (32, 3), (3, 32), (32, 64), (64, 5)]
    kinds = {type(layer) for layer in network.modules()}
    assert {torch.nn.BatchNorm1d, torch.nn.LeakyReLU, torch.nn.Dropout} <= kinds
    slopes = {
        layer.negative_slope
        for layer in network.modules()
        if isinstance(layer, torch.nn.LeakyReLU)
    }
    rates = {
        layer.p for layer in network.modules() if isinstance(layer, torch.nn.Dropout)
    }
    assert (slopes, rates) == ({0.1}, {0.2})


def test_the_settings_have_the_spec_defaults_and_are_frozen() -> None:
    settings = Settings()
    assert (settings.latent_dim, settings.epochs, settings.seed) == (8, 50, 42)
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.epochs = 1  # type: ignore[misc]


def test_the_settings_mapping_holds_every_hyperparameter() -> None:
    assert Settings(latent_dim=4, epochs=3, seed=7).mapping() == {
        "latent_dim": 4,
        "epochs": 3,
        "seed": 7,
        "hidden_sizes": [64, 32],
        "activation": "leaky_relu",
        "negative_slope": 0.1,
        "dropout": 0.2,
        "batch_norm": True,
        "optimiser": "adam",
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "schedule": "cosine",
        "gradient_clip": 1.0,
        "batch_size": 256,
        "drop_last": True,
        "min_rows": 256,
        "latents": False,
        "fit_on_label": None,
    }


# Determinism.


def test_two_seeded_runs_give_bit_identical_weights_and_scores() -> None:
    rows = matrix(300)
    first = autoencoder.train(rows, Settings(epochs=2))
    second = autoencoder.train(rows, Settings(epochs=2))
    first_weights, second_weights = weights(first), weights(second)
    assert first_weights.keys() == second_weights.keys()
    for name, tensor in first_weights.items():
        assert torch.equal(tensor, second_weights[name]), name
    assert np.array_equal(first.scores(rows), second.scores(rows))
    assert np.array_equal(first.encode(rows), second.encode(rows))


def test_a_different_seed_gives_different_weights() -> None:
    rows = matrix(300)
    first = weights(autoencoder.train(rows, Settings(epochs=1, seed=1)))
    second = weights(autoencoder.train(rows, Settings(epochs=1, seed=2)))
    assert any(not torch.equal(first[name], second[name]) for name in first)


def test_training_restores_the_global_torch_state() -> None:
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    autoencoder.train(matrix(256), QUICK)
    assert torch.equal(torch.rand(3), expected)
    assert torch.are_deterministic_algorithms_enabled() == deterministic


# Scoring and latent encoding.


def test_scores_are_one_finite_reconstruction_error_per_row() -> None:
    rows = matrix(256)
    model = autoencoder.train(rows, QUICK)
    scores = model.scores(rows)
    assert scores.shape == (256,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()
    with torch.no_grad():
        model.network.eval()
        inputs = torch.as_tensor(rows, dtype=torch.float32)
        expected = ((model.network(inputs) - inputs) ** 2).mean(dim=1)
    assert np.array_equal(scores, expected.numpy().astype(np.float64))


def test_scoring_uses_evaluation_mode_and_leaves_the_model_unchanged() -> None:
    # Ticket 34 loads into a network that build_network returns in training mode.
    rows = matrix(256)
    model = autoencoder.train(rows, QUICK)
    before = {name: tensor.clone() for name, tensor in weights(model).items()}
    with torch.no_grad():
        model.network.eval()
        inputs = torch.as_tensor(rows, dtype=torch.float32)
        expected_scores = ((model.network(inputs) - inputs) ** 2).mean(dim=1)
        expected_latent = model.network[0](inputs)
    model.network.train()
    scores = model.scores(rows)
    assert not model.network.training
    model.network.train()
    latent = model.encode(rows)
    assert not model.network.training
    assert np.array_equal(scores, expected_scores.numpy().astype(np.float64))
    assert np.array_equal(latent, expected_latent.numpy().astype(np.float64))
    for name, tensor in weights(model).items():
        assert torch.equal(tensor, before[name]), name


def test_the_latent_encoding_has_latent_dim_columns() -> None:
    model = autoencoder.train(matrix(256), Settings(latent_dim=3, epochs=1))
    encoded = model.encode(matrix(10))
    assert encoded.shape == (10, 3)
    assert encoded.dtype == np.float64
    assert np.isfinite(encoded).all()


# Without torch.


def test_without_torch_the_core_works_and_training_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = ("netgen.record", "netgen.table", "netgen.stages")
    for name in list(sys.modules):
        if name == "torch" or name.startswith("torch.") or name.startswith(core):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch", None)

    record = importlib.import_module("netgen.record")
    table = importlib.import_module("netgen.table")
    fill = importlib.import_module("netgen.stages.fill")
    scaler = importlib.import_module("netgen.stages.scaler")
    hidden = importlib.import_module("netgen.stages.autoencoder")

    frame = pd.DataFrame(
        {
            "src_ip": pd.Series(["192.0.2.1", "192.0.2.2"], dtype="string"),
            "duration": pd.Series([1.0, float("inf")], dtype=float),
        }
    )
    flows = table.FlowTable(
        frame, {"src_ip": record.Role.IDENTITY, "duration": record.Role.FEATURE}
    )
    filled = fill.fit(flows).transform(flows).table
    scaled = scaler.fit(filled).transform(filled).table
    assert list(scaled.to_frame()["duration"]) == [0.0, 0.0]

    with pytest.raises(AutoencoderUnavailable, match=r"netgen\[autoencoder\]"):
        hidden.train(matrix(256), hidden.Settings(epochs=1))


# The stage: fit, transform, save and load (spec section 8).


def flow_table(
    rows: int,
    labels: list[str | None] | None = None,
    features: list[str] = FEATURES,
    seed: int = 0,
) -> FlowTable:
    # Non-negative values, so the scaler's log step stays finite.
    values = np.abs(matrix(rows, len(features), seed))
    columns: dict[str, pd.Series] = {
        "src_ip": pd.Series([f"192.0.2.{i % 250}" for i in range(rows)], dtype="string")
    }
    roles = {"src_ip": Role.IDENTITY}
    if labels is not None:
        columns["label"] = pd.Series(labels, dtype="string")
        roles["label"] = Role.LABEL
    for index, name in enumerate(features):
        columns[name] = pd.Series(values[:, index], dtype=float)
        roles[name] = Role.FEATURE
    return FlowTable(pd.DataFrame(columns), roles)


def manifest(settings: Settings, features: list[str] = FEATURES) -> dict[str, Any]:
    """The two manifest keys the stage reads: the feature list and its settings."""
    return {
        "features": [
            {"name": name, "log": record.feature_scaling(name) is record.Scaling.LOG}
            for name in features
        ],
        "autoencoder": settings.mapping(),
    }


def feature_matrix(table: FlowTable) -> npt.NDArray[np.float64]:
    # A writable copy: torch warns on a read-only array.
    return table.features()[FEATURES].to_numpy(dtype=float, copy=True)


def spy_on_training(
    monkeypatch: pytest.MonkeyPatch,
) -> list[npt.NDArray[np.float64]]:
    """Record every matrix the stage trains on, then train as usual."""
    seen: list[npt.NDArray[np.float64]] = []
    train = autoencoder.train

    def spy(rows: npt.NDArray[np.float64], settings: Settings = QUICK) -> Model:
        seen.append(rows.copy())
        return train(rows, settings)

    monkeypatch.setattr(autoencoder, "train", spy)
    return seen


EXECUTED: list[str] = []


def _execute() -> None:
    EXECUTED.append("executed")


class _Payload:
    """A pickle payload that calls a function when it is unpickled."""

    def __reduce__(self) -> tuple[Any, tuple[()]]:
        return (_execute, ())


def test_d42_autoencoder_loads_weights_only_and_is_seeded(tmp_path: Path) -> None:
    table = flow_table(300)
    first = autoencoder.fit(table, Settings(epochs=2))
    second = autoencoder.fit(table, Settings(epochs=2))
    first.save(tmp_path)
    path = tmp_path / "autoencoder.pt"

    state = torch.load(path, weights_only=True)
    assert isinstance(state, dict)
    expected = first.model.network.state_dict()
    assert list(state) == list(expected)
    for name, tensor in state.items():
        assert isinstance(tensor, torch.Tensor)
        assert torch.equal(tensor, expected[name]), name
        assert torch.equal(tensor, second.model.network.state_dict()[name]), name

    torch.save(_Payload(), path)
    with pytest.raises(ArtefactsError, match="autoencoder.pt"):
        autoencoder.load(tmp_path, manifest(Settings(epochs=2)))
    assert EXECUTED == []


def test_a_weights_file_that_is_not_a_state_dict_is_refused(tmp_path: Path) -> None:
    torch.save([torch.zeros(1)], tmp_path / "autoencoder.pt")
    with pytest.raises(ArtefactsError, match="autoencoder.pt"):
        autoencoder.load(tmp_path, manifest(QUICK))


def _write_empty(path: Path) -> None:
    path.write_bytes(b"")


def _write_truncated(path: Path) -> None:
    torch.save(autoencoder.fit(flow_table(256), QUICK).model.network.state_dict(), path)
    path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


@pytest.mark.parametrize(
    "write", [_write_empty, _write_truncated], ids=["empty", "truncated"]
)
def test_an_empty_or_truncated_weights_file_is_refused(
    tmp_path: Path, write: Any
) -> None:
    write(tmp_path / "autoencoder.pt")
    with pytest.raises(ArtefactsError, match="autoencoder.pt"):
        autoencoder.load(tmp_path, manifest(QUICK))


def _value_not_a_tensor(state: dict[Any, Any]) -> dict[Any, Any]:
    """Every key the network expects, one value an integer."""
    first = next(iter(state))
    return {**state, first: 1}


def _key_not_a_string(state: dict[Any, Any]) -> dict[Any, Any]:
    return {1: torch.zeros(1)}


@pytest.mark.parametrize(
    "corrupt",
    [_value_not_a_tensor, _key_not_a_string],
    ids=["value-not-a-tensor", "key-not-a-string"],
)
def test_a_mapping_that_is_not_names_to_tensors_is_refused(
    tmp_path: Path, corrupt: Any
) -> None:
    fitted = autoencoder.fit(flow_table(256), QUICK)
    state = corrupt(dict(weights(fitted.model)))
    path = tmp_path / "autoencoder.pt"
    torch.save(state, path)
    assert list(torch.load(path, weights_only=True)) == list(state)
    with pytest.raises(ArtefactsError, match="autoencoder.pt"):
        autoencoder.load(tmp_path, manifest(QUICK))


def test_weights_that_do_not_fit_the_manifest_features_are_refused(
    tmp_path: Path,
) -> None:
    autoencoder.fit(flow_table(256), QUICK).save(tmp_path)
    with pytest.raises(ArtefactsError, match="autoencoder.pt"):
        autoencoder.load(tmp_path, manifest(QUICK, FEATURES[:4]))


def test_the_stage_conforms_to_the_stage_protocol() -> None:
    stage: Stage = autoencoder
    assert isinstance(stage, Stage)
    fitted: Fitted = autoencoder.fit(flow_table(256), QUICK)
    assert isinstance(fitted, Fitted)
    assert isinstance(fitted, FittedAutoencoder)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fitted.latents = True  # type: ignore[misc]


def test_d36_a_fitted_autoencoder_cannot_be_fitted_again() -> None:
    table = flow_table(256)
    first = autoencoder.fit(table, QUICK)
    before_weights = {
        name: tensor.clone() for name, tensor in weights(first.model).items()
    }
    before = first.transform(table).table.to_frame()
    second = autoencoder.fit(flow_table(300, seed=1), Settings(epochs=2, seed=7))
    assert second.model is not first.model
    after_weights = weights(first.model)
    assert list(after_weights) == list(before_weights)
    for name, tensor in after_weights.items():
        assert torch.equal(tensor, before_weights[name]), name
    pd.testing.assert_frame_equal(
        first.transform(table).table.to_frame(), before, check_exact=True
    )
    assert not hasattr(first, "fit")
    assert not hasattr(first.model, "fit")
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.model = second.model  # type: ignore[misc]


def test_save_load_and_transform_give_the_fitted_transform(tmp_path: Path) -> None:
    settings = Settings(latent_dim=3, epochs=1, latents=True)
    fitted = autoencoder.fit(flow_table(256), settings)
    fitted.save(tmp_path)
    loaded = autoencoder.load(tmp_path, manifest(settings))
    test = flow_table(40, seed=1)
    pd.testing.assert_frame_equal(
        loaded.transform(test).table.to_frame(),
        fitted.transform(test).table.to_frame(),
        check_exact=True,
    )


def test_the_anomaly_score_is_a_derived_column_written_as_computed() -> None:
    table = flow_table(256)
    fitted = autoencoder.fit(table, QUICK)
    result = fitted.transform(table)
    output = result.table
    assert output.roles["anomaly_score"] is Role.DERIVED
    assert not any(name.startswith("latent_") for name in output.roles)
    assert np.array_equal(
        output.to_frame()["anomaly_score"].to_numpy(),
        fitted.model.scores(feature_matrix(table)),
    )
    pd.testing.assert_frame_equal(output.features(), table.features())
    assert (dict(result.counts), result.warnings) == ({}, ())


def test_latent_columns_appear_only_with_latents() -> None:
    table = flow_table(256)
    fitted = autoencoder.fit(table, Settings(latent_dim=3, epochs=1, latents=True))
    output = fitted.transform(table).table
    latent = [f"latent_{index}" for index in range(3)]
    assert all(output.roles[name] is Role.DERIVED for name in latent)
    assert np.array_equal(
        output.to_frame()[latent].to_numpy(),
        fitted.model.encode(feature_matrix(table)),
    )


def test_derived_columns_are_never_features_and_are_not_rescaled() -> None:
    table = flow_table(256)
    fitted = autoencoder.fit(table, Settings(latent_dim=3, epochs=1, latents=True))
    output = fitted.transform(table).table
    derived = [name for name, role in output.roles.items() if role is Role.DERIVED]
    assert derived == ["anomaly_score", "latent_0", "latent_1", "latent_2"]
    assert not set(derived) & set(output.features().columns)
    rescaled = scaler.fit(output).transform(output).table.to_frame()
    pd.testing.assert_frame_equal(
        rescaled[derived], output.to_frame()[derived], check_exact=True
    )


def test_identity_label_and_derived_columns_never_enter_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = flow_table(256, labels=["BENIGN"] * 256).with_derived(
        "anomaly_score", np.zeros(256)
    )
    seen = spy_on_training(monkeypatch)
    autoencoder.fit(table, QUICK)
    assert len(seen) == 1
    assert np.array_equal(seen[0], feature_matrix(table))


def test_fit_on_label_trains_on_the_matching_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels: list[str | None] = ["BENIGN"] * 260 + ["DDoS"] * 39 + [None]
    table = flow_table(300, labels=labels)
    fitted_scaler = scaler.fit(table)
    scaled = fitted_scaler.transform(table).table
    before = scaled.to_frame()
    seen = spy_on_training(monkeypatch)
    autoencoder.fit(scaled, Settings(epochs=1, fit_on_label="BENIGN"))
    assert len(seen[0]) == 260
    assert np.array_equal(seen[0], feature_matrix(scaled)[:260])
    # The scaler was fitted on every row; the stage changes neither it nor its output.
    assert fitted_scaler.values == scaler.fit(table).values
    pd.testing.assert_frame_equal(scaled.to_frame(), before)


# Failure modes: each raises from fit, so the caller writes nothing (D04).


@pytest.mark.parametrize(
    ("table", "settings", "message"),
    [
        (
            flow_table(300, labels=["BENIGN"] * 300),
            Settings(epochs=1, fit_on_label="DDoS"),
            "'DDoS'",
        ),
        (
            flow_table(300),
            Settings(epochs=1, fit_on_label="BENIGN"),
            "no label",
        ),
        (
            flow_table(300, labels=["BENIGN"] * 200 + ["DDoS"] * 100),
            Settings(epochs=1, fit_on_label="BENIGN"),
            r"256.*200",
        ),
        (flow_table(255), QUICK, r"256.*255"),
    ],
    ids=[
        "label-matches-no-rows",
        "table-without-label",
        "too-few-label-rows",
        "too-few-rows",
    ],
)
def test_d04_an_impossible_fit_fails_loudly_and_writes_nothing(
    table: FlowTable,
    settings: Settings,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AutoencoderUnavailable, match=message):
        autoencoder.fit(table, settings)
    assert list(tmp_path.iterdir()) == []


def test_d04_fit_without_torch_names_the_extra_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = flow_table(300)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(AutoencoderUnavailable, match=r"netgen\[autoencoder\]"):
        autoencoder.fit(table, QUICK)
    assert list(tmp_path.iterdir()) == []


def test_load_without_torch_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    autoencoder.fit(flow_table(256), QUICK).save(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(AutoencoderUnavailable, match=r"netgen\[autoencoder\]"):
        autoencoder.load(tmp_path, manifest(QUICK))


# Rejection branches.


def test_transform_refuses_a_table_with_other_feature_columns() -> None:
    fitted = autoencoder.fit(flow_table(256), QUICK)
    other = [*FEATURES[:4], record.fields_by_role(Role.FEATURE)[5].name]
    with pytest.raises(ArtefactsError, match=FEATURES[4]):
        fitted.transform(flow_table(10, features=other))


def test_load_of_a_folder_without_the_weights_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtefactsError, match="autoencoder.pt"):
        autoencoder.load(tmp_path, manifest(QUICK))
