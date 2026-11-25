"""Tests of the preparation (netgen.preparation): the order of stages, the artefacts and the manifest."""

import hashlib
import json
import tomllib
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import netgen
from netgen import preparation, record
from netgen.errors import ArtefactsError, InputError
from netgen.preparation import Artefacts, Settings, fit_preparation
from netgen.record import FlowSettings, Role, Scaling
from netgen.stages import autoencoder
from netgen.table import FlowTable

FEATURES = [field.name for field in record.fields_by_role(Role.FEATURE)]
ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "ab" * 32
QUICK = autoencoder.Settings(latent_dim=2, epochs=1)


def record_frame(rows: int = 300, seed: int = 0) -> pd.DataFrame:
    """A frame holding every field of the flow record, half the rows labelled BENIGN."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "src_ip": pd.Series(
                [f"192.0.2.{i % 250}" for i in range(rows)], dtype="string"
            ),
            "dst_ip": pd.Series(["198.51.100.1"] * rows, dtype="string"),
            "src_port": pd.Series([40000 + i for i in range(rows)], dtype="Int64"),
            "dst_port": pd.Series(rng.integers(1, 1024, rows), dtype="Int64"),
            "protocol": pd.Series(["tcp"] * rows, dtype="string"),
            "start_time": pd.Series(
                pd.date_range("2017-07-07 03:30", periods=rows, freq="s")
            ),
            "label": pd.Series(
                ["BENIGN" if i % 2 else "ATTACK" for i in range(rows)], dtype="string"
            ),
        }
    )
    for name in FEATURES:
        frame[name] = rng.exponential(10.0, rows)
    return frame


def flow_table(rows: int = 300, seed: int = 0) -> FlowTable:
    return FlowTable.from_frame(record_frame(rows, seed))


def settings(**changes: Any) -> Settings:
    return Settings(fingerprint=FINGERPRINT, **changes)


def saved(tmp_path: Path, artefacts: Artefacts) -> Path:
    artefacts.save(tmp_path)
    return tmp_path


def read(directory: Path, file: str) -> Any:
    return json.loads((directory / file).read_text(encoding="utf-8"))


def write(directory: Path, file: str, state: Any) -> None:
    (directory / file).write_text(json.dumps(state), encoding="utf-8")


def pyproject_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


# Named regression test (D37)


@pytest.mark.parametrize("column", ["duration", "src_ip"])
def test_d37_missing_columns_are_refused_not_skipped(column: str) -> None:
    artefacts = fit_preparation(flow_table(), settings())
    lacking = FlowTable.from_frame(record_frame(seed=1).drop(columns=column))

    with pytest.raises(ArtefactsError, match=rf"lacks the columns \['{column}'\]"):
        artefacts.transform(lacking)


# The manifest


def test_the_manifest_records_the_roles_settings_features_and_stages(
    tmp_path: Path,
) -> None:
    table = flow_table()
    chosen = settings(
        seed=7, label_column="Label", flow_settings=FlowSettings(60.0, 300.0)
    )
    manifest = read(saved(tmp_path, fit_preparation(table, chosen)), "manifest.json")

    assert manifest["netgen_version"] == pyproject_version()
    assert manifest["format_version"] == 1
    assert manifest["columns"] == {
        name: role.value for name, role in table.roles.items()
    }
    assert manifest["features"] == [
        {"name": name, "log": record.feature_scaling(name) is Scaling.LOG}
        for name in FEATURES
    ]
    assert manifest["label_column"] == "Label"
    assert manifest["fingerprint"] == FINGERPRINT
    assert manifest["seed"] == 7
    assert manifest["flow_settings"] == {"idle_timeout": 60.0, "active_timeout": 300.0}
    assert manifest["autoencoder"] is None
    assert manifest["stages"] == [
        {"name": "fill", "file": "fill.json"},
        {"name": "scaler", "file": "scaler.json"},
    ]


def test_the_manifest_records_every_autoencoder_setting_and_its_stage(
    tmp_path: Path,
) -> None:
    chosen = settings(seed=7, autoencoder=QUICK)
    manifest = read(
        saved(tmp_path, fit_preparation(flow_table(), chosen)), "manifest.json"
    )

    assert (
        manifest["autoencoder"]
        == autoencoder.Settings(latent_dim=2, epochs=1, seed=7).mapping()
    )
    assert manifest["stages"][-1] == {"name": "autoencoder", "file": "autoencoder.pt"}


def test_the_seed_of_the_settings_is_the_seed_of_the_autoencoder() -> None:
    chosen = settings(seed=7, autoencoder=autoencoder.Settings(seed=1))

    assert chosen.autoencoder is not None
    assert chosen.autoencoder.seed == 7


def test_the_manifest_version_is_the_pyproject_version() -> None:
    assert netgen.__version__ == pyproject_version()


def test_the_version_comes_from_the_package_metadata_when_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata, "version", lambda name: "9.9.9")

    assert netgen._version() == "9.9.9"


def test_the_version_falls_back_to_pyproject_without_package_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)

    assert netgen._version() == pyproject_version()


def test_the_fingerprint_is_the_sha256_of_the_input_bytes(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(b"src_ip,dst_ip\n192.0.2.1,198.51.100.1\n")

    assert (
        preparation.fingerprint(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    )


# The order of stages


def test_the_scaler_fits_on_the_filled_table() -> None:
    frame = record_frame()
    frame.loc[[0, 5], "duration"] = [np.inf, -np.inf]
    frame.loc[3, "fwd_packets"] = np.nan
    artefacts = fit_preparation(FlowTable.from_frame(frame), settings())

    learned = artefacts.scaler.values
    for name in ("duration", "fwd_packets"):
        assert np.isfinite([learned[name].mean, learned[name].std]).all()
    assert artefacts.fit_counts["filled"]["duration"] == 2
    assert [name for name, _, _ in artefacts.stages()] == ["fill", "scaler"]


def test_the_autoencoder_trains_on_the_scaled_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = spy_on_training(monkeypatch)
    table = flow_table()
    artefacts = fit_preparation(table, settings(autoencoder=QUICK))

    filled = artefacts.fill.transform(table).table
    scaled = artefacts.scaler.transform(filled).table.features()[FEATURES]
    assert np.array_equal(seen[0], scaled.to_numpy(dtype=float))
    assert [name for name, _, _ in artefacts.stages()] == [
        "fill",
        "scaler",
        "autoencoder",
    ]


def test_fit_refuses_a_table_that_lacks_a_record_field() -> None:
    lacking = FlowTable.from_frame(record_frame().drop(columns="duration"))

    with pytest.raises(InputError, match="duration"):
        fit_preparation(lacking, settings())


@pytest.mark.parametrize(
    ("column", "values"),
    [
        ("protocol", pd.Series([6] * 300, dtype="Int64")),
        ("start_time", pd.Series(["2017-07-07 03:30"] * 300, dtype="string")),
    ],
)
def test_transform_refuses_a_table_off_the_flow_record(
    column: str, values: pd.Series
) -> None:
    artefacts = fit_preparation(flow_table(), settings())
    off_record = FlowTable.from_frame(record_frame(seed=1).assign(**{column: values}))

    with pytest.raises(InputError, match=rf"The column '{column}' has dtype"):
        artefacts.transform(off_record)


def test_fit_keeps_its_counts_and_warnings_for_the_report() -> None:
    frame = record_frame()
    frame["duration"] = np.nan
    frame["fwd_packets"] = 5.0
    promoted = FlowTable.from_frame(frame, dst_port_as_feature=True)
    fitted = fit_preparation(promoted, settings(dst_port_as_feature=True))

    counts = fitted.fit_counts
    assert counts["filled"]["duration"] == 300
    assert counts["no_finite_value"] == ("duration",)
    assert set(counts["constant_features"]) == {"duration", "fwd_packets"}
    assert set(counts["out_of_range"]) == set(fitted.scaler.values)
    assert counts["constant_mismatch"] == {"duration": 0, "fwd_packets": 0}
    assert any("'duration' has no finite value" in w for w in fitted.fit_warnings)
    assert any(
        "identity field 'dst_port' is a feature" in w for w in fitted.fit_warnings
    )


def test_counts_and_warnings_of_the_fit_are_not_saved(tmp_path: Path) -> None:
    frame = record_frame()
    frame["duration"] = np.nan
    fitted = fit_preparation(FlowTable.from_frame(frame), settings())
    directory = saved(tmp_path, fitted)
    loaded = Artefacts.load(directory)

    assert fitted.fit_counts and fitted.fit_warnings
    assert dict(loaded.fit_counts) == {}
    assert loaded.fit_warnings == ()
    assert set(read(directory, "manifest.json")) == {
        "netgen_version",
        "format_version",
        "columns",
        "features",
        "label_column",
        "fingerprint",
        "seed",
        "flow_settings",
        "autoencoder",
        "stages",
    }


def spy_on_training(
    monkeypatch: pytest.MonkeyPatch,
) -> list[npt.NDArray[np.float64]]:
    """Record every matrix the autoencoder trains on, then train as usual."""
    seen: list[npt.NDArray[np.float64]] = []
    train = autoencoder.train

    def spy(
        rows: npt.NDArray[np.float64], chosen: autoencoder.Settings = QUICK
    ) -> autoencoder.Model:
        seen.append(rows.copy())
        return train(rows, chosen)

    monkeypatch.setattr(autoencoder, "train", spy)
    return seen


def test_fit_on_label_restricts_the_autoencoder_rows_not_the_scaler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = spy_on_training(monkeypatch)
    frame = record_frame(600)
    chosen = settings(
        autoencoder=autoencoder.Settings(latent_dim=2, epochs=1, fit_on_label="BENIGN")
    )
    artefacts = fit_preparation(FlowTable.from_frame(frame), chosen)

    assert len(seen[0]) == int((frame["label"] == "BENIGN").sum())
    for name in FEATURES:
        values = frame[name].to_numpy(dtype=float)
        if record.feature_scaling(name) is Scaling.LOG:
            values = np.log1p(values)
        assert artefacts.scaler.values[name].mean == pytest.approx(values.mean())


def test_dst_port_as_feature_is_recorded_and_applied_at_transform(
    tmp_path: Path,
) -> None:
    fitted = fit_preparation(flow_table(), settings(dst_port_as_feature=True))
    directory = saved(tmp_path, fitted)
    manifest = read(directory, "manifest.json")
    loaded = Artefacts.load(directory)

    assert manifest["columns"]["dst_port"] == "feature"
    assert {"name": "dst_port", "log": False} in manifest["features"]
    assert loaded.settings.dst_port_as_feature
    output = loaded.transform(flow_table(seed=1)).table
    assert output.roles["dst_port"] is Role.FEATURE


def test_without_dst_port_as_feature_a_promoted_port_is_an_identity_column() -> None:
    promoted = FlowTable.from_frame(record_frame(), dst_port_as_feature=True)
    artefacts = fit_preparation(promoted, settings())

    assert artefacts.roles["dst_port"] is Role.IDENTITY
    assert "dst_port" not in artefacts.scaler.values


# Save, load and transform


@pytest.mark.parametrize("stage", [None, QUICK], ids=["without", "with-autoencoder"])
def test_d06_save_load_and_transform_give_the_fitted_transform(
    tmp_path: Path, stage: autoencoder.Settings | None
) -> None:
    fitted = fit_preparation(flow_table(), settings(autoencoder=stage))
    loaded = Artefacts.load(saved(tmp_path, fitted))
    other = flow_table(seed=1)

    assert loaded.settings == fitted.settings
    assert loaded.roles == fitted.roles
    assert loaded.fill.values == fitted.fill.values
    assert loaded.scaler.values == fitted.scaler.values
    pd.testing.assert_frame_equal(
        loaded.transform(other).table.to_frame(),
        fitted.transform(other).table.to_frame(),
    )


@pytest.mark.parametrize(
    ("stage", "files"),
    [
        (None, ["fill.json", "manifest.json", "scaler.json"]),
        (QUICK, ["autoencoder.pt", "fill.json", "manifest.json", "scaler.json"]),
    ],
    ids=["without", "with-autoencoder"],
)
def test_the_artefacts_folder_holds_exactly_the_stage_files_and_the_manifest(
    tmp_path: Path, stage: autoencoder.Settings | None, files: list[str]
) -> None:
    saved(tmp_path, fit_preparation(flow_table(), settings(autoencoder=stage)))

    assert sorted(path.name for path in tmp_path.iterdir()) == files


def test_derived_columns_are_added_and_identity_and_label_pass_through() -> None:
    chosen = settings(
        autoencoder=autoencoder.Settings(latent_dim=2, epochs=1, latents=True)
    )
    artefacts = fit_preparation(flow_table(), chosen)
    other = flow_table(seed=1)
    output = artefacts.transform(other).table

    derived = [name for name, role in output.roles.items() if role is Role.DERIVED]
    assert derived == ["anomaly_score", "latent_0", "latent_1"]
    passed = [
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "start_time",
        "label",
    ]
    pd.testing.assert_frame_equal(output.to_frame()[passed], other.to_frame()[passed])


def test_a_label_passes_through_when_the_fit_input_had_no_label() -> None:
    unlabelled = FlowTable.from_frame(record_frame().drop(columns="label"))
    artefacts = fit_preparation(unlabelled, settings())
    labelled = flow_table(seed=1)

    result = artefacts.transform(labelled)

    assert result.table.roles["label"] is Role.LABEL
    pd.testing.assert_series_equal(
        result.table.to_frame()["label"], labelled.to_frame()["label"]
    )
    assert not any("label" in warning for warning in result.warnings)


def test_latents_in_the_manifest_decide_the_latent_columns_at_transform(
    tmp_path: Path,
) -> None:
    directory = saved(
        tmp_path, fit_preparation(flow_table(), settings(autoencoder=QUICK))
    )
    manifest = read(directory, "manifest.json")
    manifest["autoencoder"]["latents"] = True
    write(directory, "manifest.json", manifest)

    output = Artefacts.load(directory).transform(flow_table(seed=1)).table

    assert "latent_0" in output.roles
    assert "latent_1" in output.roles


def test_extra_columns_that_are_not_features_are_dropped_with_a_warning() -> None:
    artefacts = fit_preparation(flow_table(), settings())
    other = flow_table(seed=1)
    extra = other.with_derived("anomaly_score", np.zeros(300))

    result = artefacts.transform(extra)

    assert "anomaly_score" not in result.table.roles
    assert any("['anomaly_score']" in warning for warning in result.warnings)


# Refusals


def test_an_extra_feature_column_is_refused_naming_scaler_json() -> None:
    artefacts = fit_preparation(flow_table(), settings())
    frame = record_frame(seed=1).assign(bogus=1.0)
    roles = {**record.default_roles(frame.columns), "bogus": Role.FEATURE}

    with pytest.raises(
        ArtefactsError, match=r"scaler\.json: missing \[\], extra \['bogus'\]"
    ):
        artefacts.transform(FlowTable(frame, roles))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda state: state.pop("duration"), r"missing \[\], extra \['duration'\]"),
        (
            lambda state: state.update(bogus=state["duration"]),
            r"missing \['bogus'\], extra \[\]",
        ),
    ],
    ids=["lacks-a-feature", "has-another-feature"],
)
def test_feature_columns_that_differ_from_scaler_json_are_refused(
    tmp_path: Path, edit: Callable[[dict[str, Any]], object], message: str
) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))
    state = read(directory, "scaler.json")
    edit(state)
    write(directory, "scaler.json", state)
    loaded = Artefacts.load(directory)

    with pytest.raises(ArtefactsError, match=rf"scaler\.json: {message}"):
        loaded.transform(flow_table(seed=1))


@pytest.mark.parametrize("version", [2, 0, "1", 1.0, True, None])
def test_a_format_version_other_than_1_is_refused(
    tmp_path: Path, version: object
) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))
    manifest = read(directory, "manifest.json")
    manifest["format_version"] = version
    write(directory, "manifest.json", manifest)

    with pytest.raises(ArtefactsError, match="format_version"):
        Artefacts.load(directory)


def test_a_missing_folder_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArtefactsError, match="does not exist"):
        Artefacts.load(tmp_path / "absent")


@pytest.mark.parametrize(
    ("file", "stage"),
    [
        ("manifest.json", None),
        ("fill.json", None),
        ("scaler.json", None),
        ("autoencoder.pt", QUICK),
    ],
)
def test_a_missing_file_is_refused(
    tmp_path: Path, file: str, stage: autoencoder.Settings | None
) -> None:
    directory = saved(
        tmp_path, fit_preparation(flow_table(), settings(autoencoder=stage))
    )
    (directory / file).unlink()

    with pytest.raises(ArtefactsError, match=rf"has no {file}"):
        Artefacts.load(directory)


@pytest.mark.parametrize("file", ["manifest.json", "fill.json", "scaler.json"])
@pytest.mark.parametrize(
    "content",
    [b"{not json", b"\xff\xfe\x00", b'{"duration": NaN}'],
    ids=["text", "bytes", "nan"],
)
def test_a_file_that_is_not_valid_json_is_refused(
    tmp_path: Path, file: str, content: bytes
) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))
    (directory / file).write_bytes(content)

    with pytest.raises(
        ArtefactsError, match=rf"{file} is malformed: it is not valid JSON"
    ):
        Artefacts.load(directory)


@pytest.mark.parametrize(
    ("file", "path", "literal"),
    [
        ("scaler.json", ["duration", "mean"], "1e999"),
        ("scaler.json", ["duration", "min"], "-1e999"),
        ("scaler.json", ["duration", "std"], "1" + "0" * 400),
        ("fill.json", ["duration", "max"], "1e999"),
        ("manifest.json", ["flow_settings", "idle_timeout"], "1e999"),
    ],
    ids=["mean", "negative", "integer", "fill", "timeout"],
)
def test_a_number_out_of_the_float_range_is_refused(
    tmp_path: Path, file: str, path: list[str], literal: str
) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))
    state = read(directory, file)
    set_key(path, "OUT_OF_RANGE")(state)
    content = json.dumps(state).replace('"OUT_OF_RANGE"', literal)
    (directory / file).write_text(content, encoding="utf-8")

    with pytest.raises(
        ArtefactsError,
        match=rf"{file} is malformed: the number {literal[:20]}.* is out of the float range.*netgen fit",
    ):
        Artefacts.load(directory)


def test_a_file_that_cannot_be_read_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))
    read_text = Path.read_text

    def refuse(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.name == "scaler.json":
            raise PermissionError(13, "Permission denied")
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)

    with pytest.raises(
        ArtefactsError, match=r"cannot read scaler\.json: Permission denied"
    ):
        Artefacts.load(directory)


def set_key(path: list[str], value: object) -> Callable[[dict[str, Any]], None]:
    def edit(state: dict[str, Any]) -> None:
        for key in path[:-1]:
            state = state[key]
        state[path[-1]] = value

    return edit


def drop_key(path: list[str]) -> Callable[[dict[str, Any]], None]:
    def edit(state: dict[str, Any]) -> None:
        for key in path[:-1]:
            state = state[key]
        del state[path[-1]]

    return edit


MALFORMED = [
    (
        "fill.json",
        drop_key(["duration", "max"]),
        "the entry 'duration' lacks the key 'max'",
    ),
    (
        "fill.json",
        set_key(["duration", "median"], "1.5"),
        "'median' is '1.5', not a number",
    ),
    ("fill.json", set_key(["duration", "max"], True), "'max' is True, not a number"),
    (
        "fill.json",
        set_key(["duration"], 3),
        "the entry 'duration' is not a JSON object",
    ),
    (
        "scaler.json",
        drop_key(["duration", "std"]),
        "the entry 'duration' lacks the key 'std'",
    ),
    (
        "scaler.json",
        set_key(["duration", "log"], "false"),
        "'log' is 'false', not true or false",
    ),
    (
        "scaler.json",
        set_key(["duration", "constant"], 0),
        "'constant' is 0, not true or false",
    ),
    (
        "scaler.json",
        set_key(["duration", "mean"], None),
        "'mean' is None, not a number",
    ),
    (
        "manifest.json",
        set_key(["columns", "duration"], "featuer"),
        "the column 'duration' has the role 'featuer'",
    ),
    (
        "manifest.json",
        set_key(["columns", "duration"], 5),
        "the column 'duration' has the role 5",
    ),
    (
        "manifest.json",
        set_key(["columns"], ["duration"]),
        "'columns' is \\['duration'\\], not a JSON object",
    ),
    (
        "manifest.json",
        drop_key(["flow_settings", "idle_timeout"]),
        "flow_settings lacks the key 'idle_timeout'",
    ),
    (
        "manifest.json",
        set_key(["flow_settings", "idle_timeout"], "120"),
        "'idle_timeout' is '120', not a number",
    ),
    (
        "manifest.json",
        set_key(["flow_settings", "idle_timeout"], True),
        "'idle_timeout' is True, not a number",
    ),
    (
        "manifest.json",
        set_key(["flow_settings", "idle_timeout"], 0),
        "idle_timeout is 0 seconds",
    ),
    (
        "manifest.json",
        set_key(["flow_settings", "idle_timeout"], -1),
        "idle_timeout is -1 seconds",
    ),
    (
        "manifest.json",
        set_key(["flow_settings", "active_timeout"], False),
        "'active_timeout' is False, not a number or null",
    ),
    (
        "manifest.json",
        set_key(["flow_settings", "active_timeout"], 0),
        "active_timeout is 0 seconds",
    ),
    ("manifest.json", set_key(["seed"], "42"), "'seed' is '42', not an integer"),
    (
        "manifest.json",
        set_key(["fingerprint"], None),
        "'fingerprint' is None, not a string",
    ),
    (
        "manifest.json",
        set_key(["label_column"], 3),
        "'label_column' is 3, not a string or null",
    ),
    (
        "manifest.json",
        set_key(["autoencoder"], "off"),
        "'autoencoder' is 'off', not a JSON object or null",
    ),
]


@pytest.mark.parametrize(("file", "edit", "problem"), MALFORMED)
def test_malformed_artefacts_are_refused_naming_the_file_and_the_fix(
    tmp_path: Path, file: str, edit: Callable[[dict[str, Any]], None], problem: str
) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))
    state = read(directory, file)
    edit(state)
    write(directory, file, state)

    with pytest.raises(
        ArtefactsError, match=rf"{file} is malformed: .*{problem}.*netgen fit"
    ):
        Artefacts.load(directory)


@pytest.mark.parametrize("file", ["manifest.json", "fill.json", "scaler.json"])
def test_a_file_that_is_not_a_json_object_is_refused(tmp_path: Path, file: str) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))
    write(directory, file, ["not", "an", "object"])

    with pytest.raises(
        ArtefactsError, match=rf"{file} is malformed: the file is not a JSON object"
    ):
        Artefacts.load(directory)


def test_a_loaded_role_is_the_role_member_not_its_string(tmp_path: Path) -> None:
    directory = saved(tmp_path, fit_preparation(flow_table(), settings()))

    roles = Artefacts.load(directory).roles

    assert all(isinstance(role, Role) for role in roles.values())
    assert roles["duration"] is Role.FEATURE
