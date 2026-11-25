"""The preparation: the fixed order of the learning stages, and the artefacts folder.

The order is fill step, scaler, then the autoencoder when its settings are
given. It is not configurable, because another order can leak test rows into
fitting. Each stage fits on the training table after the stages before it
have transformed that table.

The artefacts folder holds `manifest.json` and one file per stage.
`Artefacts.save` writes into the folder it is given; the atomic renaming
belongs to the caller (`netgen.output`).
"""

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import netgen
from netgen.errors import ArtefactsError, NetGenError
from netgen.record import FlowSettings, Role
from netgen.stages import (
    INTEGER,
    NUMBER,
    OBJECT,
    TEXT,
    Fitted,
    autoencoder,
    check_columns,
    fill,
    json_field,
    json_object,
    malformed,
    read_json,
    scaler,
)
from netgen.stages.autoencoder import FittedAutoencoder
from netgen.stages.autoencoder import Settings as AutoencoderSettings
from netgen.stages.fill import FittedFill
from netgen.stages.scaler import FittedScaler
from netgen.table import Count, FlowTable, TableResult, check_record

MANIFEST = "manifest.json"
FORMAT_VERSION = 1
_NEEDED = (Role.IDENTITY, Role.FEATURE)


@dataclass(frozen=True)
class Settings:
    """What a fit is given and the manifest records.

    `fingerprint` is the SHA-256 hex digest of the input file (see
    `fingerprint`). `seed` is the one seed of the run: it replaces the seed of
    the autoencoder settings.
    """

    fingerprint: str
    seed: int = 42
    flow_settings: FlowSettings = field(default_factory=FlowSettings)
    label_column: str | None = None
    autoencoder: AutoencoderSettings | None = None
    dst_port_as_feature: bool = False

    def __post_init__(self) -> None:
        if self.autoencoder is not None and self.autoencoder.seed != self.seed:
            object.__setattr__(
                self,
                "autoencoder",
                dataclasses.replace(self.autoencoder, seed=self.seed),
            )


def fingerprint(path: Path) -> str:
    """The SHA-256 hex digest of the bytes of the file at `path`."""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


@dataclass(frozen=True)
class Artefacts:
    """The fitted stages, the roles in force and the settings of a fit.

    `fit_counts` and `fit_warnings` are the observations of the fit, for the
    report. They are never saved: a loaded `Artefacts` has none.
    """

    settings: Settings
    roles: Mapping[str, Role]
    fill: FittedFill
    scaler: FittedScaler
    autoencoder: FittedAutoencoder | None = None
    fit_counts: Mapping[str, Count] = field(
        default_factory=lambda: MappingProxyType({})
    )
    fit_warnings: tuple[str, ...] = ()

    def stages(self) -> tuple[tuple[str, str, Fitted], ...]:
        """The fitted stages in order, each with its name and its file."""
        stages: list[tuple[str, str, Fitted]] = [
            ("fill", fill.FILE, self.fill),
            ("scaler", scaler.FILE, self.scaler),
        ]
        if self.autoencoder is not None:
            stages.append(("autoencoder", autoencoder.FILE, self.autoencoder))
        return tuple(stages)

    def transform(self, table: FlowTable) -> TableResult:
        """The table after every stage, with the counts and warnings of all stages.

        The roles in force are the manifest's, whatever roles `table` carries.
        Refuses with `ArtefactsError` a table that lacks a column the
        artefacts need, or whose feature columns differ from `scaler.json`.
        Keeps a label column of `table`. Drops the other columns the artefacts
        do not know, with a warning.
        """
        current, warnings = self._recast(table)
        check_record(current)
        counts: dict[str, Count] = {}
        for _, _, fitted in self.stages():
            result = fitted.transform(current)
            current = result.table
            counts.update(result.counts)
            warnings += result.warnings
        return TableResult(current, MappingProxyType(counts), warnings)

    def _recast(self, table: FlowTable) -> tuple[FlowTable, tuple[str, ...]]:
        """`table` with the manifest's roles, and the warning for the columns it drops."""
        frame = table.to_frame()
        columns = [str(name) for name in frame.columns]
        missing = [
            name
            for name, role in self.roles.items()
            if role in _NEEDED and name not in columns
        ]
        if missing:
            raise ArtefactsError(
                f"The input lacks the columns {missing}, which the artefacts need. "
                "Transform an input that holds every column the artefacts were fitted on."
            )
        # A label is carried through untouched, also when the fit input had none.
        roles = {
            **self.roles,
            **{
                name: Role.LABEL
                for name in columns
                if name not in self.roles and table.roles[name] is Role.LABEL
            },
        }
        extra = [name for name in columns if name not in roles]
        kept = [name for name in roles if name in columns]
        recast = FlowTable(frame[kept], {name: roles[name] for name in kept})
        # An extra input feature is not dropped: it would change the feature set silently.
        features = list(recast.features().columns) + [
            name for name in extra if table.roles[name] is Role.FEATURE
        ]
        check_columns(features, self.scaler.values, scaler.FILE)
        warnings = (
            (
                f"The input columns {extra} are not columns of the artefacts and are dropped.",
            )
            if extra
            else ()
        )
        return recast, warnings

    def manifest(self) -> dict[str, Any]:
        """The content of `manifest.json`. Report numbers never go into it."""
        settings = self.settings
        return {
            "netgen_version": netgen.__version__,
            "format_version": FORMAT_VERSION,
            "columns": {name: role.value for name, role in self.roles.items()},
            "features": [
                {"name": name, "log": learned.log}
                for name, learned in self.scaler.values.items()
            ],
            "label_column": settings.label_column,
            "fingerprint": settings.fingerprint,
            "seed": settings.seed,
            "flow_settings": {
                "idle_timeout": settings.flow_settings.idle_timeout,
                "active_timeout": settings.flow_settings.active_timeout,
            },
            "autoencoder": None
            if settings.autoencoder is None
            else settings.autoencoder.mapping(),
            "stages": [{"name": name, "file": file} for name, file, _ in self.stages()],
        }

    def save(self, directory: Path) -> None:
        """Write each stage's file and `manifest.json` into the existing folder `directory`."""
        for _, _, fitted in self.stages():
            fitted.save(directory)
        (directory / MANIFEST).write_text(
            json.dumps(self.manifest(), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "Artefacts":
        """Read an artefacts folder. Refuses a missing or malformed folder with `ArtefactsError`."""
        if not directory.is_dir():
            raise ArtefactsError(
                f"The artefacts folder {directory} does not exist. "
                "Give the folder `netgen fit` wrote, or run `netgen fit` to write it."
            )
        manifest = json_object(read_json(directory, MANIFEST), MANIFEST, "the file")
        version = manifest.get("format_version")
        if type(version) is not int or version != FORMAT_VERSION:
            raise ArtefactsError(
                f"The artefacts in {directory} have format_version {version!r}; this NetGen reads "
                f"format_version {FORMAT_VERSION} only. Run `netgen fit` to write the artefacts again."
            )
        roles = _roles(manifest)
        stage = json_field(
            manifest, "autoencoder", OBJECT.or_null(), MANIFEST, "the manifest"
        )
        settings = Settings(
            fingerprint=json_field(
                manifest, "fingerprint", TEXT, MANIFEST, "the manifest"
            ),
            seed=json_field(manifest, "seed", INTEGER, MANIFEST, "the manifest"),
            flow_settings=_flow_settings(manifest),
            label_column=json_field(
                manifest, "label_column", TEXT.or_null(), MANIFEST, "the manifest"
            ),
            autoencoder=None
            if stage is None
            else autoencoder.Settings.from_mapping(stage),
            dst_port_as_feature=roles.get("dst_port") is Role.FEATURE,
        )
        return cls(
            settings,
            roles,
            fill.load(directory, manifest),
            scaler.load(directory, manifest),
            None if stage is None else autoencoder.load(directory, manifest),
        )


def fit_preparation(table: FlowTable, settings: Settings) -> Artefacts:
    """Fit the stages in order on the training table `table`.

    `settings.dst_port_as_feature` decides the role of `dst_port`. Refuses
    with `InputError` a table that does not meet the flow record.
    """
    check_record(table)
    roles = {
        **table.roles,
        "dst_port": Role.FEATURE if settings.dst_port_as_feature else Role.IDENTITY,
    }
    training = FlowTable(table.to_frame(), roles)
    fitted_fill = fill.fit(training)
    filled = fitted_fill.transform(training)
    fitted_scaler = scaler.fit(filled.table)
    scaled = fitted_scaler.transform(filled.table)
    fitted_autoencoder = (
        None
        if settings.autoencoder is None
        else autoencoder.fit(scaled.table, settings.autoencoder)
    )
    return Artefacts(
        settings,
        MappingProxyType(roles),
        fitted_fill,
        fitted_scaler,
        fitted_autoencoder,
        MappingProxyType({**filled.counts, **scaled.counts}),
        filled.warnings + scaled.warnings,
    )


def _roles(manifest: Mapping[str, Any]) -> Mapping[str, Role]:
    """The manifest's roles as `Role` members, so a plain string never passes as a role."""
    columns = json_field(manifest, "columns", OBJECT, MANIFEST, "the manifest")
    roles: dict[str, Role] = {}
    for name, value in columns.items():
        try:
            roles[name] = Role(value)
        except ValueError as error:
            raise malformed(
                MANIFEST,
                f"the column {name!r} has the role {value!r}; the roles are {[role.value for role in Role]}",
            ) from error
    return MappingProxyType(roles)


def _flow_settings(manifest: Mapping[str, Any]) -> FlowSettings:
    """The manifest's flow settings, refused when a timeout breaks its rules."""
    where = "flow_settings"
    entry = json_field(manifest, where, OBJECT, MANIFEST, "the manifest")
    idle = json_field(entry, "idle_timeout", NUMBER, MANIFEST, where)
    active = json_field(entry, "active_timeout", NUMBER.or_null(), MANIFEST, where)
    try:
        return FlowSettings(idle, active)
    except NetGenError as error:
        raise malformed(MANIFEST, str(error).rstrip(".")) from error
