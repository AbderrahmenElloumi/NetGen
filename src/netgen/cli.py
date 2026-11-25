"""The command line: the `fit`, `transform` and `build` commands.

The commands are thin. The readers, `netgen.table`, `netgen.preparation`,
`netgen.output`, `netgen.report` and `netgen.benchmark` do the work. This
module is the only place that configures logging and warning filters, and it
does so in `main` only.
"""

import argparse
import logging
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from netgen import benchmark, report
from netgen.errors import InputError, NetGenError
from netgen.output import write_directory, write_files
from netgen.preparation import Artefacts, Settings, fingerprint, fit_preparation
from netgen.readers import Reader, detect, reader_named
from netgen.record import FlowSettings
from netgen.stages import autoencoder
from netgen.table import Count, FlowTable, TableResult, stratified_split

DEFAULT_LABEL_COL = "Label"

# A capture's start times are UTC; a flow export's are local times.
START_TIME_ZONES: dict[str, report.StartTimeZone] = {
    "capture": "utc",
    "flow-export": "local",
}

REPORT = "report.json"
ARTEFACTS = "artefacts"


def build_parser() -> argparse.ArgumentParser:
    """The parser of every command and option."""
    parser = argparse.ArgumentParser(
        prog="netgen",
        description="Turn network traffic recordings into tabular datasets.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser(
        "fit", help="learn the preparation from INPUT and save the artefacts"
    )
    _add_input(fit)
    fit.add_argument(
        "--artefacts",
        type=Path,
        required=True,
        metavar="DIR",
        help="artefacts folder to write",
    )
    _add_common(fit)
    _add_fit_options(fit)
    fit.set_defaults(run=_fit)

    transform = commands.add_parser(
        "transform", help="apply saved artefacts to INPUT and write the dataset"
    )
    _add_input(transform)
    transform.add_argument(
        "--artefacts",
        type=Path,
        required=True,
        metavar="DIR",
        help="artefacts folder to read",
    )
    transform.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="FILE",
        help="dataset file to write; the report goes beside it",
    )
    _add_common(transform)
    _add_output_format(transform)
    transform.set_defaults(run=_transform)

    build = commands.add_parser(
        "build",
        help="split INPUT, fit on the training split and transform both splits",
    )
    _add_input(build)
    build.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="DIR",
        help="folder to write: the artefacts, the two splits and the report",
    )
    _add_common(build)
    _add_fit_options(build)
    _add_output_format(build)
    build.add_argument(
        "--benchmark",
        action="store_true",
        help="run the Random Forest benchmark (needs the benchmark extra)",
    )
    build.set_defaults(run=_build)
    return parser


def _add_fit_options(parser: argparse.ArgumentParser) -> None:
    """The options of `fit` and `build`: the label column, flow settings, seed and autoencoder."""
    parser.add_argument(
        "--label-col",
        default=DEFAULT_LABEL_COL,
        metavar="NAME",
        help="input column the flow-export reader maps to the label "
        f"(default: {DEFAULT_LABEL_COL})",
    )
    parser.add_argument(
        "--idle-timeout",
        type=_positive_float,
        default=120.0,
        metavar="SECONDS",
        help="idle timeout of flow formation (default: 120)",
    )
    parser.add_argument(
        "--active-timeout",
        type=_positive_float,
        default=None,
        metavar="SECONDS",
        help="active timeout of flow formation (default: off)",
    )
    parser.add_argument(
        "--dst-port-as-feature",
        action="store_true",
        help="promote the destination port from identity to feature",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="one seed for the split and the autoencoder (default: 42)",
    )
    parser.add_argument(
        "--autoencoder",
        action="store_true",
        help="turn the autoencoder stage on",
    )
    parser.add_argument(
        "--latents",
        action="store_true",
        help="also write latent columns (requires --autoencoder)",
    )
    parser.add_argument(
        "--latent-dim",
        type=_positive_int,
        default=8,
        metavar="N",
        help="latent size (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=50,
        metavar="N",
        help="training epochs (default: 50)",
    )
    parser.add_argument(
        "--fit-on-label",
        default=None,
        metavar="VALUE",
        help="fit the autoencoder only on training flows carrying this label",
    )


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-format",
        choices=("parquet", "csv"),
        default="parquet",
        help="dataset file format (default: parquet)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code: 0, or 2 for a user error."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    warnings.simplefilter("default")
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except NetGenError as error:
        print(
            "\n".join([f"netgen: error: {error}", *getattr(error, "__notes__", ())]),
            file=sys.stderr,
        )
        return 2
    return 0


def _fit(args: argparse.Namespace) -> None:
    stage = _autoencoder(args)
    flow_settings = FlowSettings(args.idle_timeout, args.active_timeout)
    with write_directory(args.artefacts, force=args.force) as folder:
        reader = _reader(args)
        result = reader.read(args.input, flow_settings, label_col=args.label_col)
        settings = _settings(args, flow_settings, stage, fingerprint(args.input))
        artefacts = fit_preparation(result.table, settings)
        prepared = artefacts.transform(result.table)
        artefacts.save(folder)
        counts = {**result.counts, **artefacts.fit_counts, **prepared.counts}
        entry = _report(
            "fit",
            settings.fingerprint,
            reader,
            artefacts,
            {"train": _entry(artefacts, prepared, counts)},
        )
        report.write(entry, folder / REPORT)
    warned = [*result.warnings, *artefacts.fit_warnings, *prepared.warnings]
    print(report.summary(entry, warned))
    print(f"Artefacts written to {args.artefacts}")


def _transform(args: argparse.Namespace) -> None:
    report_path = args.out.with_name(f"{args.out.stem}.report.json")
    with write_files(args.out, report_path, force=args.force) as (dataset, written):
        artefacts = Artefacts.load(args.artefacts)
        reader = _reader(args)
        result = reader.read(
            args.input,
            artefacts.settings.flow_settings,
            label_col=artefacts.settings.label_column or DEFAULT_LABEL_COL,
        )
        prepared = artefacts.transform(result.table)
        _write_dataset(prepared.table, dataset, args.output_format)
        counts = {**result.counts, **prepared.counts}
        entry = _report(
            "transform",
            fingerprint(args.input),
            reader,
            artefacts,
            {"transform": _entry(artefacts, prepared, counts)},
        )
        report.write(entry, written)
    print(report.summary(entry, [*result.warnings, *prepared.warnings]))
    print(f"Dataset written to {args.out}, report to {report_path}")


def _build(args: argparse.Namespace) -> None:
    stage = _autoencoder(args)
    flow_settings = FlowSettings(args.idle_timeout, args.active_timeout)
    with write_directory(args.out, force=args.force) as folder:
        reader = _reader(args)
        result = reader.read(args.input, flow_settings, label_col=args.label_col)
        if args.benchmark:
            # Every label group keeps at least one row in the training split,
            # so the classes of the whole table are those of the training split.
            benchmark.preflight(result.table)
        train, test = stratified_split(result.table, seed=args.seed)
        if len(test.to_frame()) == 0:
            raise InputError(
                "The test split would be empty: a label group gives the test split "
                "a flow only when it holds 2 flows or more, and no group of the "
                f"input's {len(train.to_frame())} flow(s) does. "
                "Give an input that forms at least 2 flows with the same label."
            )
        settings = _settings(args, flow_settings, stage, fingerprint(args.input))
        artefacts = fit_preparation(train, settings)
        prepared_train = artefacts.transform(train)
        prepared_test = artefacts.transform(test)
        train_entry = _entry(
            artefacts,
            prepared_train,
            {**result.counts, **artefacts.fit_counts, **prepared_train.counts},
        )
        test_entry = _entry(
            artefacts, prepared_test, {**result.counts, **prepared_test.counts}
        )
        scores = (
            benchmark.run(prepared_train.table, prepared_test.table, seed=args.seed)
            if args.benchmark
            else None
        )
        saved = folder / ARTEFACTS
        saved.mkdir()
        artefacts.save(saved)
        report.write(
            _report(
                "build", settings.fingerprint, reader, artefacts, {"train": train_entry}
            ),
            saved / REPORT,
        )
        entry = _report(
            "build",
            settings.fingerprint,
            reader,
            artefacts,
            {"train": train_entry, "test": test_entry},
            scores,
        )
        report.write(entry, folder / REPORT)
        for name, prepared in (("train", prepared_train), ("test", prepared_test)):
            dataset = folder / f"{name}.{args.output_format}"
            _write_dataset(prepared.table, dataset, args.output_format)
    warned = [
        *result.warnings,
        *artefacts.fit_warnings,
        *prepared_train.warnings,
        *prepared_test.warnings,
    ]
    print(report.summary(entry, list(dict.fromkeys(warned))))
    if scores is not None:
        for name, scored in scores["runs"].items():
            print(
                f"benchmark {name}: macro F1 {scored['model']['macro_f1']:.3f}, "
                f"baseline macro F1 {scored['baseline']['macro_f1']:.3f}"
            )
    print(f"Build written to {args.out}")


def _autoencoder(args: argparse.Namespace) -> autoencoder.Settings | None:
    """The autoencoder settings of `fit` or `build`, or `None` without `--autoencoder`."""
    if args.latents and not args.autoencoder:
        raise NetGenError(
            "--latents requires --autoencoder. Add --autoencoder, or drop --latents."
        )
    if args.fit_on_label is not None and not args.autoencoder:
        raise NetGenError(
            "--fit-on-label requires --autoencoder. "
            "Add --autoencoder, or drop --fit-on-label."
        )
    if not args.autoencoder:
        return None
    return autoencoder.Settings(
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        seed=args.seed,
        latents=args.latents,
        fit_on_label=args.fit_on_label,
    )


def _settings(
    args: argparse.Namespace,
    flow_settings: FlowSettings,
    stage: autoencoder.Settings | None,
    digest: str,
) -> Settings:
    return Settings(
        fingerprint=digest,
        seed=args.seed,
        flow_settings=flow_settings,
        label_column=args.label_col,
        autoencoder=stage,
        dst_port_as_feature=args.dst_port_as_feature,
    )


def _write_dataset(table: FlowTable, path: Path, output_format: str) -> None:
    frame = table.to_frame()
    if output_format == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, engine="pyarrow", index=False)


def _reader(args: argparse.Namespace) -> Reader:
    if args.input_format is not None:
        return reader_named(args.input_format)
    return detect(args.input)


def _entry(
    artefacts: Artefacts, prepared: TableResult, counts: Mapping[str, Count]
) -> dict[str, Any]:
    """The report entry of one split: its prepared table and its counts."""
    stage = artefacts.settings.autoencoder
    return report.split(
        prepared.table,
        counts,
        normal_label=stage.fit_on_label if stage is not None else None,
    )


def _report(
    command: report.Command,
    digest: str,
    reader: Reader,
    artefacts: Artefacts,
    splits: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The report of one command: the split entries under its header, and the benchmark."""
    return report.build(
        command=command,
        fingerprint=digest,
        seed=artefacts.settings.seed,
        start_time_zone=START_TIME_ZONES[reader.name],
        splits=splits,
        benchmark=scores,
    )


def _add_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input", type=Path, metavar="INPUT", help="capture or flow export to read"
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force", action="store_true", help="replace an existing target"
    )
    parser.add_argument(
        "--input-format",
        default=None,
        metavar="NAME",
        help="select the reader by name (capture or flow-export) and skip detection",
    )


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive number")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer of at least 1")
    return value
