# NetGen usage

NetGen turns network-traffic recordings into tabular datasets for machine learning. This page describes how to run it: the three commands, their options, the files they write and the report. The terms follow [CONTEXT.md](../CONTEXT.md). The design, the flow record and the reasons behind it are in [architecture.md](architecture.md).

## Running a command

NetGen has three commands: `fit`, `transform` and `build`. This page writes each command as `netgen <command> ...`. From a clone, the same command runs as `python -m netgen <command> ...` inside the `dev` container service, from the repository root. A path in the command is relative to the repository root.

| Command | Purpose | Writes |
| --- | --- | --- |
| `netgen fit INPUT --artefacts DIR` | Learn the preparation from INPUT and save the artefacts. | The artefacts folder. No dataset. |
| `netgen transform INPUT --artefacts DIR --out FILE` | Apply saved artefacts to INPUT. | The dataset file and its report. |
| `netgen build INPUT --out DIR` | Split INPUT, fit on the training split, transform both splits. | The artefacts, the two splits and the report. |

`netgen --help` and `netgen <command> --help` print the options.

Every command writes its output all or nothing: a failed run leaves no partial output. A command refuses an existing target unless `--force` is given, and with `--force` it replaces the target only when the new output is complete. The target is the artefacts folder for `fit`, the dataset file and its report for `transform`, and the output folder for `build`.

After a successful run the command prints a console summary on standard output: per split the rows, the labels and the totals of the counts, then each warning, then the paths written. Warnings do not stop a run.

## Inputs

NetGen reads two inputs.

| Input | Reader name | Recognised by |
| --- | --- | --- |
| A capture: pcap or pcapng, IPv4 and IPv6 | `capture` | The first four bytes of the file: a pcap magic of either byte order, microsecond or nanosecond, or the pcapng section header. |
| A flow export: a CICFlowMeter CSV in the layout of the CIC-IDS2017 data | `flow-export` | The first line: a CSV header that holds `Flow Duration`, `Total Fwd Packets`, `Total Backward Packets`, `Fwd Packet Length Max` and `Flow IAT Mean`. Spaces around the header names are ignored. |

Detection asks the capture reader first, then the flow-export reader. An input that is neither fails with an error that names both formats. `--input-format` names the reader and skips detection; the named reader then gives its own error for a file it cannot read.

A capture is unlabelled. NetGen forms its flows with the flow settings (below), and the start times are UTC. A flow export already holds flows, so the flow settings do not apply to it; its start times are the local times as written in the file, read day first, with or without seconds. A flow-export column that the flow record does not hold is dropped with a warning. A flow export is read as UTF-8 or, if that fails, as cp1252; any other encoding fails.

The reader fills the fields of the flow record, which are the same for both inputs. A capture and a flow export give the same dataset columns, so artefacts fitted on one can transform the other.

## Flow settings and role options

The flow settings and the role options belong to `fit` and `build`. They decide how flows form and what each column is for, so the artefacts record them in the manifest. `transform` reads them from the manifest and accepts no override. To change them, fit again.

| Setting | Option | Effect |
| --- | --- | --- |
| Idle timeout | `--idle-timeout` | Seconds a flow stays silent before it ends. Capture only. |
| Active timeout | `--active-timeout` | Seconds a flow lasts before it is cut into a new flow. Capture only. |
| Label column | `--label-col` | The input column that the flow-export reader maps to the label. A capture ignores it. |
| Destination port role | `--dst-port-as-feature` | Promotes the destination port from identity column to feature. |

A flow export ignores the timeouts, and the manifest still records them.

**Promoted destination port.** A promoted destination port is scaled linearly. This is crude: port numbers are categories, not quantities, so their order and distance carry no meaning. `fit` and `build` print a warning when the option is on.

## `netgen fit`

```bash
netgen fit INPUT --artefacts DIR [options]
```

Learns the preparation from all flows of INPUT and writes the artefacts folder `DIR`. It writes no dataset.

| Option | Meaning | Default |
| --- | --- | --- |
| `--artefacts DIR` | Artefacts folder to write. Required. | none |
| `--force` | Replace an existing target. | off |
| `--input-format NAME` | Select the reader by name, `capture` or `flow-export`, and skip detection. | detect |
| `--label-col NAME` | Input column that the flow-export reader maps to the label. | `Label` |
| `--idle-timeout SECONDS` | Idle timeout of flow formation. A positive number. | 120 |
| `--active-timeout SECONDS` | Active timeout of flow formation. A positive number. | off |
| `--dst-port-as-feature` | Promote the destination port from identity to feature. | off |
| `--seed N` | One seed for the split and the autoencoder. An integer. | 42 |
| `--autoencoder` | Turn the autoencoder stage on. | off |
| `--latents` | Also write latent columns. Requires `--autoencoder`. | off |
| `--latent-dim N` | Latent size. An integer of at least 1. | 8 |
| `--epochs N` | Training epochs. An integer of at least 1. | 50 |
| `--fit-on-label VALUE` | Fit the autoencoder only on training flows that carry this label. The scaler still fits on every training flow. Requires `--autoencoder`. | all flows |

Output: the artefacts folder `DIR`, described below. The `fit` report in `DIR` holds one split, `train`.

## `netgen transform`

```bash
netgen transform INPUT --artefacts DIR --out FILE [options]
```

Applies the artefacts in `DIR` to INPUT and writes the dataset. This is the only command that makes a dataset from existing artefacts.

| Option | Meaning | Default |
| --- | --- | --- |
| `--artefacts DIR` | Artefacts folder to read. Required. | none |
| `--out FILE` | Dataset file to write. The report goes beside it. Required. | none |
| `--force` | Replace an existing dataset file or report. | off |
| `--input-format NAME` | Select the reader by name, `capture` or `flow-export`, and skip detection. | detect |
| `--output-format {parquet,csv}` | Format of the dataset file. | `parquet` |

`transform` takes the flow settings, the column roles, the label column and the autoencoder settings, latent columns included, from the manifest.

Output: `FILE` and `<FILE stem>.report.json` in the same folder, so `out/flows.parquet` gives `out/flows.report.json`. The format comes from `--output-format` only, never from the file name. The report holds one split, `transform`. The command carries the label of INPUT when it has one, also when the fit input had none.

`transform` refuses:

- an input that lacks a column the artefacts need;
- artefacts whose `format_version` is not 1;
- feature columns that differ from those in `scaler.json`;
- a missing or malformed artefacts folder.

It drops any other input column that the artefacts do not know and warns, listing the columns.

## `netgen build`

```bash
netgen build INPUT --out DIR [options]
```

Reads INPUT once, splits the flows into a training split and a test split, fits the preparation on the training split only, then transforms both splits with it.

| Option | Meaning | Default |
| --- | --- | --- |
| `--out DIR` | Folder to write: the artefacts, the two splits and the report. Required. | none |
| `--force` | Replace an existing target. | off |
| `--input-format NAME` | Select the reader by name, `capture` or `flow-export`, and skip detection. | detect |
| `--label-col NAME` | Input column that the flow-export reader maps to the label. | `Label` |
| `--idle-timeout SECONDS` | Idle timeout of flow formation. A positive number. | 120 |
| `--active-timeout SECONDS` | Active timeout of flow formation. A positive number. | off |
| `--dst-port-as-feature` | Promote the destination port from identity to feature. | off |
| `--seed N` | One seed for the split and the autoencoder. An integer. | 42 |
| `--autoencoder` | Turn the autoencoder stage on. | off |
| `--latents` | Also write latent columns. Requires `--autoencoder`. | off |
| `--latent-dim N` | Latent size. An integer of at least 1. | 8 |
| `--epochs N` | Training epochs. An integer of at least 1. | 50 |
| `--fit-on-label VALUE` | Fit the autoencoder only on training flows that carry this label. The scaler still fits on every training flow. Requires `--autoencoder`. | all flows |
| `--output-format {parquet,csv}` | Format of the two dataset files. | `parquet` |
| `--benchmark` | Run the Random Forest benchmark. Needs the `benchmark` extra. | off |

`build` has no `--artefacts` option: it writes the artefacts into `DIR/artefacts`.

**Split.** The split is random, 80/20 and stratified by label, with `--seed`. Flows without a label form one group. A group of `n` flows sends `(n + 2) // 5` flows to the test split, and at least 1 when `n` is 2 or more. A group of one flow stays in the training split. The split happens before anything is fitted.

Output, with `.csv` instead of `.parquet` under `--output-format csv`:

| Path | Content |
| --- | --- |
| `DIR/artefacts/` | The artefacts of the fit on the training split, with the report of the training split. |
| `DIR/train.parquet` | The training split, transformed. |
| `DIR/test.parquet` | The test split, transformed. |
| `DIR/report.json` | The report of both splits, and the benchmark when it ran. |

`build` refuses an input whose test split would be empty, that is an input in which every label group holds one flow. It also refuses everything that `fit` refuses.

## Exit codes and errors

| Code | Meaning |
| --- | --- |
| 0 | Success. |
| 2 | A user error: NetGen raises a `NetGenError`. |
| 2 | A usage error: an unknown option or a bad value. |
| other | Any other exception is a bug. It keeps its traceback and the interpreter's exit code. |

Every user error is a `NetGenError` or a subclass: `InputError`, `ArtefactsError`, `AutoencoderUnavailable` and `BenchmarkUnavailable`. The command catches it, prints `netgen: error: <message>` on standard error, prints any note on the following lines, writes no traceback and exits with 2. A message names the cause and the fix. A usage error prints the usage line and `netgen <command>: error: <message>`. `--idle-timeout` and `--active-timeout` must be positive finite numbers, `--latent-dim` and `--epochs` integers of at least 1, and `--seed` an integer; the parser refuses any other value with exit code 2.

A user error leaves nothing written. The commands refuse, with exit code 2:

- an existing target without `--force`;
- a missing input, an input that no reader recognises, and an unknown `--input-format` name (the message lists the valid names);
- a flow-export header that is not a CICFlowMeter header or lacks a column the flow record needs, a start time that does not parse, and a file that is neither UTF-8 nor cp1252;
- a truncated or corrupt capture;
- `--latents` without `--autoencoder`;
- `--fit-on-label` without `--autoencoder`;
- every refusal of `transform`, listed above;
- every refusal of the autoencoder and of the benchmark, described below.

## Artefacts

The artefacts folder is what one fit learned and the settings needed to repeat it. The folder holds plain JSON and, with the autoencoder, a weights-only file. NetGen reads none of it with pickle.

| File | Content |
| --- | --- |
| `manifest.json` | The settings and the structure of the fit. |
| `fill.json` | What the fill step learned. |
| `scaler.json` | What the scaler learned. |
| `autoencoder.pt` | The network weights, written with `--autoencoder` only. A `state_dict` loaded with `weights_only=True`. |
| `report.json` | The report of the fit (the training split). |

### `manifest.json`

| Key | Content |
| --- | --- |
| `netgen_version` | The NetGen version that wrote the artefacts. |
| `format_version` | The artefacts format. Always 1. |
| `columns` | Column name to role: `identity`, `label`, `feature`. |
| `features` | The feature list in order: a list of `{"name": ..., "log": true or false}`, with `log` the flag of the log step. |
| `label_column` | The value of `--label-col`. |
| `fingerprint` | The SHA-256 digest of the fit input. |
| `seed` | The seed of the run. |
| `flow_settings` | `{"idle_timeout": ..., "active_timeout": ...}`, with `active_timeout` set to `null` when off. |
| `autoencoder` | `null` without the autoencoder; otherwise the settings below. |
| `stages` | The stages in order, a list of `{"name": ..., "file": ...}`: `fill`, `scaler`, and `autoencoder` when it ran. |

The `autoencoder` settings are `latent_dim`, `epochs`, `seed`, `latents`, `fit_on_label` and the fixed hyperparameters `hidden_sizes`, `activation`, `negative_slope`, `dropout`, `batch_norm`, `optimiser`, `learning_rate`, `weight_decay`, `schedule`, `gradient_clip`, `batch_size`, `drop_last` and `min_rows`. Report numbers never go into the manifest.

### `fill.json`

One entry per feature, in feature order: the largest finite training value and the median of the finite training values. The fill step replaces an infinite value with `max` and a missing value with `median`. A feature with no finite training value gets 0 and 0, and a warning.

```json
{
  "duration": {"max": 119.8, "median": 0.02},
  "fwd_packets": {"max": 412.0, "median": 3.0}
}
```

### `scaler.json`

One entry per feature, in feature order. The statistics are those of the training column after the log step.

```json
{
  "duration": {"log": true, "mean": 0.9, "std": 1.4, "min": 0.0, "max": 4.8, "constant": false},
  "is_icmp": {"log": false, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "constant": true}
}
```

| Key | Content |
| --- | --- |
| `log` | Whether the feature takes the log step (`log1p`) before the Z-score. The flow record declares it. |
| `mean`, `std` | The training mean and the population standard deviation. |
| `min`, `max` | The training range, for counting out-of-range values. |
| `constant` | Whether the training standard deviation is below 1e-12. A constant feature is written as 0. |

The scaler clips no value. A value outside the training range passes through the formula and is counted in the report.

## Dataset

A dataset has one row per flow. The columns come in four groups, in this order, and each group follows the order of the flow record.

| Group | Columns | Content |
| --- | --- | --- |
| Identity | `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `start_time` | As read. Never scaled. A port is missing for ICMP. |
| Label | `label` | Present only when the input has labels. Carried through untouched. |
| Features | `duration`, `fwd_packets`, `bwd_packets`, `fwd_bytes`, `bwd_bytes`, `fwd_len_min`, `fwd_len_max`, `fwd_len_mean`, `fwd_len_std`, `bwd_len_min`, `bwd_len_max`, `bwd_len_mean`, `bwd_len_std`, `bytes_per_s`, `packets_per_s`, `fwd_packets_per_s`, `bwd_packets_per_s`, `iat_mean`, `iat_std`, `iat_max`, `iat_min`, `fwd_iat_total`, `bwd_iat_total`, `fwd_header_bytes`, `bwd_header_bytes`, `down_up_ratio`, `is_tcp`, `is_udp`, `is_icmp` | Filled and scaled. No missing or infinite value remains. |
| Derived | `anomaly_score`, then `latent_0` to `latent_<n-1>` | `anomaly_score` with `--autoencoder`; the latent columns also with `--latents`, with `n` the value of `--latent-dim`. Written as computed, never rescaled. |

With `--dst-port-as-feature`, `dst_port` moves from the identity group to the first place of the feature group and is scaled. The meaning and unit of each field are declared in `src/netgen/record.py`; [architecture.md](architecture.md) explains the column roles.

The dataset is Parquet by default and CSV with `--output-format csv`.

## Report

Every command writes a report in one JSON shape. It records what the command saw and did: observations only, never learned state. The file is strict JSON: a statistic that is undefined, empty or not finite is `null`, and the file never holds `NaN` or `Infinity`.

| Command | Report file | `splits` |
| --- | --- | --- |
| `fit` | `DIR/report.json` in the artefacts folder | `train` |
| `transform` | `<FILE stem>.report.json` | `transform` |
| `build` | `DIR/report.json` | `train` and `test` |
| `build` | `DIR/artefacts/report.json` | `train` only, and `benchmark` is `null` |

The report has three keys: `header`, `splits` and `benchmark`.

### Header

| Key | Content |
| --- | --- |
| `report_version` | The report format. Always 1. |
| `netgen_version` | The NetGen version. |
| `command` | `fit`, `transform` or `build`. |
| `input_fingerprint` | The SHA-256 digest of the input the command read. |
| `seed` | The seed of the run. At `transform` it is the manifest's. |
| `start_time_zone` | `utc` for a capture, `local` for a flow export. |

### Split entry

Each entry under `splits` describes the prepared table of one split, after the fill step, the scaler and the autoencoder.

| Key | Content |
| --- | --- |
| `rows` | The number of flows. |
| `labels` | Label to number of flows, sorted by label, for the labelled rows only. `{}` without a label column. |
| `duplicates` | The number of rows equal to an earlier row over the feature columns, after preparation. |
| `features` | Per feature, in record order: `mean`, `std` (population), `min`, `max`, `median` and `share_zero`, the share of rows equal to 0. |
| `counts` | The counts below. |
| `anomaly_score` | `null` without the derived column; otherwise the section below. |

### Counts

| Key | Content |
| --- | --- |
| `filled` | Per column, the values the fill step replaced: infinite and missing values together. |
| `no_finite_value` | The features with no finite training value. Filled at `fit` and `build`; empty at `transform`, because the artefacts do not store it. |
| `non_ip_dropped` | The non-IP packets a capture reader dropped. 0 for a flow export. |
| `encoding` | `utf-8` or `cp1252` for a flow export. `null` for a capture. |
| `out_of_range` | Per column, the values outside the minimum and maximum in `scaler.json`. |
| `constant_features` | The features that `scaler.json` marks constant. |
| `constant_mismatch` | Per column, the values of a constant feature that differ from the training constant. |

The per-column counts list only the columns above 0, in feature order. In a `build` report the `train` and `test` entries both repeat the reader's counts (`non_ip_dropped`, `encoding`) for the whole input.

### Anomaly score

Present when the autoencoder ran. The section holds `median`, `p90`, `p99` (linear interpolation), `max`, `median_by_label` (label to median score, `{}` when the split is unlabelled) and `auc`.

`auc` is the ROC AUC of the score as a detector of the flows that are not normal. NetGen computes it with the rank (Mann-Whitney) formula, with average ranks for ties. The label given by `--fit-on-label` is the normal class, and every other labelled row is positive. Unlabelled rows are left out. `auc` is `null` without `--fit-on-label`, without labelled rows in both groups, or when a score is not finite. At `transform` the normal label comes from the manifest.

### Benchmark

`benchmark` is `null` unless `build --benchmark` ran. Otherwise it holds:

| Key | Content |
| --- | --- |
| `settings` | `trees` 100, `class_weight` `balanced`, `seed` (the run's) and `n_jobs` 2. |
| `unlabelled_skipped` | `train` and `test`: the unlabelled rows left out of training and scoring. The datasets keep them. |
| `runs` | `features`, and `features_plus_derived` when the autoencoder ran. |

Each run holds `columns` (the columns it used, in dataset order), `model` (the Random Forest) and `baseline` (the majority class). `model` and `baseline` each hold:

| Key | Content |
| --- | --- |
| `macro_f1` | The mean F1 over the classes. |
| `balanced_accuracy` | The mean recall over the classes with test support. `null` when the test split has no labelled row. |
| `per_class` | Label to `precision`, `recall`, `f1` and `support`. A score with denominator 0 is 0. |
| `confusion_matrix` | `labels`, sorted, and `matrix`: rows are the true label, columns the predicted label. |

The label set is the union of the labelled rows of both splits. The baseline predicts the most frequent training label; a tie goes to the first label in sorted order. The benchmark has no AUC. The console prints the macro F1 of each run and of the baseline.

## The benchmark and its refusals

`--benchmark` trains a Random Forest on the transformed training split and scores it on the transformed test split. It never splits the data again. One run uses the feature columns. When the autoencoder ran, a second run adds the derived columns. The result goes into `DIR/report.json` only; it is never saved as artefacts.

The benchmark needs the `benchmark` extra, scikit-learn. The `dev` container has it; elsewhere install `pip install 'netgen[benchmark]'`. Before anything is fitted or written, `build` refuses, with exit code 2:

- scikit-learn missing (`BenchmarkUnavailable`);
- an input with no label column;
- fewer than 2 classes among the labelled training rows (the message gives the count and the classes found).

## Autoencoder

`--autoencoder` adds the optional third stage. It needs the `autoencoder` extra, torch; the `dev` container has it. The stage trains on the scaled features, on the CPU, with a fixed seed and deterministic algorithms. Its refusals have exit code 2 and happen before anything is written:

- torch missing (`AutoencoderUnavailable`, which names `pip install 'netgen[autoencoder]'`);
- fewer than 256 training rows, after `--fit-on-label` is applied;
- `--fit-on-label` given for an input with no label, or matching no training flow;
- a training loss that is not finite.

## Reproducibility

The same input, seed, locked versions and machine give an identical dataset and identical artefacts. Across machines the results agree within a tolerance and are not promised identical. One seed covers the split, the weights, the dropout and the batch order of the autoencoder; the manifest records it.

## Examples

```bash
# A capture to a dataset, with the autoencoder and latent columns.
netgen build data/samples/demo_capture.pcap --out out/demo_capture --autoencoder --latents

# A labelled flow export, with the benchmark.
netgen build data/samples/demo_flows.csv --out out/demo_flows --benchmark

# Fit once, then transform new flows with the same artefacts.
netgen fit data/samples/demo_flows.csv --artefacts out/artefacts
netgen transform data/samples/demo_flows.csv --artefacts out/artefacts --out out/flows.parquet
```

Further reading: [CONTEXT.md](../CONTEXT.md) for the terms, [architecture.md](architecture.md) for the design, and the decisions in [the flow record ADR](adr/0001-common-flow-record.md) and [the learning-stage protocol ADR](adr/0002-own-learning-stage-protocol.md).
