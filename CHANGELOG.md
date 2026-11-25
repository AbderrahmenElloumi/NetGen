# Changelog

All notable changes to NetGen are recorded in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.1.0] - 2026-11-25

### Added

- The flow record: one list of fields that every input fills, with 6
  identity fields, the optional label and 29 features. Each field declares
  its type, unit, role and scaling: `log` for heavy-tailed features (counts,
  sizes, durations, gaps, rates), `linear` for the down/up ratio and the
  `is_tcp`, `is_udp` and `is_icmp` indicators.
- The fill step, the first learning stage. It learns from the training
  data the largest finite value and the median of each feature, and saves
  both in `fill.json`. Infinite values become the largest finite value and
  missing values become the median; the replacements are counted per
  column. A feature with no finite training value is filled with 0, with a
  warning.
- Flows from a capture carry the full feature set of the flow record, with
  the count of packets read and of non-IP packets dropped. A flow whose start
  time lies after 2262 gets a missing `start_time` and a warning.
- The autoencoder stage adds the anomaly score (`anomaly_score`) and,
  optionally, latent columns (`latent_0` to `latent_<n-1>`) as derived
  columns: written as computed, never rescaled and never learned from.
- `--fit-on-label` chooses the autoencoder's training rows: only the
  training flows carrying that label. It requires `--autoencoder`. The
  scaler still fits on every training flow. The command wiring comes with
  the commands.
- The preparation fits its stages in one fixed order: the fill step, the
  scaler, then the optional autoencoder. Each stage fits on the training
  data after the stages before it have transformed that data.
- Artefacts carry `manifest.json`: the NetGen version, the format version,
  the column roles, the feature list with each feature's log flag, the label
  column, the input fingerprint (SHA-256), the seed, the flow settings, every
  autoencoder setting and the stages in order. `transform` takes the roles,
  the flow settings and the autoencoder's latent columns from it.
- Transforming refuses an input that lacks a column the artefacts need,
  artefacts with a format version other than 1, feature columns that differ
  from `scaler.json`, and an artefacts file that is malformed. Other unknown
  columns are dropped with a warning that lists them.
- Every command writes a JSON report in one shape: a header (report
  version, NetGen version, command, input fingerprint, seed, start-time
  zone), and per split the rows, the label counts, the duplicate rows, the
  statistics of each feature after preparation, the fill, out-of-range and
  constant-feature counts, and, when the autoencoder ran, the anomaly-score
  percentiles with a rank-formula ROC AUC. The file is strict JSON: an
  undefined statistic is `null`. A short console summary goes with it.
- The optional extra `benchmark` (scikit-learn) and a rewritten Random Forest
  benchmark, for `build --benchmark`. It runs on the feature columns and,
  when the autoencoder ran, a second time with the derived columns added. Its
  fixed settings are recorded: 100 trees, balanced class weights, the seed,
  2 jobs. It refuses to run, before anything is fitted, without scikit-learn,
  without a label column, or with fewer than 2 classes among the labelled
  training rows.
- The `fit` command learns the preparation from an input and writes the
  artefacts folder with its report, and no dataset. The `transform` command
  applies saved artefacts to an input and writes the dataset with
  `<out stem>.report.json` beside it, taking the flow settings and the label
  column from the manifest.
- The `build` command reads an input once, splits the flows, fits the
  preparation on the training split and transforms both splits. It writes
  `DIR/artefacts/` with the report of the training split, `DIR/train.parquet`,
  `DIR/test.parquet` and `DIR/report.json` with both splits, all or nothing.
- The split is a stratified random 80/20 split by label with the seed;
  unlabelled flows form one group. An input whose test split would be empty,
  such as one of a single flow, is refused.
- `build --benchmark` adds the Random Forest benchmark to `DIR/report.json`
  and its macro F1, with the baseline's, to the console summary.
- `--input-format capture|flow-export` selects the reader and skips
  detection. `--output-format parquet|csv` selects the dataset format;
  Parquet is the default, and the format is never inferred from the file
  name.
- Exit codes: 0 on success; 2 on a user error, with the message on standard
  error and no traceback. Any other error keeps its traceback.
- `docs/usage.md`, the user guide: every option of `fit`, `transform` and
  `build`, the artefacts folder, the output files and the report.
- `docs/architecture.md`, the design: the stages, the flow record, the
  column roles and the reasons behind them, with a Mermaid diagram.
- `README.md` and its French translation `README.fr.md`: the pitch, the
  installation, the demo on the synthetic capture and flow export with what
  comes out, the three commands, the output files and the links to the
  other documents.
- `CONTRIBUTING.md`, the contributing guide: the container services, the
  editor setup, the tests and checks, test-first defect fixes, the branch
  policy, the commit style, the changelog shape, the release process, and
  how issues and pull requests are handled.

### Changed

- New commands `netgen fit`, `netgen transform` and `netgen build`, also
  available as `python -m netgen`. They replace the old `main.py` entry point.
- Option names: `--idle-timeout` replaces `--flow-timeout`, `--out`
  replaces `--output`, the input is a positional argument instead of
  `--input`, and `--label-col` defaults to `Label`, the CICFlowMeter column
  name. The autoencoder weights go into the artefacts folder named by
  `--artefacts`, not to a `--save-model` path.
- A new artefact format: an artefacts folder holding `manifest.json`,
  `fill.json`, `scaler.json` and an optional `autoencoder.pt`.
- The report replaces the old validation report.
- The Python floor is 3.12.
- The autoencoder is off by default and is the optional extra
  `netgen[autoencoder]`.
- The autoencoder is seeded, runs on the CPU only with deterministic torch
  algorithms, and gives bit-identical weights and scores for the same seed.
  It needs at least 256 training rows, drops the incomplete last batch of
  each epoch, fails loudly on a non-finite training loss and takes its input
  size from the feature list. Its output layer is linear.
- The autoencoder's weights are saved in `autoencoder.pt` as a weights-only
  `state_dict` and loaded with `weights_only=True`, never with pickle.
- The Random Forest benchmark trains on the training split and scores the
  build's test split; it does not split the data again. It reports macro
  F1, balanced accuracy, per-class precision, recall, F1 and support, and a
  confusion matrix with label names, for the model and for a majority-class
  baseline. Unlabelled rows are skipped and counted; the dataset keeps them.
- Tests and commands run through the container services:
  `docker compose run --rm <service>`.
- Scaling is one method for every feature: a Z-score with the training mean
  and the population standard deviation. The log step (`log1p`) applies only
  to the heavy-tailed features the flow record declares `log`.
- A constant feature is kept and written as 0. The report names it and, at
  `transform`, counts the values that differ from the training constant.
- No value is clipped. Values outside the training range pass through the
  formula and are counted per column.
- Flows from a capture are bidirectional: both directions of a conversation
  form one flow, and forward is the direction of the first packet. A flow
  closes at an RST, or once both sides have sent FIN and the last ACK has
  arrived. The idle timeout is 120 seconds by default, and an optional
  active timeout cuts a long-lived flow. A zero or negative timeout is
  refused.
- Input detection reads only the magic bytes of a capture and the header of
  a CICFlowMeter CSV flow export, asking each reader in a fixed order.
  Unknown input fails with a clear error that names the two supported
  formats.
- Output is written all or nothing: each command writes into a temporary
  folder beside the target and renames it into place only on success. An
  existing target is refused unless `--force` is given.

### Removed

- The old single command and its options: `--input`, `--output`,
  `--no-learning`, `--no-normalize`, `--no-validate`, `--batch-size`,
  `--save-model` and `--flow-timeout`.
- The benchmark's quality bands (POOR to EXCELLENT) and its impurity
  feature importances.
- The old normaliser (`DatasetNormalizer`), with its Min-Max option and its
  name-prefix log rule. The scaler and `scaler.json` replace it.
- Input formats Zeek `conn.log`, syslog, JSON, generic CSV, ARFF and Parquet.
  Baseline commit `6f660a2` still holds their code.
- The TCP flag-count features (FIN, SYN, RST, PSH, ACK, URG). The
  CIC-IDS2017 flow exports hold 0 or 1 in these columns, not a count per
  flow, so a capture and a flow export cannot fill them alike.
- `main.py` and `requirements.txt`. `pyproject.toml` plus `uv.lock` is the
  only dependency list.
- The dpkt fallback of the capture reader. scapy is the only capture reader.
- The old validation output, with its highly correlated feature pairs and
  its claim of a comparison with a reference dataset.
- The old code guide and testing guide under `docs/`. `docs/usage.md` and
  `docs/architecture.md` replace them.

### Fixed

- An input that lacks a column the artefacts need is refused, not skipped.
- The autoencoder is never silently skipped. When it is asked for but
  impossible (torch missing, fewer than 256 training rows, `--fit-on-label`
  matching no row or given for an input without labels, a non-finite loss),
  the fit fails with a clear error before anything is written.
- Payload bytes of a captured packet exclude the IP and transport headers
  and the Ethernet padding: a pure ACK carries 0 bytes.
- An ICMP error packet stays ICMP, whatever header it embeds, and ICMPv6 of
  any type counts as ICMP.
- Non-IP packets (ARP and the like) in a capture are dropped and counted.
- A capture is recognised by every pcap magic number (both byte orders,
  microsecond and nanosecond) and by the pcapng magic.
- The packet that expires a flow starts the new flow; it is no longer lost.
- Closing ACKs join their flow and no longer start phantom flows.
- A rate over zero duration, such as that of a one-packet flow, is a missing
  value, not a huge number. The down/up ratio is the integer part of
  backward over forward packets, so a flow with no reverse traffic gives 0,
  never the forward count.
- A CICFlowMeter CSV is recognised only by its distinctive header names; any
  other CSV gets a clear error. Header names are stripped of spaces, and a
  repeated name, such as the second `Fwd Header Length`, keeps its first copy
  with a warning. Columns outside the flow record are dropped with one
  warning that lists them.
- A CICFlowMeter CSV is read as UTF-8 with a cp1252 fallback, and the report
  records the encoding used.
- CICFlowMeter protocol numbers map to names, including floats such as
  `6.0`; ICMPv6 counts as ICMP, and an unknown or missing number is `other`.
- CICFlowMeter timestamps are read day-first, with or without seconds. A
  timestamp that does not parse fails the read and names its line.
- CICFlowMeter times convert from microseconds to seconds. Negative times and
  sizes, and infinite values, become missing values. A rate over zero
  duration and a down/up ratio over zero forward packets become missing
  values too, as they do for a capture.
- The old format detector and its misclassifications are gone.
- The preparation no longer fits on every row before the split, and the
  autoencoder no longer scores its training rows from training mode: `build`
  fits on the training split only and makes both datasets with `transform`.
- A failed command leaves no partial output, and a failed `--force` run
  leaves the old output untouched.

[0.1.0]: https://github.com/AbderrahmenElloumi/NetGen/releases/tag/v0.1.0
