English | [Français](README.fr.md)

# NetGen

[![CI](https://github.com/AbderrahmenElloumi/NetGen/actions/workflows/ci.yml/badge.svg)](https://github.com/AbderrahmenElloumi/NetGen/actions/workflows/ci.yml)

NetGen turns network-traffic recordings into tabular datasets for machine learning. It reads a capture (pcap or pcapng) or a flow export (CICFlowMeter CSV), groups the packets into flows, and writes one row per flow with the same fixed set of features whatever the source. It learns its preparation (filling, scaling and an optional autoencoder that adds an anomaly score) from the training data only, and saves it as artefacts, so new recordings get exactly the same treatment. Every dataset comes with a report that describes its columns and counts what was changed.

## Installation

Install the command with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/AbderrahmenElloumi/NetGen@v0.1.0
```

The autoencoder and the benchmark need the `autoencoder` and `benchmark` extras. This command installs both, with the CPU build of PyTorch from its own package index:

```bash
uv tool install "netgen[autoencoder,benchmark] @ git+https://github.com/AbderrahmenElloumi/NetGen@v0.1.0" --index https://download.pytorch.org/whl/cpu
```

NetGen also runs from a clone, through the container services, with no Python on the host:

```bash
git clone https://github.com/AbderrahmenElloumi/NetGen.git
cd NetGen
docker compose run --rm dev python -m netgen --help
```

From a clone, every command in this README runs as `docker compose run --rm dev python -m netgen ...` from the repository root.

## Demo

The repository holds a small synthetic capture and a synthetic flow export. These two commands build a dataset from each:

```bash
docker compose run --rm dev python -m netgen build data/samples/demo_capture.pcap --out out/demo_capture --autoencoder --latents
docker compose run --rm dev python -m netgen build data/samples/demo_flows.csv --out out/demo_flows --benchmark
```

The first command splits the flows of the capture, fits the preparation and the autoencoder on the training split, and prints a summary:

```text
train: 320 rows (unlabelled)
  filled values: 28
  non-IP packets dropped: 4
  constant features: none
  out-of-range values: 0
test: 80 rows (unlabelled)
  filled values: 0
  non-IP packets dropped: 4
  constant features: none
  out-of-range values: 1
Build written to out/demo_capture
```

It writes this folder:

```text
out/demo_capture/
├── artefacts/
│   ├── autoencoder.pt
│   ├── fill.json
│   ├── manifest.json
│   ├── report.json
│   └── scaler.json
├── report.json
├── test.parquet
└── train.parquet
```

Each split has 44 columns: 6 identity columns, 29 scaled features, `anomaly_score` and `latent_0` to `latent_7`. The report gives, per split, the rows, the statistics of each feature, the counts of the changes, and the anomaly-score distribution. The test split's entry:

```json
"anomaly_score": {"median": 0.25196661055088043, "p90": 0.8270255446434027, "p99": 1.9631604933738696, "max": 2.1366825103759766, "median_by_label": {}, "auc": null}
```

The second command works on labelled flows. Its summary ends with the Random Forest benchmark against a majority-class baseline:

```text
benchmark features: macro F1 1.000, baseline macro F1 0.286
```

## Commands

`fit` learns the preparation from an input and saves the artefacts:

```bash
netgen fit data/samples/demo_flows.csv --artefacts out/artefacts
```

`transform` applies saved artefacts to a new input and writes the dataset:

```bash
netgen transform data/samples/real/cicids2017_excerpt.csv --artefacts out/artefacts --out out/excerpt.parquet
```

`build` splits an input, fits on the training split and transforms both splits:

```bash
netgen build data/samples/demo_capture.pcap --out out/demo_capture --autoencoder
```

`netgen <command> --help` lists the options.

## Output files

- **Dataset**: one row per flow, in Parquet (or CSV with `--output-format csv`): `DIR/train.parquet` and `DIR/test.parquet` for `build`, the `--out` file for `transform`.
- **Artefacts folder**: the learned preparation (`manifest.json`, `fill.json`, `scaler.json` and, with the autoencoder, `autoencoder.pt`), which `transform` applies to new inputs.
- **`report.json`**: per split, the rows, the labels, the statistics of each feature, the counts of filled, dropped and out-of-range values, and the anomaly score and benchmark when they ran.

## Documentation

- [docs/usage.md](docs/usage.md): every option, the artefacts, the output files and the report.
- [docs/architecture.md](docs/architecture.md): the stages, the flow record and the design decisions.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to work on NetGen.
- [CHANGELOG.md](CHANGELOG.md): the changes of each version.

## Origin

Started as a university project (Cybersecurity and AI, 2025-26); redesigned since.

## Licence

MIT, see [LICENSE](LICENSE). NetGen reads captures with scapy, which is licensed under GPL-2.0.
