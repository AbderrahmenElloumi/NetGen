[English](README.md) | Français

# NetGen

[![CI](https://github.com/AbderrahmenElloumi/NetGen/actions/workflows/ci.yml/badge.svg)](https://github.com/AbderrahmenElloumi/NetGen/actions/workflows/ci.yml)

NetGen transforme des enregistrements de trafic réseau en jeux de données tabulaires pour l'apprentissage automatique. Il lit une capture (pcap ou pcapng) ou un export de flux (CSV de CICFlowMeter), regroupe les paquets en flux et écrit une ligne par flux, avec le même ensemble fixe de caractéristiques quelle que soit la source. Il apprend sa préparation (remplissage, mise à l'échelle et un autoencodeur facultatif qui ajoute un score d'anomalie) sur les seules données d'entraînement et l'enregistre sous forme d'artefacts : les nouveaux enregistrements reçoivent ainsi exactement le même traitement. Chaque jeu de données s'accompagne d'un rapport qui décrit ses colonnes et compte ce qui a été modifié.

## Installation

Installez la commande avec [uv](https://docs.astral.sh/uv/) :

```bash
uv tool install git+https://github.com/AbderrahmenElloumi/NetGen@v0.1.0
```

L'autoencodeur et le benchmark nécessitent les extras `autoencoder` et `benchmark`. Cette commande installe les deux, avec la version CPU de PyTorch tirée de son propre index de paquets :

```bash
uv tool install "netgen[autoencoder,benchmark] @ git+https://github.com/AbderrahmenElloumi/NetGen@v0.1.0" --index https://download.pytorch.org/whl/cpu
```

NetGen fonctionne aussi depuis un clone, par les services de conteneur, sans Python sur l'hôte :

```bash
git clone https://github.com/AbderrahmenElloumi/NetGen.git
cd NetGen
docker compose run --rm dev python -m netgen --help
```

Depuis un clone, chaque commande de ce README s'exécute sous la forme `docker compose run --rm dev python -m netgen ...` depuis la racine du dépôt.

## Démo

Le dépôt contient une petite capture synthétique et un export de flux synthétique. Ces deux commandes construisent un jeu de données à partir de chacun :

```bash
docker compose run --rm dev python -m netgen build data/samples/demo_capture.pcap --out out/demo_capture --autoencoder --latents
docker compose run --rm dev python -m netgen build data/samples/demo_flows.csv --out out/demo_flows --benchmark
```

La première commande partage les flux de la capture, ajuste la préparation et l'autoencodeur sur la partition d'entraînement, puis affiche un résumé :

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

Elle écrit ce dossier :

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

Chaque partition compte 44 colonnes : 6 colonnes d'identité, 29 caractéristiques mises à l'échelle, `anomaly_score` et `latent_0` à `latent_7`. Le rapport donne, pour chaque partition, les lignes, les statistiques de chaque caractéristique, le décompte des modifications et la distribution du score d'anomalie. L'entrée de la partition de test :

```json
"anomaly_score": {"median": 0.25196661055088043, "p90": 0.8270255446434027, "p99": 1.9631604933738696, "max": 2.1366825103759766, "median_by_label": {}, "auc": null}
```

La seconde commande travaille sur des flux étiquetés. Son résumé se termine par le benchmark Random Forest, comparé à une référence qui prédit la classe majoritaire :

```text
benchmark features: macro F1 1.000, baseline macro F1 0.286
```

## Commandes

`fit` apprend la préparation à partir d'une entrée et enregistre les artefacts :

```bash
netgen fit data/samples/demo_flows.csv --artefacts out/artefacts
```

`transform` applique des artefacts enregistrés à une nouvelle entrée et écrit le jeu de données :

```bash
netgen transform data/samples/real/cicids2017_excerpt.csv --artefacts out/artefacts --out out/excerpt.parquet
```

`build` partage une entrée, ajuste sur la partition d'entraînement et transforme les deux partitions :

```bash
netgen build data/samples/demo_capture.pcap --out out/demo_capture --autoencoder
```

`netgen <command> --help` liste les options.

## Fichiers produits

- **Jeu de données** : une ligne par flux, en Parquet (ou en CSV avec `--output-format csv`) : `DIR/train.parquet` et `DIR/test.parquet` pour `build`, le fichier `--out` pour `transform`.
- **Dossier des artefacts** : la préparation apprise (`manifest.json`, `fill.json`, `scaler.json` et, avec l'autoencodeur, `autoencoder.pt`), que `transform` applique aux nouvelles entrées.
- **`report.json`** : pour chaque partition, les lignes, les étiquettes, les statistiques de chaque caractéristique, le décompte des valeurs remplies, écartées et hors plage, ainsi que le score d'anomalie et le benchmark lorsqu'ils ont été calculés.

## Documentation

- [docs/usage.md](docs/usage.md) : toutes les options, les artefacts, les fichiers produits et le rapport.
- [docs/architecture.md](docs/architecture.md) : les étapes, l'enregistrement de flux et les choix de conception.
- [CONTRIBUTING.md](CONTRIBUTING.md) : comment travailler sur NetGen.
- [CHANGELOG.md](CHANGELOG.md) : les changements de chaque version.

## Origine

Né comme projet universitaire (Cybersécurité et IA, 2025-26) ; repensé depuis.

## Licence

MIT, voir [LICENSE](LICENSE). NetGen lit les captures avec scapy, distribué sous licence GPL-2.0.
