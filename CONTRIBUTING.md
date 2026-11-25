# Contributing to NetGen

This guide holds every developer instruction for NetGen. User documentation is
in [README.md](README.md) and [docs/usage.md](docs/usage.md); the design is in
[docs/architecture.md](docs/architecture.md).

## Issues and pull requests

Issues are welcome: bug reports and questions alike. Pull requests are not
accepted for now. Replies may be slow.

## Container services

Every NetGen command runs in a fresh container, started by
`docker compose run --rm` followed by the name of a service. Git, Docker and the editor stay on the
host. The host has no Python environment for NetGen, on purpose: do not run
`python`, `pytest` or `uv` on the host, and do not create a `.venv`.

[compose.yaml](compose.yaml) defines five services:

| Service | Purpose | Repository mount | Network |
| --- | --- | --- | --- |
| `test` | the test suite on Python 3.15, the everyday command | read-only | off |
| `test-floor` | the test suite on Python 3.12, the supported floor | read-only | off |
| `check` | `ruff check`, `ruff format --check`, `mypy`, then the demo build into `out/` | read-only, `out/` writable | off |
| `dev` | the formatter, linter fixes, an interactive shell, NetGen commands | read-write | off |
| `lock` | update `uv.lock`, set the version with `uv version` | only `pyproject.toml` and `uv.lock`, read-write | on |

Every service runs as a non-root user with 2 CPUs, 4 GB of memory, no swap,
at most 256 processes and a read-only root filesystem. `/tmp` is a tmpfs, and
pytest writes its temporary files and cache there. The `NETGEN_MEMORY`
environment variable raises the memory limit for runs on a large dataset. In
`dev`, the folder of external datasets is mounted read-only.

The source is mounted at run time, not copied into the image. Rebuild the
images with `docker compose build` after a change to
[Dockerfile](Dockerfile), [pyproject.toml](pyproject.toml) or
[uv.lock](uv.lock).

Common commands:

```bash
docker compose run --rm test
docker compose run --rm test-floor
docker compose run --rm check
docker compose run --rm dev ruff format
docker compose run --rm dev ruff check --fix
docker compose run --rm dev
docker compose run --rm lock uv lock
docker compose run --rm lock uv add <package>
```

On Windows, in Git Bash, prefix `docker compose` with `MSYS_NO_PATHCONV=1`
when an argument holds a Linux path: Git Bash rewrites such paths into
Windows paths. Keep LF line endings in files that a container runs.

## Editor setup

The editor uses the interpreter inside the container, not one on the host.

1. Build the images: `docker compose build`.
2. The interpreter is `/opt/venv/bin/python` in the `netgen:3.15` image. It
   holds every dependency, both extras and the development tools. The
   repository is mounted at `/app`, and `PYTHONPATH=/app/src` makes the
   `netgen` package importable without an install.
3. PyCharm: add a Docker Compose interpreter on the `dev` service of
   [compose.yaml](compose.yaml), with the interpreter path
   `/opt/venv/bin/python`.

VS Code cannot attach to the `dev` container. Its Dev Containers attach
installs and runs a server under `$HOME` and downloads extensions. In `dev`,
`HOME` is `/tmp`, a tmpfs mounted `noexec`; the root filesystem is read-only;
and the network is off. These limits keep the container sandboxed, so the
editor setup does not loosen them.

## Tests and checks

The test command runs the whole suite on both Python versions:

```bash
docker compose run --rm test && docker compose run --rm test-floor
```

`docker compose run --rm check` runs `ruff check`, `ruff format --check` and
`mypy`, then builds the demo datasets end to end into `out/`.

Slow tests carry the `slow` marker. While iterating, deselect them by hand,
and skip the coverage gate with `--no-cov` when running a subset:

```bash
docker compose run --rm test pytest -m "not slow" --no-cov
```

The suite enforces 90% branch coverage on the capture reader and flow
aggregation taken together.

The documents are tested too: the tests in `tests/docs/` check that the user
documents match the command line, `compose.yaml` and the repository's files.

## Defect fixes

A defect fix starts with a failing test that reproduces the defect, then the
fix makes it pass. Regression tests for the audited defects are named
`test_dNN_...` after the defect id, for example
`test_d26_expiring_packet_starts_new_flow`, and sit in the test file of the
module that owns the defect. `grep test_d` lists them.

## Glossary and decisions

The glossary is [CONTEXT.md](CONTEXT.md). Hard-to-reverse design decisions
are recorded as ADRs in [docs/adr/](docs/adr/). Use the glossary's terms in
code, tests and documents.

## Branches

Work happens on `dev`. `main` changes only by the maintainer's merge.

## Commit style

A commit subject is one line: a prefix with a colon, then a short lowercase
summary of the change. `IMPL:` and `FIX:` subjects end with the ticket number
in parentheses, for example `IMPL: build command in netgen.cli (ticket 41)`.
`REVIEW:` and `REREVIEW:` subjects name the ticket in the summary, for
example `REVIEW: pass and close ticket 44`.

| Prefix | Commit |
| --- | --- |
| `IMPL:` | first implementation of a ticket |
| `FIX:` | fixes for the findings of a review |
| `REVIEW:`, `REREVIEW:` | a review, and a review after fixes |
| `TRIAGE:` | decisions and changes to the backlog |
| `GARDEN:`, `REGARDEN:` | upkeep that keeps the specification and the glossary true to the code |
| `docs:`, `chore:` | the maintainer's documentation and maintenance commits |

## Changelog

[CHANGELOG.md](CHANGELOG.md) follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and is written by
hand. `## Unreleased` sits at the top, and each version has a
`## [X.Y.Z]` section. Entries go under the headings Added, Changed, Removed
and Fixed.

Every change of user-visible behaviour adds an entry under `## Unreleased`.
Changing the scaling of a field, or adding a field to the flow record, needs
an entry too. Stamping a version belongs to a release.

## Releases

The maintainer cuts releases from a private repository. Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The public
repository holds one squashed commit per version, and each version has an
annotated tag named after it, such as `v0.1.0`, with its changelog notes and
a GitHub Release.

Tags match the private history. The same tag name points at different
commits in the public and the private repository, which never share history.
