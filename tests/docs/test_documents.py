"""Documentation tests: the user documents stay true to the code, to compose.yaml and to the public allowlist.

Each rule runs against the real documents, and its checker also runs on a
doctored string, which proves the rule can fail.
"""

from pathlib import Path

import pytest
from allowlist import REPO, load, parse
from documents import (
    RELEASE_PLACEHOLDERS,
    badges,
    code_blocks,
    compose_services,
    docker_services_named,
    heading_levels,
    language_line_missing,
    missing_paths,
    parser_commands,
    parser_options,
    private_paths,
    release_placeholders,
    top_level_entries,
    unknown_options,
    unlisted_commands,
    unlisted_options,
    unreleased_missing,
)

from netgen.cli import build_parser

# The five user documents. CHANGELOG.md may name removed files and formats, so
# only its `## Unreleased` heading is tested.
DOCUMENTS = [
    "README.md",
    "README.fr.md",
    "docs/usage.md",
    "docs/architecture.md",
    "CONTRIBUTING.md",
]
SERVICES = {"test", "test-floor", "check", "dev", "lock"}


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def top_level() -> set[str]:
    return top_level_entries(REPO, (REPO / "scripts" / "public-allowlist.txt"))


# Commands and options


def test_parser_lists_the_three_commands() -> None:
    assert parser_commands(build_parser()) == {"fit", "transform", "build"}


def test_usage_lists_every_command_and_option() -> None:
    parser = build_parser()
    usage = read("docs/usage.md")
    assert unlisted_commands(parser_commands(parser), usage) == set()
    assert unlisted_options(parser_options(parser), usage) == set()


def test_usage_names_only_options_of_the_parser() -> None:
    assert (
        unknown_options(parser_options(build_parser()), read("docs/usage.md")) == set()
    )


def test_unlisted_command_and_option_are_found() -> None:
    usage = "Run `netgen fit INPUT --seed N`.\n"
    assert unlisted_commands({"fit", "build"}, usage) == {"build"}
    assert unlisted_options({"--seed", "--epochs"}, usage) == {"--epochs"}


def test_option_only_in_prose_is_unlisted() -> None:
    assert unlisted_options({"--seed"}, "Pass --seed to fix it.\n") == {"--seed"}


def test_unknown_option_is_found() -> None:
    usage = "```\nnetgen build INPUT --out DIR --colour\n```\n"
    assert unknown_options({"--out"}, usage) == {"--colour"}


def test_docker_rm_is_not_taken_for_an_option() -> None:
    usage = "`docker compose run --rm dev netgen build INPUT --out DIR`\n"
    assert unknown_options({"--out"}, usage) == set()


# The README pair


def test_readme_pair_has_the_same_headings_and_code_blocks() -> None:
    english, french = read("README.md"), read("README.fr.md")
    assert heading_levels(english) == heading_levels(french)
    assert code_blocks(english) == code_blocks(french)


ENGLISH_LINE = "English | [Français](README.fr.md)"
FRENCH_LINE = "[English](README.md) | Français"


def test_readme_pair_opens_with_the_language_line() -> None:
    assert not language_line_missing(read("README.md"), ENGLISH_LINE)
    assert not language_line_missing(read("README.fr.md"), FRENCH_LINE)


def test_missing_language_line_is_found() -> None:
    assert language_line_missing("English | Français\n# NetGen\n", ENGLISH_LINE)
    assert language_line_missing(f"# NetGen\n{ENGLISH_LINE}\n", ENGLISH_LINE)
    assert language_line_missing("", ENGLISH_LINE)


def test_heading_levels_differ_on_a_doctored_copy() -> None:
    assert heading_levels("# A\n## B\n") != heading_levels("# A\n### B\n")


def test_headings_inside_code_blocks_do_not_count() -> None:
    assert heading_levels("# A\n```\n# a shell comment\n```\n") == [1]


def test_code_blocks_differ_on_a_doctored_copy() -> None:
    first = "```sh\nnetgen build x\n```\n"
    assert code_blocks(first) == ["netgen build x"]
    assert code_blocks(first) != code_blocks("```sh\nnetgen build y\n```\n")


def test_readme_pair_carries_the_same_badges() -> None:
    english = badges(read("README.md"))
    assert english
    assert english == badges(read("README.fr.md"))


def test_badges_differ_on_a_doctored_copy() -> None:
    first = "# A\n\n[![CI](https://x/badge.svg)](https://x)\n"
    assert badges(first) == ["[![CI](https://x/badge.svg)](https://x)"]
    assert badges(first) != badges("# A\n")
    assert badges("```\n[![CI](https://x/badge.svg)](https://x)\n```\n") == []


@pytest.mark.parametrize("document", [*DOCUMENTS, "CHANGELOG.md"])
def test_documents_hold_no_release_placeholder(document: str) -> None:
    assert release_placeholders(read(document)) == set()


def test_release_placeholder_is_found() -> None:
    text = "uv tool install git+https://github.com/{}/NetGen@{}\n".format(
        *RELEASE_PLACEHOLDERS
    )
    assert release_placeholders(text) == set(RELEASE_PLACEHOLDERS)
    assert release_placeholders("git+https://github.com/me/NetGen@v0.1.0\n") == set()


# Repository paths


@pytest.mark.parametrize("document", DOCUMENTS)
def test_named_paths_exist(document: str, top_level: set[str]) -> None:
    assert missing_paths(read(document), document, REPO, top_level) == set()


def test_missing_path_is_found(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "usage.md").write_text("x")
    text = (
        "See `docs/usage.md`, [usage](usage.md#options) and `docs/gone.md`.\n"
        "```\nnetgen build data/x.pcap --out out/run/x DIR/train.parquet\n```\n"
        "[home](https://example.org/docs/x) and [gone](../docs/old.md).\n"
    )
    found = missing_paths(text, "docs/guide.md", tmp_path, {"docs", "data", "out"})
    # out/ is a run's output location, and DIR/ is a placeholder, not a path.
    assert found == {"docs/gone.md", "data/x.pcap", "docs/old.md"}


def test_link_leaving_the_repository_is_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "x.md").write_text("x")
    text = "[outside](../x.md) and [gone](../y.md)\n"
    # ../x.md exists next to the repository, but outside it: still missing.
    found = missing_paths(text, "README.md", repo, {"docs"})
    assert found == {"../x.md", "../y.md"}


# Compose services


def test_compose_lists_the_five_services() -> None:
    assert compose_services(read("compose.yaml")) == SERVICES


@pytest.mark.parametrize("document", DOCUMENTS)
def test_compose_runs_name_real_services(document: str) -> None:
    assert docker_services_named(read(document)) <= compose_services(
        read("compose.yaml")
    )


def test_unknown_service_is_found() -> None:
    text = "`docker compose run --rm tests` and `docker compose run --rm <service>`\n"
    assert docker_services_named(text) - SERVICES == {"tests"}


def test_compose_services_are_the_keys_under_services() -> None:
    compose = "x-a: &a\n  cpus: 2\nservices:\n  web:\n    image: x\n    environment:\n      KEY: v\n  db:\n    image: y\nvolumes:\n  data:\n"
    assert compose_services(compose) == {"web", "db"}


# Changelog


def test_changelog_has_unreleased() -> None:
    assert not unreleased_missing(read("CHANGELOG.md"))


def test_missing_unreleased_is_found() -> None:
    assert unreleased_missing("# Changelog\n\n## [0.1.0] - 2026-11-24\n")
    assert unreleased_missing("```\n## Unreleased\n```\n### Unreleased\n")


# The public allowlist


@pytest.mark.parametrize("document", DOCUMENTS)
def test_documents_name_only_public_paths(document: str, top_level: set[str]) -> None:
    allowlist = load()
    assert private_paths(read(document), document, top_level, allowlist) == set()


def test_private_path_is_found() -> None:
    allowlist = parse("docs/\n!docs/agents/\nREADME.md\n")
    text = "See `docs/usage.md`, `docs/agents/project.md`, `scripts/release.sh` and `out/x`.\n"
    found = private_paths(
        text, "README.md", {"docs", "scripts", "out", "README.md"}, allowlist
    )
    assert found == {"docs/agents/project.md", "scripts/release.sh"}
