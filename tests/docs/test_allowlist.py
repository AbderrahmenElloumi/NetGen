"""Tests of the public allowlist: its reader in tests/docs and the private file scripts/public-allowlist.txt."""

import pytest
from allowlist import ALLOWLIST, Allowlist, is_public, load, parse

# The entries of the public tree (spec section 14, Release process).
EXPECTED_ENTRIES = [
    "src/",
    "tests/",
    "!tests/release/",
    "docs/",
    "!docs/agents/",
    "data/samples/",
    "README.md",
    "README.fr.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "CONTEXT.md",
    "pyproject.toml",
    "uv.lock",
    "Dockerfile",
    "compose.yaml",
    ".gitignore",
    ".dockerignore",
    ".gitattributes",
    ".github/workflows/",
    "out/.gitkeep",
]
PUBLIC = [
    "src/netgen/cli.py",
    "tests/test_cli.py",
    "docs/usage.md",
    "data/samples/demo_capture.pcap",
    ".gitignore",
    ".dockerignore",
    ".gitattributes",
    ".github/workflows/ci.yml",
    "out/.gitkeep",
]
PRIVATE = [
    "tests/release/x.py",
    "docs/agents/project.md",
    ".scratch/x.md",
    "AGENTS.md",
    "CLAUDE.md",
    "scripts/release.sh",
    "data/external/x.csv",
    ".claude/x",
]


@pytest.fixture
def real_allowlist() -> Allowlist:
    return load(ALLOWLIST)


def test_comments_and_blank_lines_are_ignored() -> None:
    allowlist = parse("# a comment\n\nsrc/\n  \n!src/private/\n")
    assert allowlist.include == ("src/",)
    assert allowlist.exclude == ("src/private/",)


def test_directory_entry_matches_everything_below_it() -> None:
    allowlist = parse("docs/\n")
    assert is_public("docs/usage.md", allowlist)
    assert is_public("docs/a/b/c.md", allowlist)
    assert is_public("docs", allowlist)
    assert not is_public("docsx/usage.md", allowlist)


def test_file_entry_matches_exactly_one_file() -> None:
    allowlist = parse("README.md\n")
    assert is_public("README.md", allowlist)
    assert not is_public("README.md.bak", allowlist)
    assert not is_public("README.md/x", allowlist)


def test_path_on_no_entry_is_private() -> None:
    assert not is_public("AGENTS.md", parse("src/\n"))


def test_exclusion_beats_inclusion_whatever_the_order() -> None:
    for text in ("tests/\n!tests/release/\n", "!tests/release/\ntests/\n"):
        allowlist = parse(text)
        assert is_public("tests/test_cli.py", allowlist)
        assert not is_public("tests/release/x.py", allowlist)
    assert not is_public("README.md", parse("README.md\n!README.md\n"))


@pytest.mark.parametrize(
    "line",
    [
        "/src/",
        r"src\netgen/",
        "./src/",
        "../src/",
        "src/../scripts/",
        "!",
        "a b",
        "src//x",
    ],
)
def test_malformed_entry_is_refused(line: str) -> None:
    with pytest.raises(ValueError, match="allowlist line 2"):
        parse(f"src/\n{line}\n")


def test_absent_allowlist_skips(tmp_path) -> None:
    with pytest.raises(pytest.skip.Exception, match="public-allowlist.txt is absent"):
        load(tmp_path / "public-allowlist.txt")


def test_allowlist_file_holds_the_public_entries() -> None:
    if not ALLOWLIST.exists():
        pytest.skip("scripts/public-allowlist.txt is absent")
    lines = [
        line.strip()
        for line in ALLOWLIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert sorted(lines) == sorted(EXPECTED_ENTRIES)


@pytest.mark.parametrize("path", PUBLIC)
def test_public_paths(real_allowlist: Allowlist, path: str) -> None:
    assert is_public(path, real_allowlist)


@pytest.mark.parametrize("path", PRIVATE)
def test_private_paths(real_allowlist: Allowlist, path: str) -> None:
    assert not is_public(path, real_allowlist)
