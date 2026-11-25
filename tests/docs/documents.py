"""Checkers of the user documents. Each takes document text and returns what breaks a rule.

Markdown is read with regular expressions and compose.yaml with a line scan:
no Markdown or YAML library is installed, and the tests add no dependency.
"""

import argparse
import posixpath
import re
from pathlib import Path

from allowlist import Allowlist, is_public, parse

FENCE = re.compile(r"^(```|~~~)[^\n]*\n(.*?)^\1[ \t]*$", re.DOTALL | re.MULTILINE)
CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.DOTALL)
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^(#{1,6})\s", re.MULTILINE)
OPTION = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
DOCKER_RUN = re.compile(r"docker compose run --rm ([A-Za-z0-9][A-Za-z0-9_.-]*)")
TOKEN_EDGES = "`'\"()[]{},;:"
# A token holding one of these is a placeholder or a pattern, not a path.
PLACEHOLDER = re.compile(r"[<>*${}]|://")


def code_blocks(text: str) -> list[str]:
    """The contents of the fenced code blocks, in order."""
    return [match.group(2).rstrip("\n") for match in FENCE.finditer(text)]


def _prose(text: str) -> str:
    return FENCE.sub("", text)


def code_fragments(text: str) -> list[str]:
    """The fenced code blocks, then the inline code spans of the prose."""
    spans = [match.group(2) for match in CODE_SPAN.finditer(_prose(text))]
    return code_blocks(text) + spans


def heading_levels(text: str) -> list[int]:
    """The level of each heading, in order; lines inside code blocks are not headings."""
    return [len(match.group(1)) for match in HEADING.finditer(_prose(text))]


def language_line_missing(text: str, expected: str) -> bool:
    """True when the document's first line is not the expected language line."""
    return text.splitlines()[:1] != [expected]


def badges(text: str) -> list[str]:
    """The badge lines (an image link that opens a line), outside code blocks."""
    return [line for line in _prose(text).splitlines() if line.startswith("[![")]


# The handle placeholder is split: the release leak scan refuses a public file
# that holds it whole, and this file is public.
RELEASE_PLACEHOLDERS = ("<" + "owner>", "vX.Y.Z")


def release_placeholders(text: str) -> set[str]:
    """The placeholders of the release preparation that the document still holds."""
    return {placeholder for placeholder in RELEASE_PLACEHOLDERS if placeholder in text}


def unreleased_missing(text: str) -> bool:
    return re.search(r"^## Unreleased[ \t]*$", _prose(text), re.MULTILINE) is None


# Commands and options


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def parser_commands(parser: argparse.ArgumentParser) -> set[str]:
    return set(_subparsers(parser))


def parser_options(parser: argparse.ArgumentParser) -> set[str]:
    """The long options of every command, without argparse's own --help."""
    return {
        option
        for command in _subparsers(parser).values()
        for action in command._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }


def _command_text(text: str) -> list[str]:
    """The code fragments with any leading `docker compose run --rm <service>` stripped."""
    return [
        re.sub(r"docker compose run --rm \S+", "", fragment)
        for fragment in code_fragments(text)
    ]


def documented_options(text: str) -> set[str]:
    return {
        option
        for fragment in _command_text(text)
        for option in OPTION.findall(fragment)
    }


def unlisted_commands(commands: set[str], text: str) -> set[str]:
    fragments = _command_text(text)
    return {
        command
        for command in commands
        if not any(
            re.search(rf"\bnetgen {command}\b", fragment) for fragment in fragments
        )
    }


def unlisted_options(options: set[str], text: str) -> set[str]:
    return options - documented_options(text)


def unknown_options(options: set[str], text: str) -> set[str]:
    return documented_options(text) - options - {"--help"}


# Compose services


def compose_services(text: str) -> set[str]:
    """The keys one level under the top-level `services:` key."""
    services: set[str] = set()
    inside = False
    for line in text.splitlines():
        if line and not line[0].isspace() and not line.startswith("#"):
            inside = line.rstrip() == "services:"
            continue
        match = re.match(r"^  ([A-Za-z0-9][A-Za-z0-9_.-]*):", line)
        if inside and match:
            services.add(match.group(1))
    return services


def docker_services_named(text: str) -> set[str]:
    """The service of every `docker compose run --rm X` in the document's code."""
    return {
        service
        for fragment in code_fragments(text)
        for service in DOCKER_RUN.findall(fragment)
    }


# Repository paths


def top_level_entries(repo: Path, allowlist_path: Path) -> set[str]:
    """The top-level files and directories of the repository and of the allowlist."""
    entries = {child.name for child in repo.iterdir()}
    if allowlist_path.exists():
        allowlist = parse(allowlist_path.read_text(encoding="utf-8"))
        for entry in allowlist.include + allowlist.exclude:
            entries.add(entry.split("/")[0])
    return entries


def named_paths(text: str, document: str, top_level: set[str]) -> set[str]:
    """The repository paths a document names, relative to the repository root.

    A path is a code token or a link target whose first segment is a top-level
    entry. Link targets are relative to the document's folder. Paths under
    `out/` are output locations created by a run: they are left out here, so
    both the existence rule and the allowlist rule exempt them.
    """
    candidates: set[str] = set()
    for fragment in code_fragments(text):
        for token in re.split(r"[\s=]+", fragment):
            candidates.add(token.strip(TOKEN_EDGES).rstrip("."))
    folder = posixpath.dirname(document)
    for target in LINK.findall(_prose(text)):
        if PLACEHOLDER.search(target) or target.startswith(("#", "mailto:")):
            continue
        target = target.split("#")[0]
        candidates.add(posixpath.normpath(posixpath.join(folder, target)))
    return {
        path.rstrip("/")
        for path in candidates
        if path
        and not PLACEHOLDER.search(path)
        and path.split("/")[0] in top_level
        and path.split("/")[0] != "out"
    }


def missing_paths(
    text: str, document: str, repo: Path, top_level: set[str]
) -> set[str]:
    return {
        path
        for path in named_paths(text, document, top_level | {".."})
        if not (repo / path).exists() or path.startswith("..")
    }


def private_paths(
    text: str, document: str, top_level: set[str], allowlist: Allowlist
) -> set[str]:
    return {
        path
        for path in named_paths(text, document, top_level)
        if not is_public(path, allowlist)
    }
