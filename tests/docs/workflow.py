"""Checkers of the CI workflow. Each takes the workflow or compose.yaml text.

The workflow is read with a line scan: no YAML library is installed, and the
tests add no dependency.
"""

import re

USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(\S+)")
PINNED = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
DOCKER_RUN = re.compile(r"docker compose run --rm ([A-Za-z0-9][A-Za-z0-9_.-]*)")
CHOWN_OUT = re.compile(r"chown\s+(\S+)\s+out/?(?:\s|$)")
COMPOSE_USER = re.compile(r"""^\s+user:\s*["']?([^"'\s]+)["']?\s*$""", re.MULTILINE)
DEV = re.compile(r"(?<![\w-])dev(?![\w-])")
CHECKOUT = re.compile(r"^\s*-\s+uses:\s*actions/checkout@")
PERSIST = re.compile(r"^\s+persist-credentials:\s*(\S+)\s*$")
JOB_TIMEOUT = re.compile(r"^    timeout-minutes:\s*(\S+)\s*$", re.MULTILINE)


def triggers(text: str) -> dict[str, list[str]]:
    """The events under the top-level `on:` key, each with its branches."""
    events: dict[str, list[str]] = {}
    inside = False
    event = ""
    in_branches = False
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():
            inside = line.rstrip() == "on:"
            continue
        if not inside:
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()
        if indent == 2:
            event = content.rstrip(":")
            events[event] = []
            in_branches = False
        elif content.startswith("branches:"):
            value = content.removeprefix("branches:").strip()
            in_branches = not value
            if value:
                events[event] = [b.strip() for b in value.strip("[]").split(",")]
        elif in_branches and content.startswith("- "):
            events[event].append(content[2:].strip())
        else:
            in_branches = False
    return events


def unpinned_actions(text: str) -> list[str]:
    """Every `uses:` line whose action ref is not a 40-hex commit SHA."""
    return [
        line.strip()
        for line in text.splitlines()
        if (match := USES.match(line)) and not PINNED.match(match.group(1))
    ]


def compose_runs(text: str) -> list[str]:
    """The service of every `docker compose run --rm X`, in order."""
    return DOCKER_RUN.findall(text)


def names_dev(text: str) -> bool:
    """True when the word `dev` appears anywhere, as a branch or a service."""
    return DEV.search(text) is not None


def out_owner(text: str) -> str | None:
    """The owner given to `out/` by a `chown`, or None when no step sets it."""
    match = CHOWN_OUT.search(text)
    return match.group(1) if match else None


def compose_user(text: str) -> str | None:
    """The `user:` setting of compose.yaml, or None when it sets none."""
    match = COMPOSE_USER.search(text)
    return match.group(1) if match else None


def checkout_credentials(text: str) -> str | None:
    """The `persist-credentials:` value of the actions/checkout step, or None."""
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if CHECKOUT.match(line):
            step = len(line) - len(line.lstrip())
            for later in lines[number + 1 :]:
                if not later.strip():
                    continue
                if len(later) - len(later.lstrip()) <= step:
                    return None
                if match := PERSIST.match(later):
                    return match.group(1)
    return None


def job_timeout(text: str) -> str | None:
    """The job-level `timeout-minutes:` value, or None when the job sets none."""
    match = JOB_TIMEOUT.search(text)
    return match.group(1) if match else None
