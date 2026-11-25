"""Structure tests of the CI workflow .github/workflows/ci.yml (spec section 11, Continuous integration).

GitHub runs the workflow, not the container, so these tests read it with a
line scan. Each checker also runs on a doctored string, which proves the rule
can fail.
"""

import pytest
from allowlist import REPO
from documents import compose_services
from workflow import (
    checkout_credentials,
    compose_runs,
    compose_user,
    job_timeout,
    names_dev,
    out_owner,
    triggers,
    unpinned_actions,
)

WORKFLOW = ".github/workflows/ci.yml"
SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow() -> str:
    return read(WORKFLOW)


# Triggers


def test_triggers_are_main_and_manual_dispatch(workflow: str) -> None:
    assert triggers(workflow) == {
        "push": ["main"],
        "pull_request": ["main"],
        "workflow_dispatch": [],
    }


def test_no_dev_branch_anywhere(workflow: str) -> None:
    assert not names_dev(workflow)


def test_dev_anywhere_is_found() -> None:
    assert names_dev("      - run: git fetch origin dev\n")
    assert not names_dev("      - run: docker compose run --rm test-floor\n")


def test_dev_trigger_is_found() -> None:
    text = "on:\n  push:\n    branches: [main, dev]\n  workflow_dispatch:\n"
    assert triggers(text) == {"push": ["main", "dev"], "workflow_dispatch": []}


def test_block_list_branches_are_read() -> None:
    text = "on:\n  pull_request:\n    branches:\n      - main\njobs:\n"
    assert triggers(text) == {"pull_request": ["main"]}


# Pinned actions


def test_every_action_is_pinned_to_a_commit_sha(workflow: str) -> None:
    assert unpinned_actions(workflow) == []
    assert "uses:" in workflow


def test_pinned_action_carries_its_tag_in_a_comment(workflow: str) -> None:
    for line in workflow.splitlines():
        if "uses:" in line:
            assert line.split("#", 1)[1].strip().startswith("v")


@pytest.mark.parametrize(
    "line",
    [
        "      - uses: actions/x@v4",
        "      - uses: actions/x@main",
        "      - uses: actions/x",
        f"      - uses: actions/x@{SHA[:39]}",
        f"      - uses: actions/x@{SHA.upper()}",
        f"      - uses: actions/x@{SHA}0",
    ],
)
def test_floating_or_malformed_ref_is_unpinned(line: str) -> None:
    pinned = f"      - uses: actions/checkout@{SHA} # v7.0.1"
    assert unpinned_actions(f"{pinned}\n{line}\n") == [line.strip()]


# Service steps


def test_services_run_in_order(workflow: str) -> None:
    assert compose_runs(workflow) == ["test", "test-floor", "check"]


def test_each_run_names_a_compose_service(workflow: str) -> None:
    assert set(compose_runs(workflow)) <= compose_services(read("compose.yaml"))


def test_runs_are_read_in_order() -> None:
    text = (
        "      - run: docker compose run --rm check\n"
        "      - run: docker compose run --rm test\n"
    )
    assert compose_runs(text) == ["check", "test"]


# out/ preparation for check


def test_out_is_made_writable_for_the_compose_user(workflow: str) -> None:
    user = compose_user(read("compose.yaml"))
    assert user == "1000:1000"
    assert out_owner(workflow) == user


def test_out_owner_mismatch_is_found() -> None:
    text = "      - run: mkdir -p out && sudo chown 1001:1001 out\n"
    assert out_owner(text) == "1001:1001"
    assert out_owner("      - run: mkdir -p out\n") is None


def test_compose_user_is_read() -> None:
    assert compose_user('x-limits: &limits\n  user: "1234:5678"\n') == "1234:5678"
    assert compose_user("services:\n") is None


# What the workflow must not hold


@pytest.mark.parametrize("word", ["secrets.", "cache", "release", "tags:"])
def test_no_secret_cache_release_or_tag_trigger(workflow: str, word: str) -> None:
    assert word not in workflow


def test_minimum_permission(workflow: str) -> None:
    assert "permissions:\n  contents: read\n" in workflow


# Checkout credentials and job time limit


def test_checkout_does_not_persist_credentials(workflow: str) -> None:
    assert checkout_credentials(workflow) == "false"


def test_job_sets_a_time_limit(workflow: str) -> None:
    assert job_timeout(workflow) == "30"


CHECKOUT = f"      - uses: actions/checkout@{SHA} # v7.0.1\n"


@pytest.mark.parametrize(
    ("text", "value"),
    [
        (CHECKOUT + "      - name: test\n", None),
        (CHECKOUT + "        with:\n          persist-credentials: true\n", "true"),
        (CHECKOUT + "        with:\n          persist-credentials: false\n", "false"),
        (
            CHECKOUT
            + "      - uses: actions/other@x\n"
            + "        with:\n          persist-credentials: false\n",
            None,
        ),
        (
            "      - uses: actions/other@x\n        with:\n          persist-credentials: false\n",
            None,
        ),
    ],
)
def test_checkout_credentials_are_read(text: str, value: str | None) -> None:
    assert checkout_credentials(text) == value


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("jobs:\n  ci:\n    runs-on: ubuntu-latest\n    steps:\n", None),
        ("jobs:\n  ci:\n    timeout-minutes: 45\n    steps:\n", "45"),
        (
            "jobs:\n  ci:\n    steps:\n      - run: x\n        timeout-minutes: 30\n",
            None,
        ),
    ],
)
def test_job_timeout_is_read(text: str, value: str | None) -> None:
    assert job_timeout(text) == value
