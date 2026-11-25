import pytest


def test_package_imports() -> None:
    import netgen  # noqa: F401


@pytest.mark.slow
def test_slow_marker_is_selectable() -> None:
    pass
