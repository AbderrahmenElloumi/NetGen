# One file for both images: compose passes the full Python version.
ARG PYTHON_VERSION=3.15.2

# base: Python, git and uv. The lock service uses it, so it builds without uv.lock.
FROM python:${PYTHON_VERSION}-slim-trixie AS base

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 1000 --user-group --no-create-home --home-dir /tmp netgen

ENV UV_PYTHON=/usr/local/bin/python \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# runtime: the locked dependencies, both extras and the dev group, outside /app.
# The source is mounted at run time; this layer rebuilds only when the lock changes.
FROM base AS runtime

COPY pyproject.toml uv.lock /opt/lock/
RUN cd /opt/lock \
    && uv sync --locked --no-install-project --all-extras --no-cache \
    && rm -rf /opt/lock

USER netgen
