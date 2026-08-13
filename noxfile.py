"""Nox sessions for the FedCRG quality gate.

Sessions reuse pyproject.toml configuration; they never duplicate Ruff, Pyright, or
pytest configuration.
"""

from __future__ import annotations

import nox

nox.options.sessions = ["lint", "typecheck", "unit"]


@nox.session
def format(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "format", "src", "tests")


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff")
    session.run("ruff", "check", "src", "tests")
    session.run("ruff", "format", "--check", "src", "tests")


@nox.session
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".")
    session.install("pyright")
    session.run("pyright", "src/fedcrg")


@nox.session
def unit(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "tests/unit")


@nox.session
def integration(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "tests/integration")


@nox.session
def contract(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "tests/contract")


@nox.session
def regression(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "tests/regression")


@nox.session
def audit(session: nox.Session) -> None:
    session.install("-e", ".")
    session.run("python", "tools/audit_repository.py")


@nox.session
def quality(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.install("pyright")
    session.run("ruff", "format", "--check", "src", "tests")
    session.run("ruff", "check", "src", "tests")
    session.run("pyright", "src/fedcrg")
    session.run("pytest")
