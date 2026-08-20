"""Integration tests for the CLI entry point."""

from __future__ import annotations

from click.testing import CliRunner

from fedcrg.cli import cli


def test_cli_help_exposes_target_surface() -> None:
    """The top-level --help output lists every expected subcommand."""
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in (
        "validate",
        "preprocess",
        "plan",
        "run",
        "status",
        "monitor",
        "results",
    ):
        assert command in result.output
    for command in ("campaign", "report"):
        assert command not in result.output


def test_cli_validate_primary_experiment() -> None:
    """Validating a known experiment id succeeds and reports valid=true."""
    result = CliRunner().invoke(cli, ["validate", "primary_nbaiot"])
    assert result.exit_code == 0
    assert '"valid": true' in result.output


def test_cli_rejects_unknown_command() -> None:
    """An unrecognized subcommand exits with a non-zero status."""
    result = CliRunner().invoke(cli, ["not-a-command"])
    assert result.exit_code != 0
