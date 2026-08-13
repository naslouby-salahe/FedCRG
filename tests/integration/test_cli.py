from click.testing import CliRunner
from fedcrg.cli.app import cli


def test_cli_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "experiment" in result.output
    assert "verify" in result.output
