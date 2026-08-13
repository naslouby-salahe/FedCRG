"""Freeze exact runtime distribution versions after the first validated protocol run."""

from pathlib import Path

from fedcrg.artifacts.environment_lock import EnvironmentLocker


if __name__ == "__main__":
    result = EnvironmentLocker().freeze(Path("requirements.lock"))
    print(f"wrote {result.path} sha256={result.sha256.value}")
