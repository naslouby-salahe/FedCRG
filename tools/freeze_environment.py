"""Freeze exact runtime distribution versions after the first validated protocol run."""

import subprocess
from pathlib import Path

from fedcrg.evidence.store import capture_environment


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]
    environment = capture_environment(repository_root)
    frozen = subprocess.run(
        ("python", "-m", "pip", "freeze"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    output = Path("requirements.lock")
    output.write_text(frozen, encoding="utf-8")
    print(f"wrote {output} git={environment.git_commit}")
