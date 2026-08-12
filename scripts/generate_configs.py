#!/usr/bin/env python3
"""
Generate YAML configuration files for FedCRG.

This script generates the normative configuration files from the Pydantic
configuration models in fedcrg/config.py.

Usage:
    python scripts/generate_configs.py

Generated files:
    - configs/protocol_v2.yaml
    - configs/nbaiot_primary.yaml
    - configs/diad_external.yaml
    - configs/synthetic.yaml
"""

from pathlib import Path

from fedcrg.config import (
    create_protocol_v2_config,
    create_nbaiot_primary_config,
    create_diad_external_config,
    create_synthetic_config,
    save_config,
)


CONFIGS_DIR = Path("/home/naslouby/Projects/FedCRG/configs")


def main() -> None:
    """Generate all YAML configuration files."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate protocol_v2.yaml
    protocol_config = create_protocol_v2_config()
    save_config(protocol_config, CONFIGS_DIR / "protocol_v2.yaml")
    print(f"Generated: {CONFIGS_DIR / 'protocol_v2.yaml'}")

    # Generate nbaiot_primary.yaml
    nbaiot_config = create_nbaiot_primary_config()
    save_config(nbaiot_config, CONFIGS_DIR / "nbaiot_primary.yaml")
    print(f"Generated: {CONFIGS_DIR / 'nbaiot_primary.yaml'}")

    # Generate diad_external.yaml
    diad_config = create_diad_external_config()
    save_config(diad_config, CONFIGS_DIR / "diad_external.yaml")
    print(f"Generated: {CONFIGS_DIR / 'diad_external.yaml'}")

    # Generate synthetic.yaml
    synthetic_config = create_synthetic_config()
    save_config(synthetic_config, CONFIGS_DIR / "synthetic.yaml")
    print(f"Generated: {CONFIGS_DIR / 'synthetic.yaml'}")

    print("\nAll configuration files generated successfully.")


if __name__ == "__main__":
    main()
