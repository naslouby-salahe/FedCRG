"""FedCRG Setup Configuration.

This setup.py enables the FedCRG CLI to be installed and run via `fedcrg` command.
"""

from setuptools import setup, find_packages

setup(
    name="fedcrg",
    version="2.0",
    description="FedCRG: Federated Calibration Readiness Gate",
    author="FedCRG Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "click>=8.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "pandas>=2.0",
        "pydantic>=2.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3",
        "torch>=2.0",
    ],
    entry_points={
        "console_scripts": [
            "fedcrg = fedcrg.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
