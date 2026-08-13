"""Locked figure builders, every numerical value comes from generated artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def readiness_frontier(frame: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots()
    ax.plot(frame["n"], frame["coverage_probability"])
    ax.axhline(0.95, linestyle="--")
    ax.set(
        xlabel="Local calibration sample size",
        ylabel="Maximum exact in-band probability",
        title="Finite-sample readiness frontier",
    )
    return _save(fig, output)


def mismatch_power_map(frame: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots()
    for sample_count, group in frame.groupby("sample_count"):
        ax.plot(group["true_fpr"], group["declaration_probability"], label=str(sample_count))
    ax.set(
        xlabel="True reference FPR",
        ylabel="Mismatch declaration probability",
        title="Reference-mismatch evidence power",
    )
    ax.legend(title="n")
    return _save(fig, output)


def per_client_operating_points(frame: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots()
    for policy, group in frame.groupby("policy"):
        ax.plot(group["client_id"], group["fpr"], marker="o", label=str(policy))
    for level in (0.005, 0.01, 0.015):
        ax.axhline(level, linestyle="--")
    ax.set(xlabel="Client", ylabel="Final benign FPR", title="Per-client operating points")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    return _save(fig, output)


def reliability_utility_frontier(frame: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots()
    ax.scatter(frame["mebe"], frame["attack_balanced_macro_tpr"])
    for _, row in frame.iterrows():
        ax.annotate(str(row["policy"]), (row["mebe"], row["attack_balanced_macro_tpr"]))
    ax.set(xlabel="MEBE", ylabel="ABMacroTPR", title="Reliability–utility frontier")
    return _save(fig, output)


def calibration_phase_transition(frame: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots()
    ax.plot(frame["sample_count"], frame["admission_rate"], marker="o")
    ax.set(
        xlabel="Calibration/evidence sample size",
        ylabel="Admission rate",
        title="Calibration-size phase transition",
    )
    return _save(fig, output)


def assumption_stress(frame: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots()
    for condition, group in frame.groupby("condition"):
        ax.plot(group["value"], group["coverage"], marker="o", label=str(condition))
    ax.set(xlabel="Stress magnitude", ylabel="Observed coverage", title="Assumption stress")
    ax.legend()
    return _save(fig, output)


def external_replication(frame: pd.DataFrame, output: Path) -> Path:
    fig, ax = plt.subplots()
    for policy, group in frame.groupby("policy"):
        ax.plot(group["client_id"], group["fpr"], marker=".", label=str(policy))
    ax.set(xlabel="DIAD client", ylabel="Final benign FPR", title="External replication")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    return _save(fig, output)


def _save(fig: plt.Figure, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output
