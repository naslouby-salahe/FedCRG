"""Generate the manuscript decision-architecture figure from the implemented protocol states."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def build_decision_architecture_figure(output: Path) -> Path:
    """Render the evidence-admission flow without using implementation shorthand names."""

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 8)
    axis.axis("off")

    boxes = (
        (0.5, 5.8, 2.3, 1.1, "Equal-count federation\nreference evidence"),
        (3.4, 5.8, 2.5, 1.1, "Independent client\nreference-mismatch evidence"),
        (6.5, 5.8, 2.5, 1.1, "Independent local\ncalibration readiness"),
        (9.6, 5.8, 1.9, 1.1, "Deployment\ndecision"),
        (0.7, 2.4, 2.5, 1.0, "Reference retained\n(no material mismatch demonstrated)"),
        (3.5, 2.4, 2.3, 1.0, "Mismatch evidence\ninsufficient"),
        (6.1, 2.4, 2.2, 1.0, "Calibration deficit"),
        (8.6, 2.4, 2.5, 1.0, "Calibration assumption\nviolation"),
        (5.0, 0.6, 2.4, 1.0, "Client-specific threshold\npersonalization admitted"),
    )

    for x, y, width, height, label in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04",
            linewidth=1.2,
            fill=False,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
        )

    def arrow(
        start: tuple[float, float], end: tuple[float, float], label: str | None = None
    ) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.1,
            )
        )
        if label is not None:
            midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            axis.text(midpoint[0], midpoint[1] + 0.15, label, ha="center", fontsize=8)

    arrow((2.8, 6.35), (3.4, 6.35))
    arrow((5.9, 6.35), (6.5, 6.35), "mismatch established")
    arrow((9.0, 6.35), (9.6, 6.35))
    arrow((4.6, 5.8), (1.95, 3.4), "no material mismatch demonstrated")
    arrow((4.6, 5.8), (4.65, 3.4), "sample too small")
    arrow((7.7, 5.8), (7.2, 3.4), "not ready")
    arrow((7.7, 5.8), (9.85, 3.4), "selected-score tie")
    arrow((10.55, 5.8), (6.2, 1.6), "mismatch + ready + unique threshold")

    axis.text(
        0.6,
        7.45,
        "Disjoint benign evidence roles: reference / mismatch / calibration",
        fontsize=10,
        fontweight="bold",
    )
    axis.text(
        0.6,
        7.05,
        "Attack labels are absent from admission and threshold construction.",
        fontsize=9,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output
