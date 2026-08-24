"""Plotting utilities for rocking-curve simulation results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .models import RockingCurveResult
from .outputs import load_surface_output
from .screen import plot_screen_frame


def _cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached CPU copy for plotting and comparison."""

    return tensor.detach().cpu()


def plot_rocking_curve(
    result: RockingCurveResult,
    output_path: Path,
    *,
    title: str | None = None,
    beam_indices: list[tuple[int, int]] | None = None,
) -> None:
    """Plot selected rocking-curve beams, or all beams when not specified."""

    selected_beams = result.beam_indices if beam_indices is None else beam_indices
    if not selected_beams:
        raise ValueError("beam_indices must contain at least one beam")
    missing = [beam for beam in selected_beams if beam not in result.beam_indices]
    if missing:
        raise ValueError(f"requested plot beams are absent from the result: {missing}")

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(selected_beams), 3)))
    angles_deg = _cpu_tensor(result.angles_deg)
    intensities = _cpu_tensor(result.intensities)
    for color_index, (ih, ik) in enumerate(selected_beams):
        result_index = result.beam_indices.index((ih, ik))
        ax.plot(
            angles_deg.tolist(),
            intensities[:, result_index].tolist(),
            color=colors[color_index % len(colors)],
            linewidth=2.0,
            label=f"{ih} {ik}",
        )

    ax.set_xlabel("Glancing angle (deg)")
    ax.set_ylabel("Intensity")
    ax.set_title(title or "RHEED rocking curve")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=min(3, len(selected_beams)), fontsize=9)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_surface_output_file(
    input_path: Path,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot a saved ``surf-bulkE.s`` or ``surf-bulkP.s`` text file."""

    plot_rocking_curve(load_surface_output(input_path), output_path, title=title)


def plot_reference_comparison(
    reference: RockingCurveResult,
    generated: RockingCurveResult,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot a simulated rocking curve against a reference result."""

    if reference.beam_indices != generated.beam_indices:
        raise ValueError("beam-index lists differ between the reference and generated results")
    reference_angles = _cpu_tensor(reference.angles_deg)
    generated_angles = _cpu_tensor(generated.angles_deg)
    if reference_angles.shape != generated_angles.shape or not torch.allclose(
        reference_angles,
        generated_angles,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("angle grids differ between the reference and generated results")

    reference_intensities = _cpu_tensor(reference.intensities)
    generated_intensities = _cpu_tensor(generated.intensities)
    diff = generated_intensities - reference_intensities
    max_abs_diff = float(torch.max(torch.abs(diff)).item())

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    ax_top, ax_bottom = axes
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(reference.beam_indices), 3)))

    angles = reference_angles.tolist()
    for idx, (ih, ik) in enumerate(reference.beam_indices):
        color = colors[idx % len(colors)]
        label = f"{ih} {ik}"
        ax_top.plot(
            angles,
            reference_intensities[:, idx].tolist(),
            color=color,
            linewidth=2.0,
            label=f"{label} reference",
        )
        ax_top.plot(
            angles,
            generated_intensities[:, idx].tolist(),
            color=color,
            linewidth=1.6,
            linestyle="--",
            label=f"{label} torch_rheed",
        )
        ax_bottom.plot(
            angles,
            diff[:, idx].tolist(),
            color=color,
            linewidth=1.6,
            label=label,
        )

    ax_top.set_title(title or "Reference comparison")
    ax_top.set_ylabel("Intensity")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(ncol=2, fontsize=8)
    ax_top.text(
        0.99,
        0.98,
        f"max |torch_rheed - ref| = {max_abs_diff:.1e}",
        transform=ax_top.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.85"},
    )

    ax_bottom.axhline(0.0, color="black", linewidth=1.0, alpha=0.8)
    ax_bottom.set_xlabel("Glancing angle (deg)")
    ax_bottom.set_ylabel("Diff")
    ax_bottom.grid(True, alpha=0.25)
    if max_abs_diff == 0.0:
        ax_bottom.set_ylim(-1e-15, 1e-15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
