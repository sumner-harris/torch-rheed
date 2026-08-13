"""Validation helpers for comparing standalone outputs to reference files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .models import RockingCurveResult
from .outputs import load_surface_output, parse_surface_output_text


@dataclass
class ValidationResult:
    """Comparison between a simulated result and a reference text output."""

    textual_match: bool
    max_abs_diff: float
    generated_text: str
    reference_text: str


def compare_to_reference(result: RockingCurveResult, reference_path: Path) -> ValidationResult:
    """Compare a simulated rocking curve against a reference surface-output file."""

    reference = load_surface_output(reference_path)
    result_angles = result.angles_deg.detach().cpu()
    reference_angles = reference.angles_deg.detach().cpu()
    if result.beam_indices != reference.beam_indices:
        raise ValueError("beam-index lists differ between the generated and reference results")
    if result_angles.shape != reference_angles.shape or not torch.allclose(
        result_angles,
        reference_angles,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("angle grids differ between the generated and reference results")

    generated_text = result.to_surface_output_text()
    reference_text = reference_path.read_text(encoding="utf-8")
    textual_match = generated_text.strip().splitlines() == reference_text.strip().splitlines()

    generated = parse_surface_output_text(generated_text)
    diff = torch.abs(generated.intensities - reference.intensities.detach().cpu())
    return ValidationResult(
        textual_match=textual_match,
        max_abs_diff=float(diff.max().item()),
        generated_text=generated_text,
        reference_text=reference_text,
    )
