"""Helpers for reading and writing rocking-curve text outputs."""

from __future__ import annotations

from pathlib import Path

import torch

from .models import RockingCurveResult


def parse_surface_output_text(text: str) -> RockingCurveResult:
    """Parse the text contents of a ``surf-bulkE.s`` or ``surf-bulkP.s`` file."""

    lines = text.strip().splitlines()
    naz, ng, _ = [int(value) for value in lines[1].split()]
    beam_tokens = [token.strip() for token in lines[3].split(",")[1:] if token.strip()]
    beam_indices = [tuple(int(v) for v in token.split()) for token in beam_tokens]

    rows = []
    for line in lines[4:]:
        tokens = [token.strip() for token in line.split(",") if token.strip()]
        rows.append([float(token) for token in tokens])

    data = torch.tensor(rows, dtype=torch.float64)
    return RockingCurveResult(
        beam_indices=beam_indices,
        angles_deg=data[:, 0].clone(),
        intensities=data[:, 1:].clone(),
        naz=naz,
        ng=ng,
    )


def load_surface_output(path: Path) -> RockingCurveResult:
    """Read a `surf-bulkE.s` or `surf-bulkP.s` text file."""

    return parse_surface_output_text(path.read_text(encoding="utf-8"))
