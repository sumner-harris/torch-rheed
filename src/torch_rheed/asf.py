"""Atomic scattering-factor helpers bundled with the Python package."""

from __future__ import annotations

import torch

from .asf_data import AD, AP, BD, BP


def asfparam(iz: int, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the four-term scattering-factor coefficients for one element."""

    if 1 <= iz <= 98:
        ad = AD[iz - 1]
        if abs(ad[0]) > 1e-5:
            return (
                torch.tensor(ad, dtype=torch.float64, device=device),
                torch.tensor(BD[iz - 1], dtype=torch.float64, device=device),
            )

    abs_iz = abs(iz)
    if not 1 <= abs_iz <= 98:
        raise ValueError(f"|iz| must be between 1 and 98, got {iz}")

    peng_a = AP[abs_iz - 1]
    peng_b = BP[abs_iz - 1]
    return (
        torch.tensor(tuple(reversed(peng_a)), dtype=torch.float64, device=device),
        torch.tensor(tuple(reversed(peng_b)), dtype=torch.float64, device=device),
    )
