from __future__ import annotations

import unittest

import torch

from torch_rheed.simulation import (
    _compose_scattering,
    _right_solve,
    _terminate_scattering,
    _transfer_to_scattering,
)


class ScatteringCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.nb = 3
        self.batch = 4
        identity = torch.eye(2 * self.nb, dtype=torch.complex128).expand(self.batch, -1, -1)
        self.lower_transfer = identity + 0.05 * torch.randn(
            self.batch,
            2 * self.nb,
            2 * self.nb,
            dtype=torch.complex128,
        )
        self.upper_transfer = identity + 0.05 * torch.randn(
            self.batch,
            2 * self.nb,
            2 * self.nb,
            dtype=torch.complex128,
        )

    def test_redheffer_composition_matches_transfer_product(self) -> None:
        direct = _transfer_to_scattering(self.upper_transfer @ self.lower_transfer)
        composed = _compose_scattering(
            _transfer_to_scattering(self.upper_transfer),
            _transfer_to_scattering(self.lower_transfer),
        )

        for direct_block, composed_block in zip(direct, composed, strict=True):
            self.assertTrue(torch.allclose(direct_block, composed_block, rtol=1e-12, atol=1e-12))

    def test_termination_matches_transfer_mobius_update(self) -> None:
        transfer = self.upper_transfer @ self.lower_transfer
        reflection = 0.05 * torch.randn(
            self.batch,
            self.nb,
            self.nb,
            dtype=torch.complex128,
        )
        top_left = transfer[:, : self.nb, : self.nb]
        top_right = transfer[:, : self.nb, self.nb :]
        bottom_left = transfer[:, self.nb :, : self.nb]
        bottom_right = transfer[:, self.nb :, self.nb :]
        expected = _right_solve(
            top_right + top_left @ reflection,
            bottom_right + bottom_left @ reflection,
        )
        actual = _terminate_scattering(_transfer_to_scattering(transfer), reflection)

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-12, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
