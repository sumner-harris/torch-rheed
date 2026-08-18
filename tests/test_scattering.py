from __future__ import annotations

import unittest

import torch

from torch_rheed.simulation import (
    _SP6_STAGES,
    _compose_scattering,
    _integrate_sp6_reflection,
    _potential_hermitian,
    _right_solve,
    _terminate_scattering,
    _trmat,
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


class SP6IntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.nb = 2
        self.iv = [[0, 0], [1, 0]]
        self.gma = torch.tensor([[[2.0 + 0.0j, 1.5 + 0.0j]]], dtype=torch.complex128)
        self.initial_reflection = torch.tensor(
            [[[[0.02 + 0.01j, 0.005 - 0.002j], [-0.003 + 0.001j, 0.01 - 0.004j]]]],
            dtype=torch.complex128,
        )
        self.v = torch.tensor([0.08 + 0.0j, 0.015 + 0.004j], dtype=torch.complex128)
        self.vi = torch.tensor([0.0 + 0.003j, 0.0 + 0.001j], dtype=torch.complex128)

    def _sp6_result(
        self,
        step_count: int,
        extent: float = 0.5,
        rhst_threshold: float = 1_000.0,
    ) -> torch.Tensor:
        sample_count = step_count * _SP6_STAGES + 1
        potential_h = _potential_hermitian(
            self.nb,
            self.v.expand(sample_count, -1),
            self.vi.expand(sample_count, -1),
            self.iv,
        ).unsqueeze(0)
        return _integrate_sp6_reflection(
            self.initial_reflection,
            self.gma,
            potential_h,
            torch.tensor([step_count]),
            torch.tensor([extent / step_count]),
            rhst_threshold=rhst_threshold,
        )

    def _exact_constant_result(self, step_count: int, extent: float = 0.5) -> torch.Tensor:
        reflection = self.initial_reflection.reshape(1, self.nb, self.nb).clone()
        transfer = _trmat(
            self.nb,
            self.gma.reshape(1, self.nb),
            self.v.unsqueeze(0),
            self.vi.unsqueeze(0),
            self.iv,
            extent / step_count,
        )
        scattering = _transfer_to_scattering(transfer)
        for _ in range(step_count):
            reflection = _terminate_scattering(scattering, reflection)
        return reflection.reshape_as(self.initial_reflection)

    def test_sp6_converges_at_sixth_order_for_constant_potential(self) -> None:
        coarse_error = (self._sp6_result(1) - self._exact_constant_result(1)).abs().amax()
        fine_error = (self._sp6_result(2) - self._exact_constant_result(2)).abs().amax()

        self.assertLess(float(fine_error), 3e-9)
        self.assertGreater(float(coarse_error / fine_error), 50.0)

    def test_sp6_matches_exact_constant_propagation(self) -> None:
        actual = self._sp6_result(4)
        expected = self._exact_constant_result(4)

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-9, atol=4e-11))

    def test_rhst_frequency_does_not_change_sp6_result(self) -> None:
        thresholded = self._sp6_result(4, rhst_threshold=1_000.0)
        every_step = self._sp6_result(4, rhst_threshold=1.000001)

        self.assertTrue(torch.allclose(thresholded, every_step, rtol=1e-12, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
