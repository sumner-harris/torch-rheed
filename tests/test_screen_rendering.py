from __future__ import annotations

import math
import unittest

import torch

from torch_rheed.models import ScreenImageConfig
from torch_rheed.screen import (
    _apply_instrument_broadening,
    _detector_frame,
    _direct_beam_detector_coordinates,
    _sample_surface_visibility_mask,
)


class ScreenGeometryTests(unittest.TestCase):
    def test_vertical_detector_origin_is_on_the_sample_surface(self) -> None:
        angle_rad = math.radians(3.59)
        specular = torch.tensor(
            [math.cos(angle_rad), 0.0, math.sin(angle_rad)],
            dtype=torch.float64,
        )

        origin, normal, right, up = _detector_frame(
            specular,
            plane_mode="vertical",
            distance_mm=105.0,
        )

        self.assertTrue(torch.allclose(origin, torch.tensor([105.0, 0.0, 0.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(normal, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(right, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)))
        self.assertTrue(torch.allclose(up, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)))

    def test_direct_beam_is_below_the_surface_line(self) -> None:
        config = ScreenImageConfig(
            plane_mode="vertical",
            screen_distance_mm=105.0,
            reference_angle_deg=3.59,
        )

        x_mm, y_mm = _direct_beam_detector_coordinates(config, 3.59)

        self.assertAlmostEqual(x_mm, 0.0, places=12)
        self.assertAlmostEqual(y_mm, -105.0 * math.tan(math.radians(3.59)), places=12)

    def test_substrate_mask_keeps_only_vacuum_side_pixels(self) -> None:
        scattered_kz = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float64)

        actual = _sample_surface_visibility_mask(scattered_kz)

        self.assertTrue(torch.equal(actual, torch.tensor([[True, True, False]])))

    def test_instrument_broadening_uses_physical_detector_units(self) -> None:
        image = torch.zeros((1, 1, 81, 101), dtype=torch.float64)
        image[0, 0, 40, 50] = 1.0

        broadened = _apply_instrument_broadening(
            image,
            fwhm_mm=1.0,
            screen_width_mm=20.2,
            screen_height_mm=8.1,
        )[0, 0]

        x_profile = torch.sum(broadened, dim=0)
        y_profile = torch.sum(broadened, dim=1)
        x_coordinates_mm = (torch.arange(101, dtype=torch.float64) - 50.0) * 0.2
        y_coordinates_mm = (torch.arange(81, dtype=torch.float64) - 40.0) * 0.1
        sigma_x_mm = torch.sqrt(torch.sum(x_profile * x_coordinates_mm.square()) / torch.sum(x_profile))
        sigma_y_mm = torch.sqrt(torch.sum(y_profile * y_coordinates_mm.square()) / torch.sum(y_profile))
        expected_sigma_mm = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

        self.assertAlmostEqual(float(sigma_x_mm.item()), expected_sigma_mm, places=3)
        self.assertAlmostEqual(float(sigma_y_mm.item()), expected_sigma_mm, places=3)

    def test_zero_instrument_broadening_leaves_stack_unchanged(self) -> None:
        image = torch.rand((2, 3, 7, 9), dtype=torch.float32)

        broadened = _apply_instrument_broadening(
            image,
            fwhm_mm=0.0,
            screen_width_mm=9.0,
            screen_height_mm=7.0,
        )

        self.assertIs(broadened, image)


if __name__ == "__main__":
    unittest.main()
