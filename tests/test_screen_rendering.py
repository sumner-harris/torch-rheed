from __future__ import annotations

import math
import unittest

import torch

from torch_rheed.models import ScreenImageConfig
from torch_rheed.screen import (
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


if __name__ == "__main__":
    unittest.main()
