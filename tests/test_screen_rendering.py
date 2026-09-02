from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch

from torch_rheed.models import ScreenImageConfig
from torch_rheed.screen import (
    _apply_instrument_broadening,
    _beam_direction,
    _build_detector_scattered_k_grid,
    _detector_frame,
    _direct_beam_detector_coordinates,
    _sample_surface_visibility_mask,
    render_screen_stack,
)


class ScreenGeometryTests(unittest.TestCase):
    @staticmethod
    def _bulk() -> SimpleNamespace:
        return SimpleNamespace(
            source=SimpleNamespace(
                aa=4.0,
                bb=4.0,
                nh=1,
                nk=1,
                azi_deg=0.0,
            ),
            device=torch.device("cpu"),
            gam_rad=0.5 * math.pi,
            wn=20.0,
            naz=1,
        )

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

    def test_each_frame_uses_only_its_own_beam_intensity(self) -> None:
        angles = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64)
        intensities = torch.tensor([[[0.0], [1.0], [0.0]]], dtype=torch.float64)
        config = ScreenImageConfig(
            plane_mode="vertical",
            screen_distance_mm=100.0,
            screen_width_mm=40.0,
            screen_height_mm=40.0,
            pixels_x=101,
            pixels_y=101,
            correlation_length_angstrom=20.0,
            reference_angle_deg=5.0,
        )

        stack, _ = render_screen_stack(
            self._bulk(),
            [(0, 0)],
            angles,
            intensities,
            config=config,
        )

        self.assertEqual(float(torch.sum(stack[0, 0]).item()), 0.0)
        self.assertGreater(float(torch.sum(stack[0, 1]).item()), 0.0)
        self.assertEqual(float(torch.sum(stack[0, 2]).item()), 0.0)

    def test_single_frame_spot_maximum_matches_kinematic_beam_direction(self) -> None:
        bulk = self._bulk()
        angle_deg = 5.0
        config = ScreenImageConfig(
            plane_mode="vertical",
            screen_distance_mm=100.0,
            screen_width_mm=40.0,
            screen_height_mm=40.0,
            pixels_x=201,
            pixels_y=201,
            correlation_length_angstrom=20.0,
            reference_angle_deg=angle_deg,
        )
        angles = torch.tensor([angle_deg], dtype=torch.float64)
        intensities = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)

        stack, _ = render_screen_stack(
            bulk,
            [(0, 0), (0, 1)],
            angles,
            intensities,
            config=config,
        )

        direction = _beam_direction(
            bulk,
            azimuth_deg=0.0,
            glancing_deg=angle_deg,
            ih=0,
            ik=1,
        )
        self.assertIsNotNone(direction)
        specular_direction = _beam_direction(
            bulk,
            azimuth_deg=0.0,
            glancing_deg=angle_deg,
            ih=0,
            ik=0,
        )
        self.assertIsNotNone(specular_direction)
        origin, _, right, up = _detector_frame(
            specular_direction,
            plane_mode=config.plane_mode,
            distance_mm=config.screen_distance_mm,
        )
        scattered_k = _build_detector_scattered_k_grid(
            origin=origin,
            right=right,
            up=up,
            width_mm=config.screen_width_mm,
            height_mm=config.screen_height_mm,
            pixels_x=config.pixels_x,
            pixels_y=config.pixels_y,
            wn=bulk.wn,
        )
        expected_distance = sum(
            (component - bulk.wn * direction[index]).square()
            for index, component in enumerate(scattered_k)
        )
        expected_flat_index = int(torch.argmin(expected_distance).item())
        actual_flat_index = int(torch.argmax(stack[0, 0]).item())

        self.assertEqual(actual_flat_index, expected_flat_index)


if __name__ == "__main__":
    unittest.main()
