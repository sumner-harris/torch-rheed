"""Render the measured-broadening TiO2-terminated SrTiO3 screen frame."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from torch_rheed.inputs import load_bulk_input, load_surface_input
from torch_rheed.models import ScreenImageConfig
from torch_rheed.screen import (
    _apply_instrument_broadening,
    _beam_direction,
    _beam_qz_profile,
    _build_detector_scattered_k_grid,
    _detector_frame,
    _incident_wavevector_lab,
    _interp1d_batch,
    _reciprocal_steps,
    _render_screen_frame_image,
    _rod_profile,
    _rotate_lab_frame,
    _sample_surface_visibility_mask,
    _scale_screen_image,
)
from torch_rheed.simulation import simulate_bulk, simulate_rocking_curve


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_DIR = DATA_DIR / "sto_termination_study" / "results" / "termination_comparison_100_110_973K" / "110" / "TiO2"
OUTPUT = DATA_DIR / "renders" / "tio2_terminated_sto_110_sp6_19beam_screen_105mm_xy10_broadening_1p4634mm.png"
ANGLE_DEG = 3.59


def main() -> None:
    bulk_input = load_bulk_input(INPUT_DIR / "bulk.txt")
    surface_input = load_surface_input(INPUT_DIR / "surf.txt", bulk_input.ndom)
    bulk = simulate_bulk(bulk_input, device="cpu")
    result = simulate_rocking_curve(bulk, surface_input, screen_config=None, solver="sp6")

    config = ScreenImageConfig(
        plane_mode="vertical",
        screen_distance_mm=105.0,
        screen_width_mm=20.0,
        screen_height_mm=20.0,
        pixels_x=512,
        pixels_y=512,
        correlation_length_angstrom=1000.0,
        instrument_broadening_fwhm_mm=1.4634,
        reference_angle_deg=ANGLE_DEG,
        display_scale="linear",
        colormap="inferno",
    )
    azimuth_deg = bulk.source.azi_deg
    specular_direction = _beam_direction(
        bulk,
        azimuth_deg=azimuth_deg,
        glancing_deg=ANGLE_DEG,
        ih=0,
        ik=0,
    )
    if specular_direction is None:
        raise RuntimeError("the specular beam is not propagating")

    origin, _, right, up = _detector_frame(
        specular_direction,
        plane_mode=config.plane_mode,
        distance_mm=config.screen_distance_mm,
    )
    scattered_kx, scattered_ky, scattered_kz = _build_detector_scattered_k_grid(
        origin=origin,
        right=right,
        up=up,
        width_mm=config.screen_width_mm,
        height_mm=config.screen_height_mm,
        pixels_x=config.pixels_x,
        pixels_y=config.pixels_y,
        wn=bulk.wn,
    )
    incident_k = _incident_wavevector_lab(bulk.wn, ANGLE_DEG, device=bulk.device)
    qx_grid = scattered_kx - incident_k[0]
    qy_grid = scattered_ky - incident_k[1]
    qz_grid = scattered_kz - incident_k[2]
    ghx, ghy, gky = _reciprocal_steps(bulk)

    frame = torch.zeros((1, 1, config.pixels_y, config.pixels_x), dtype=torch.float32, device=bulk.device)
    batched_intensities = result.intensities.unsqueeze(0)
    for beam_index, (ih, ik) in enumerate(result.beam_indices):
        profile = _beam_qz_profile(
            bulk,
            azimuth_deg=azimuth_deg,
            angles_deg=result.angles_deg,
            beam_intensities=batched_intensities[:, :, beam_index],
            ih=ih,
            ik=ik,
            intensity_floor=config.beam_intensity_floor,
        )
        if profile is None:
            continue
        qz_axis, intensity_axis = profile
        rod_center = _rotate_lab_frame(
            torch.tensor(
                [ghx * ih, ghy * ih + gky * ik, 0.0],
                dtype=torch.float64,
                device=bulk.device,
            ),
            azimuth_deg,
        )
        delta_q_parallel = torch.sqrt(
            (qx_grid - rod_center[0]).square() + (qy_grid - rod_center[1]).square()
        )
        rod_cross_section = _rod_profile(
            delta_q_parallel,
            correlation_length_angstrom=config.correlation_length_angstrom,
            profile=config.rod_profile,
        ).to(dtype=frame.dtype)
        intensity_along_rod = _interp1d_batch(qz_grid, qz_axis, intensity_axis).to(dtype=frame.dtype)
        frame[0, 0] += intensity_along_rod[0] * rod_cross_section

    visibility = _sample_surface_visibility_mask(scattered_kz).to(dtype=frame.dtype)
    frame *= visibility[None, None]
    frame = _apply_instrument_broadening(
        frame,
        fwhm_mm=config.instrument_broadening_fwhm_mm,
        screen_width_mm=config.screen_width_mm,
        screen_height_mm=config.screen_height_mm,
    )
    frame *= visibility[None, None]

    resolved_config = replace(config, frame_azimuth_deg=azimuth_deg)
    scaled = _scale_screen_image(frame[0, 0].cpu(), resolved_config.display_scale).numpy()
    image = _render_screen_frame_image(
        scaled,
        config=resolved_config,
        angle_deg=ANGLE_DEG,
        title=(
            "TiO2-terminated SrTiO3(001), beam along [110] - SP6, 19 beams\n"
            "15.8 kV, L=105 mm, instrument FWHM=1.463 mm"
        ),
        default_prefix="RHEED CTR screen",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
