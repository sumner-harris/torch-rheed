"""Render a broadened TiO2-terminated SrTiO3 rocking-curve screen GIF."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from torch_rheed.inputs import load_bulk_input, load_surface_input
from torch_rheed.models import RockingCurveResult, ScreenImageConfig
from torch_rheed.screen import (
    _apply_instrument_broadening,
    _beam_direction,
    _beam_qz_profile,
    _build_detector_scattered_k_grid,
    _detector_frame,
    _incident_wavevector_lab,
    _interp1d_batch,
    _reciprocal_steps,
    _rod_profile,
    _rotate_lab_frame,
    _sample_surface_visibility_mask,
    write_screen_gif,
)
from torch_rheed.simulation import simulate_bulk, simulate_rocking_curve


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INPUT_DIR = (
    DATA_DIR
    / "sto_termination_study"
    / "results"
    / "one_beam_off_azimuth_12p25"
    / "ideal_single_tio2"
    / "19beam"
)
OUTPUT = DATA_DIR / "renders" / "tio2_terminated_sto_100_plus12p25_sp6_19beam_screen_105mm_xy10_broadening_0p7317mm_linear_vmax10pct.gif"


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
        pixels_x=320,
        pixels_y=320,
        correlation_length_angstrom=1000.0,
        instrument_broadening_fwhm_mm=0.7317,
        reference_angle_deg=3.59,
        display_scale="linear",
        colormap="inferno",
    )

    # Match the earlier GIF cadence: every 0.20 degrees plus the final frame.
    selected_indices = torch.arange(0, result.angles_deg.numel(), 20, device=result.angles_deg.device)
    last_index = result.angles_deg.numel() - 1
    if int(selected_indices[-1].item()) != last_index:
        selected_indices = torch.cat(
            [selected_indices, torch.tensor([last_index], device=selected_indices.device)]
        )
    selected_angles = result.angles_deg.index_select(0, selected_indices)

    azimuth_deg = bulk.source.azi_deg
    specular_direction = _beam_direction(
        bulk,
        azimuth_deg=azimuth_deg,
        glancing_deg=config.reference_angle_deg,
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
    ghx, ghy, gky = _reciprocal_steps(bulk)
    qz_profiles = [
        _beam_qz_profile(
            bulk,
            azimuth_deg=azimuth_deg,
            angles_deg=result.angles_deg,
            beam_intensities=result.intensities[None, :, beam_index],
            ih=ih,
            ik=ik,
            intensity_floor=config.beam_intensity_floor,
        )
        for beam_index, (ih, ik) in enumerate(result.beam_indices)
    ]

    frame_stack = torch.zeros(
        (selected_angles.numel(), config.pixels_y, config.pixels_x),
        dtype=torch.float32,
        device=bulk.device,
    )
    chunk_size = 4
    for start in range(0, selected_angles.numel(), chunk_size):
        stop = min(start + chunk_size, selected_angles.numel())
        incident_vectors = torch.stack(
            [
                _incident_wavevector_lab(bulk.wn, float(angle.item()), device=bulk.device)
                for angle in selected_angles[start:stop]
            ]
        )
        qx_grid = scattered_kx[None] - incident_vectors[:, 0, None, None]
        qy_grid = scattered_ky[None] - incident_vectors[:, 1, None, None]
        qz_grid = scattered_kz[None] - incident_vectors[:, 2, None, None]

        for beam_index, (ih, ik) in enumerate(result.beam_indices):
            profile = qz_profiles[beam_index]
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
            ).to(dtype=frame_stack.dtype)
            intensity_along_rod = _interp1d_batch(qz_grid, qz_axis, intensity_axis)[0].to(
                dtype=frame_stack.dtype
            )
            frame_stack[start:stop] += intensity_along_rod * rod_cross_section

    visibility = _sample_surface_visibility_mask(scattered_kz).to(dtype=frame_stack.dtype)
    frame_stack *= visibility[None]
    frame_stack = _apply_instrument_broadening(
        frame_stack[None],
        fwhm_mm=config.instrument_broadening_fwhm_mm,
        screen_width_mm=config.screen_width_mm,
        screen_height_mm=config.screen_height_mm,
    )[0]
    frame_stack *= visibility[None]
    display_ceiling = 0.10 * torch.max(frame_stack)
    display_stack = torch.clamp(frame_stack, max=display_ceiling)

    gif_result = RockingCurveResult(
        beam_indices=result.beam_indices,
        angles_deg=selected_angles,
        intensities=result.intensities.index_select(0, selected_indices),
        naz=result.naz,
        ng=selected_angles.numel(),
        screen_images=display_stack,
        screen_config=replace(config, frame_azimuth_deg=azimuth_deg),
    )
    write_screen_gif(
        gif_result,
        OUTPUT,
        frame_step=1,
        frame_duration_ms=100,
        normalize="global",
        boomerang=True,
        title=(
            "TiO2-terminated SrTiO3(001), beam along [100]+12.25 deg - SP6, 19 beams\n"
            "15.8 kV, L=105 mm, FWHM=0.732 mm, linear scale, vmax=10%"
        ),
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
