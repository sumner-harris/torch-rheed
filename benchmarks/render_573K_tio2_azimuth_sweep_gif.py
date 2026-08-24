"""Render a fixed-glancing-angle azimuth sweep for 573 K TiO2/SrTiO3(001)."""

from __future__ import annotations

from dataclasses import replace
import math
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
INPUT_DIR = (
    DATA_DIR
    / "sto_termination_study"
    / "results"
    / "temperature_delivery_package_staging"
    / "573K"
    / "100"
    / "inputs"
)
OUTPUT = DATA_DIR / "renders" / "tio2_573K_100_azimuth_m10_p10_gi3p2_fwhm0p36585mm_linear_unclipped_laue_circle.gif"

AZIMUTH_START_DEG = -10.0
AZIMUTH_STOP_DEG = 10.0
AZIMUTH_STEP_DEG = 0.5
GLANCING_DISPLAY_DEG = 3.2
GLANCING_PROFILE_START_DEG = 0.1
GLANCING_PROFILE_STOP_DEG = 7.0
GLANCING_PROFILE_STEP_DEG = 0.10


def main() -> None:
    base_bulk = load_bulk_input(INPUT_DIR / "TiO2_bulk.txt")
    surface = load_surface_input(INPUT_DIR / "TiO2_surf.txt", base_bulk.ndom)
    scan_bulk = replace(
        base_bulk,
        azi_deg=AZIMUTH_START_DEG,
        azf_deg=AZIMUTH_STOP_DEG,
        daz_deg=AZIMUTH_STEP_DEG,
        gi_deg=GLANCING_PROFILE_START_DEG,
        gf_deg=GLANCING_PROFILE_STOP_DEG,
        dg_deg=GLANCING_PROFILE_STEP_DEG,
    )
    bulk = simulate_bulk(scan_bulk, device="cpu")
    result = simulate_rocking_curve(bulk, surface, screen_config=None, solver="sp6")

    azimuths = torch.linspace(
        AZIMUTH_START_DEG,
        AZIMUTH_STOP_DEG,
        bulk.naz,
        dtype=torch.float64,
        device=bulk.device,
    )
    glancing_angles = result.angles_deg.reshape(bulk.naz, bulk.ng)[0]
    intensities = result.intensities.reshape(bulk.naz, bulk.ng, len(result.beam_indices))

    config = ScreenImageConfig(
        plane_mode="vertical",
        screen_distance_mm=105.0,
        screen_width_mm=20.0,
        screen_height_mm=20.0,
        pixels_x=320,
        pixels_y=320,
        correlation_length_angstrom=1000.0,
        instrument_broadening_fwhm_mm=0.36585,
        reference_angle_deg=GLANCING_DISPLAY_DEG,
        frame_azimuth_deg=0.0,
        display_scale="linear",
        colormap="inferno",
    )

    specular_direction = _beam_direction(
        bulk,
        azimuth_deg=0.0,
        glancing_deg=GLANCING_DISPLAY_DEG,
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
    incident_k = _incident_wavevector_lab(bulk.wn, GLANCING_DISPLAY_DEG, device=bulk.device)
    qx_grid = scattered_kx - incident_k[0]
    qy_grid = scattered_ky - incident_k[1]
    qz_grid = scattered_kz - incident_k[2]
    ghx, ghy, gky = _reciprocal_steps(bulk)

    screen_stack = torch.zeros(
        (bulk.naz, config.pixels_y, config.pixels_x),
        dtype=torch.float32,
        device=bulk.device,
    )
    for azimuth_index, azimuth_tensor in enumerate(azimuths):
        azimuth_deg = float(azimuth_tensor.item())
        for beam_index, (ih, ik) in enumerate(result.beam_indices):
            profile = _beam_qz_profile(
                bulk,
                azimuth_deg=azimuth_deg,
                angles_deg=glancing_angles,
                beam_intensities=intensities[azimuth_index, :, beam_index][None, :],
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
            ).to(dtype=screen_stack.dtype)
            intensity_along_rod = _interp1d_batch(qz_grid, qz_axis, intensity_axis)[0].to(
                dtype=screen_stack.dtype
            )
            screen_stack[azimuth_index] += intensity_along_rod * rod_cross_section
        print(f"assembled azimuth {azimuth_deg:+.2f} deg", flush=True)

    visibility = _sample_surface_visibility_mask(scattered_kz).to(dtype=screen_stack.dtype)
    screen_stack *= visibility[None]
    screen_stack = _apply_instrument_broadening(
        screen_stack[None],
        fwhm_mm=config.instrument_broadening_fwhm_mm,
        screen_width_mm=config.screen_width_mm,
        screen_height_mm=config.screen_height_mm,
    )[0]
    screen_stack *= visibility[None]

    display_ceiling = float(torch.max(screen_stack).item())
    laue_circle_radius_mm = config.screen_distance_mm * math.tan(
        math.radians(GLANCING_DISPLAY_DEG)
    )
    forward_frames = []
    for azimuth_index, azimuth_tensor in enumerate(azimuths.cpu()):
        azimuth_deg = float(azimuth_tensor.item())
        scaled = _scale_screen_image(
            screen_stack[azimuth_index].cpu(),
            config.display_scale,
            maximum=display_ceiling,
        ).numpy()
        forward_frames.append(
            _render_screen_frame_image(
                scaled,
                config=config,
                angle_deg=GLANCING_DISPLAY_DEG,
                title=(
                    "TiO2-terminated SrTiO3(001), 573 K - SP6, 19 beams\n"
                    f"azimuth from [100] = {azimuth_deg:+.2f} deg, FWHM=0.366 mm, linear scale"
                ),
                default_prefix="RHEED azimuth sweep",
                laue_circle_radius_mm=laue_circle_radius_mm,
            )
        )

    gif_frames = list(forward_frames)
    gif_frames.extend(frame.copy() for frame in forward_frames[-2::-1])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        gif_frames[0].save(
            OUTPUT,
            save_all=True,
            append_images=gif_frames[1:],
            duration=100,
            loop=0,
            disposal=2,
        )
    finally:
        for frame in gif_frames:
            frame.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
