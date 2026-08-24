"""Synthetic detector-image generation for standalone ``torch_rheed`` results."""

from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models import BulkSimulation, RockingCurveBatchResult, RockingCurveResult, ScreenImageConfig


EK = 0.262466
C2M = 511.001
TWOPI = 2.0 * math.pi


DEFAULT_SCREEN_IMAGE_CONFIG = ScreenImageConfig()


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    """Return a normalized copy of one 3-vector."""

    norm = torch.linalg.norm(vector)
    if float(norm.item()) <= 0.0:
        raise ValueError("cannot normalize a zero-length detector vector")
    return vector / norm


def _rotate_lab_frame(direction: torch.Tensor, azimuth_deg: float) -> torch.Tensor:
    """Rotate a crystal-frame beam direction into the fixed laboratory frame."""

    az = math.radians(azimuth_deg)
    cos_az = math.cos(az)
    sin_az = math.sin(az)
    x_value = float(direction[0].item())
    y_value = float(direction[1].item())
    return torch.tensor(
        [
            cos_az * x_value + sin_az * y_value,
            -sin_az * x_value + cos_az * y_value,
            float(direction[2].item()),
        ],
        dtype=direction.dtype,
        device=direction.device,
    )


def _reciprocal_steps(bulk: BulkSimulation) -> tuple[float, float, float]:
    """Return the in-plane reciprocal-lattice step vectors for one bulk cell."""

    ghx = TWOPI / (bulk.source.aa * bulk.source.nh)
    ghy = -TWOPI / (bulk.source.aa * math.tan(bulk.gam_rad) * bulk.source.nh)
    gky = TWOPI / (bulk.source.bb * math.sin(bulk.gam_rad) * bulk.source.nk)
    return ghx, ghy, gky


def _beam_direction(
    bulk: BulkSimulation,
    *,
    azimuth_deg: float,
    glancing_deg: float,
    ih: int,
    ik: int,
) -> torch.Tensor | None:
    """Return the outgoing beam direction for one open diffraction beam."""

    ghx, ghy, gky = _reciprocal_steps(bulk)

    az = math.radians(azimuth_deg)
    ga = math.radians(glancing_deg)
    cos_ga = math.cos(ga)
    wnx = bulk.wn * cos_ga * math.cos(az)
    wny = bulk.wn * cos_ga * math.sin(az)
    wgx = ghx * ih + wnx
    wgy = ghy * ih + gky * ik + wny
    sval = bulk.wn * bulk.wn - wgx * wgx - wgy * wgy
    if sval <= 0.0:
        return None

    kz = math.sqrt(sval)
    crystal_direction = torch.tensor([wgx, wgy, kz], dtype=torch.float64, device=bulk.device) / bulk.wn
    return _normalize(_rotate_lab_frame(crystal_direction, azimuth_deg))


def _detector_frame(
    specular_direction: torch.Tensor,
    *,
    plane_mode: str,
    distance_mm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the detector's sample-plane origin and orthonormal basis."""

    up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=specular_direction.device)

    if plane_mode == "vertical":
        horizontal = torch.tensor(
            [float(specular_direction[0].item()), float(specular_direction[1].item()), 0.0],
            dtype=torch.float64,
            device=specular_direction.device,
        )
        normal = _normalize(horizontal)
        right = _normalize(torch.linalg.cross(up, normal))
        origin = normal * distance_mm
        return origin, normal, right, up

    normal = _normalize(specular_direction)
    right = torch.linalg.cross(up, normal)
    if float(torch.linalg.norm(right).item()) < 1.0e-12:
        right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=specular_direction.device)
    right = _normalize(right)
    up_axis = _normalize(torch.linalg.cross(normal, right))
    plane_anchor = normal * distance_mm
    surface_direction = torch.tensor(
        [float(specular_direction[0].item()), float(specular_direction[1].item()), 0.0],
        dtype=torch.float64,
        device=specular_direction.device,
    )
    surface_direction = _normalize(surface_direction)
    scale = float(torch.dot(plane_anchor, normal).item() / torch.dot(surface_direction, normal).item())
    origin = surface_direction * scale
    return origin, normal, right, up_axis


def _direct_beam_detector_coordinates(
    config: ScreenImageConfig,
    glancing_deg: float,
) -> tuple[float, float]:
    """Return the direct-beam intersection in sample-surface detector coordinates."""

    reference_deg = glancing_deg if config.reference_angle_deg is None else config.reference_angle_deg
    reference_rad = math.radians(reference_deg)
    specular_direction = torch.tensor(
        [math.cos(reference_rad), 0.0, math.sin(reference_rad)],
        dtype=torch.float64,
    )
    origin, normal, right, up = _detector_frame(
        specular_direction,
        plane_mode=config.plane_mode,
        distance_mm=config.screen_distance_mm,
    )

    glancing_rad = math.radians(glancing_deg)
    direct_direction = torch.tensor(
        [math.cos(glancing_rad), 0.0, -math.sin(glancing_rad)],
        dtype=torch.float64,
    )
    denominator = float(torch.dot(direct_direction, normal).item())
    if abs(denominator) < 1.0e-12:
        return float("nan"), float("nan")
    distance_along_ray = float(torch.dot(origin, normal).item()) / denominator
    intersection = direct_direction * distance_along_ray
    relative = intersection - origin
    return float(torch.dot(relative, right).item()), float(torch.dot(relative, up).item())


def _sample_surface_visibility_mask(scattered_kz: torch.Tensor) -> torch.Tensor:
    """Return the vacuum-side detector pixels not occluded by the substrate."""

    return scattered_kz >= 0.0


def _reference_angle(config: ScreenImageConfig, beam_indices: list[tuple[int, int]], angles_deg: torch.Tensor, intensities: torch.Tensor) -> float:
    """Choose the detector-centering angle for a result batch."""

    if config.reference_angle_deg is not None:
        return config.reference_angle_deg

    zero_zero_index = next(index for index, beam in enumerate(beam_indices) if beam == (0, 0))
    peak_index = int(torch.argmax(intensities[0, :, zero_zero_index]).item())
    return float(angles_deg[peak_index].item())


def _incident_wavevector_lab(wn: float, glancing_deg: float, *, device: torch.device) -> torch.Tensor:
    """Return the lab-frame incident wavevector for one glancing angle."""

    ga = math.radians(glancing_deg)
    cos_ga = math.cos(ga)
    return torch.tensor(
        [wn * cos_ga, 0.0, -wn * math.sin(ga)],
        dtype=torch.float64,
        device=device,
    )


def _build_detector_scattered_k_grid(
    *,
    origin: torch.Tensor,
    right: torch.Tensor,
    up: torch.Tensor,
    width_mm: float,
    height_mm: float,
    pixels_x: int,
    pixels_y: int,
    wn: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the fixed detector ``k_f`` grid for one detector plane."""

    x_centers_mm = torch.linspace(
        -0.5 * width_mm + 0.5 * width_mm / pixels_x,
        0.5 * width_mm - 0.5 * width_mm / pixels_x,
        pixels_x,
        dtype=torch.float64,
        device=origin.device,
    )
    y_centers_mm = torch.linspace(
        0.5 * height_mm - 0.5 * height_mm / pixels_y,
        -0.5 * height_mm + 0.5 * height_mm / pixels_y,
        pixels_y,
        dtype=torch.float64,
        device=origin.device,
    )
    y_grid_mm, x_grid_mm = torch.meshgrid(y_centers_mm, x_centers_mm, indexing="ij")
    detector_points = (
        origin[None, None, :]
        + x_grid_mm[:, :, None] * right[None, None, :]
        + y_grid_mm[:, :, None] * up[None, None, :]
    )
    directions = detector_points / torch.linalg.norm(detector_points, dim=2, keepdim=True)
    scattered_k = wn * directions
    return scattered_k[:, :, 0], scattered_k[:, :, 1], scattered_k[:, :, 2]


def _beam_qz_profile(
    bulk: BulkSimulation,
    *,
    azimuth_deg: float,
    angles_deg: torch.Tensor,
    beam_intensities: torch.Tensor,
    ih: int,
    ik: int,
    intensity_floor: float,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the shared ``q_z`` axis and batched intensities for one CTR rod."""

    qz_values: list[torch.Tensor] = []
    intensity_columns: list[torch.Tensor] = []

    for angle_index in range(angles_deg.numel()):
        intensity_column = beam_intensities[:, angle_index]
        if float(torch.max(intensity_column).item()) <= intensity_floor:
            continue
        angle_deg = float(angles_deg[angle_index].item())
        direction = _beam_direction(
            bulk,
            azimuth_deg=azimuth_deg,
            glancing_deg=angle_deg,
            ih=ih,
            ik=ik,
        )
        if direction is None:
            continue
        qz_value = bulk.wn * direction[2] + bulk.wn * math.sin(math.radians(angle_deg))
        qz_values.append(qz_value.to(dtype=torch.float64))
        intensity_columns.append(intensity_column)

    if len(qz_values) < 2:
        return None

    qz_tensor = torch.stack(qz_values, dim=0)
    intensity_tensor = torch.stack(intensity_columns, dim=1)
    qz_sorted, order = torch.sort(qz_tensor)
    intensity_sorted = intensity_tensor.index_select(1, order)
    keep = torch.ones_like(qz_sorted, dtype=torch.bool)
    keep[1:] = qz_sorted[1:] != qz_sorted[:-1]
    qz_unique = qz_sorted[keep]
    intensity_unique = intensity_sorted[:, keep]
    if qz_unique.numel() < 2:
        return None
    return qz_unique, intensity_unique


def _interp1d_batch(query: torch.Tensor, axis: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Linearly interpolate ``values`` sampled on ``axis`` onto ``query`` for every batch row."""

    flat_query = query.reshape(-1)
    result = torch.zeros((values.shape[0], flat_query.numel()), dtype=values.dtype, device=values.device)

    if axis.numel() < 2:
        return result.reshape(values.shape[0], *query.shape)

    lower_equal = flat_query == axis[0]
    if bool(torch.any(lower_equal).item()):
        result[:, lower_equal] = values[:, :1]

    indices = torch.searchsorted(axis, flat_query)
    inside = (indices > 0) & (indices < axis.numel())
    if bool(torch.any(inside).item()):
        idx1 = indices[inside]
        idx0 = idx1 - 1
        x0 = axis[idx0]
        x1 = axis[idx1]
        denominator = torch.where(torch.abs(x1 - x0) > 0.0, x1 - x0, torch.ones_like(x1))
        weight = ((flat_query[inside] - x0) / denominator).to(dtype=values.dtype)
        y0 = values.index_select(1, idx0)
        y1 = values.index_select(1, idx1)
        result[:, inside] = y0 + (y1 - y0) * weight.unsqueeze(0)

    return result.reshape(values.shape[0], *query.shape)


def _source_divergence_quadrature(config: ScreenImageConfig, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return glancing-angle source-divergence offsets and normalized Gaussian weights."""

    fwhm_deg = config.source_glancing_divergence_fwhm_deg
    samples = config.source_divergence_samples
    if fwhm_deg <= 0.0 or samples <= 1:
        return (
            torch.zeros(1, dtype=torch.float64, device=device),
            torch.ones(1, dtype=torch.float64, device=device),
        )
    if samples < 1:
        raise ValueError("source_divergence_samples must be positive")

    sigma_deg = fwhm_deg / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    span_sigma = 3.0
    offsets_deg = torch.linspace(
        -span_sigma * sigma_deg,
        span_sigma * sigma_deg,
        samples,
        dtype=torch.float64,
        device=device,
    )
    weights = torch.exp(-0.5 * (offsets_deg / sigma_deg).square())
    weights = weights / torch.sum(weights)
    return offsets_deg, weights


def _rod_profile(delta_q_parallel: torch.Tensor, *, correlation_length_angstrom: float, profile: str) -> torch.Tensor:
    """Return the reciprocal-space truncation-rod cross-section for one beam."""

    scaled = delta_q_parallel * correlation_length_angstrom
    if profile == "lorentzian":
        return 1.0 / (1.0 + scaled.square())
    if profile == "lorentzian2":
        base = 1.0 + scaled.square()
        return 1.0 / base.square()
    raise ValueError(f"unsupported rod profile: {profile}")


def _gaussian_kernel_1d(
    sigma_pixels: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    truncate: float = 4.0,
) -> torch.Tensor:
    """Return a normalized one-dimensional Gaussian convolution kernel."""

    radius = max(1, math.ceil(truncate * sigma_pixels))
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (coordinates / sigma_pixels).square())
    return kernel / torch.sum(kernel)


def _apply_instrument_broadening(
    screen_stack: torch.Tensor,
    *,
    fwhm_mm: float,
    screen_width_mm: float,
    screen_height_mm: float,
) -> torch.Tensor:
    """Apply an isotropic physical-space Gaussian PSF to a ``(B,N,H,W)`` stack."""

    if fwhm_mm <= 0.0:
        return screen_stack

    _, _, pixels_y, pixels_x = screen_stack.shape
    sigma_mm = fwhm_mm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    sigma_x_pixels = sigma_mm / (screen_width_mm / pixels_x)
    sigma_y_pixels = sigma_mm / (screen_height_mm / pixels_y)
    flattened = screen_stack.reshape(-1, 1, pixels_y, pixels_x)

    kernel_x = _gaussian_kernel_1d(
        sigma_x_pixels,
        dtype=screen_stack.dtype,
        device=screen_stack.device,
    )
    radius_x = kernel_x.numel() // 2
    broadened = F.conv2d(flattened, kernel_x.reshape(1, 1, 1, -1), padding=(0, radius_x))

    kernel_y = _gaussian_kernel_1d(
        sigma_y_pixels,
        dtype=screen_stack.dtype,
        device=screen_stack.device,
    )
    radius_y = kernel_y.numel() // 2
    broadened = F.conv2d(broadened, kernel_y.reshape(1, 1, -1, 1), padding=(radius_y, 0))
    return broadened.reshape_as(screen_stack)


def _render_screen_stack_ctr(
    bulk: BulkSimulation,
    beam_indices: list[tuple[int, int]],
    angles_deg: torch.Tensor,
    intensities: torch.Tensor,
    *,
    config: ScreenImageConfig,
) -> tuple[torch.Tensor, ScreenImageConfig]:
    """Render the fixed-angle CTR/coherence-length detector-image stack."""

    if config.correlation_length_angstrom <= 0.0:
        raise ValueError("correlation_length_angstrom must be positive for CTR screen rendering")

    azimuth_deg = bulk.source.azi_deg
    frame_azimuth_deg = azimuth_deg if config.frame_azimuth_deg is None else config.frame_azimuth_deg
    reference_angle_deg = _reference_angle(config, beam_indices, angles_deg, intensities)
    specular_direction = _beam_direction(
        bulk,
        azimuth_deg=frame_azimuth_deg,
        glancing_deg=reference_angle_deg,
        ih=0,
        ik=0,
    )
    if specular_direction is None:
        raise ValueError(f"00 beam is non-propagating at detector reference angle {reference_angle_deg:.6f} deg")

    origin, normal, right, up = _detector_frame(
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
            angles_deg=angles_deg,
            beam_intensities=intensities[:, :, beam_index],
            ih=ih,
            ik=ik,
            intensity_floor=config.beam_intensity_floor,
        )
        for beam_index, (ih, ik) in enumerate(beam_indices)
    ]

    batch_size, n_angles, _ = intensities.shape
    image_dtype = torch.float32
    screen_stack = torch.zeros(
        (batch_size, n_angles, config.pixels_y, config.pixels_x),
        dtype=image_dtype,
        device=bulk.device,
    )
    divergence_offsets_deg, divergence_weights = _source_divergence_quadrature(config, device=bulk.device)

    for angle_index in range(n_angles):
        angle_deg = float(angles_deg[angle_index].item())
        for divergence_offset_deg, divergence_weight in zip(divergence_offsets_deg, divergence_weights, strict=True):
            sample_angle_deg = angle_deg + float(divergence_offset_deg.item())
            incident_k = _incident_wavevector_lab(bulk.wn, sample_angle_deg, device=bulk.device)
            qx_grid = scattered_kx - incident_k[0]
            qy_grid = scattered_ky - incident_k[1]
            qz_grid = scattered_kz - incident_k[2]

            for beam_index, (ih, ik) in enumerate(beam_indices):
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
                delta_q_parallel = torch.sqrt((qx_grid - rod_center[0]).square() + (qy_grid - rod_center[1]).square())
                rod_cross_section = _rod_profile(
                    delta_q_parallel,
                    correlation_length_angstrom=config.correlation_length_angstrom,
                    profile=config.rod_profile,
                ).to(dtype=image_dtype)
                intensity_along_rod = _interp1d_batch(qz_grid, qz_axis, intensity_axis).to(dtype=image_dtype)
                screen_stack[:, angle_index, :, :] += (
                    float(divergence_weight.item())
                    * intensity_along_rod
                    * rod_cross_section[None, :, :]
                )

    visibility = _sample_surface_visibility_mask(scattered_kz).to(dtype=image_dtype)
    screen_stack = screen_stack * visibility[None, None, :, :]
    screen_stack = _apply_instrument_broadening(
        screen_stack,
        fwhm_mm=config.instrument_broadening_fwhm_mm,
        screen_width_mm=config.screen_width_mm,
        screen_height_mm=config.screen_height_mm,
    )
    # Preserve the requested hard substrate mask after the detector PSF is applied.
    screen_stack = screen_stack * visibility[None, None, :, :]

    return screen_stack, ScreenImageConfig(
        plane_mode=config.plane_mode,
        screen_distance_mm=config.screen_distance_mm,
        screen_width_mm=config.screen_width_mm,
        screen_height_mm=config.screen_height_mm,
        pixels_x=config.pixels_x,
        pixels_y=config.pixels_y,
        correlation_length_angstrom=config.correlation_length_angstrom,
        rod_profile=config.rod_profile,
        beam_intensity_floor=config.beam_intensity_floor,
        instrument_broadening_fwhm_mm=config.instrument_broadening_fwhm_mm,
        source_glancing_divergence_fwhm_deg=config.source_glancing_divergence_fwhm_deg,
        source_divergence_samples=config.source_divergence_samples,
        reference_angle_deg=reference_angle_deg,
        frame_azimuth_deg=config.frame_azimuth_deg,
        display_scale=config.display_scale,
        colormap=config.colormap,
    )


def render_screen_stack(
    bulk: BulkSimulation,
    beam_indices: list[tuple[int, int]],
    angles_deg: torch.Tensor,
    intensities: torch.Tensor,
    *,
    config: ScreenImageConfig | None = None,
) -> tuple[torch.Tensor, ScreenImageConfig]:
    """Render a batched CTR detector-image stack shaped ``(B,N,H,W)``."""

    if bulk.naz != 1:
        raise NotImplementedError("screen-image generation currently requires naz=1 so each frame is a glancing-angle sweep at fixed azimuth")
    if intensities.ndim != 3:
        raise ValueError(f"expected batched intensities shaped (B,N,NB), got {tuple(intensities.shape)}")
    if intensities.shape[1] != angles_deg.numel():
        raise ValueError("screen rendering requires one intensity row per glancing-angle sample")
    if angles_deg.numel() < 2:
        raise ValueError(
            "CTR screen-image generation requires at least two glancing-angle samples; "
            "use a local rocking-curve window even when you want one nominal detector frame"
        )

    resolved_config = config or DEFAULT_SCREEN_IMAGE_CONFIG
    if resolved_config.pixels_x < 1 or resolved_config.pixels_y < 1:
        raise ValueError("screen-image pixel dimensions must both be positive")
    if resolved_config.plane_mode not in {"vertical", "specular-normal"}:
        raise ValueError(f"unsupported detector plane mode: {resolved_config.plane_mode}")
    if resolved_config.source_glancing_divergence_fwhm_deg < 0.0:
        raise ValueError("source_glancing_divergence_fwhm_deg must be non-negative")
    if resolved_config.instrument_broadening_fwhm_mm < 0.0:
        raise ValueError("instrument_broadening_fwhm_mm must be non-negative")
    if resolved_config.source_divergence_samples < 1:
        raise ValueError("source_divergence_samples must be at least 1")
    return _render_screen_stack_ctr(
        bulk,
        beam_indices,
        angles_deg,
        intensities,
        config=resolved_config,
    )


def _scale_screen_image(image: torch.Tensor, mode: str, maximum: float | None = None) -> torch.Tensor:
    """Apply a display-oriented scaling to one raw screen frame."""

    if maximum is None:
        maximum = float(torch.max(image).item())
    if maximum <= 0.0:
        return torch.zeros_like(image)
    normalized = image / maximum
    if mode == "linear":
        return normalized
    if mode == "sqrt":
        return torch.sqrt(normalized)
    if mode == "log":
        scale = math.log1p(1.0e3)
        return torch.log1p(1.0e3 * normalized) / scale
    raise ValueError(f"unsupported display scale: {mode}")


def _screen_extent(config: ScreenImageConfig) -> tuple[float, float, float, float]:
    """Return the detector extent tuple used by plotting helpers."""

    return (
        -0.5 * config.screen_width_mm,
        0.5 * config.screen_width_mm,
        -0.5 * config.screen_height_mm,
        0.5 * config.screen_height_mm,
    )


def _frame_title(angle_deg: float, default_prefix: str, title: str | None) -> str:
    """Return a display title for one detector frame."""

    if title is not None:
        return f"{title} | angle {angle_deg:.2f} deg"
    return f"{default_prefix} | angle {angle_deg:.2f} deg"


def _render_screen_frame_image(
    frame: np.ndarray,
    *,
    config: ScreenImageConfig,
    angle_deg: float,
    title: str | None,
    default_prefix: str,
    laue_circle_radius_mm: float | None = None,
) -> Image.Image:
    """Render one scaled CTR detector frame as an RGB image suitable for GIF export."""

    figure_width = 8.0
    figure_height = max(4.5, figure_width * config.screen_height_mm / config.screen_width_mm)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
    ax.imshow(
        frame,
        cmap=config.colormap,
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        extent=_screen_extent(config),
        aspect="equal",
    )
    ax.axhline(
        0.0,
        color="white",
        linewidth=1.25,
        linestyle=(0, (6, 4)),
        alpha=0.9,
        zorder=4,
    )
    if laue_circle_radius_mm is not None:
        laue_circle = plt.Circle(
            (0.0, 0.0),
            laue_circle_radius_mm,
            fill=False,
            edgecolor="lime",
            linewidth=1.25,
            linestyle=(0, (3, 3)),
            alpha=0.9,
            zorder=4,
        )
        ax.add_patch(laue_circle)
    direct_x_mm, direct_y_mm = _direct_beam_detector_coordinates(config, angle_deg)
    if math.isfinite(direct_x_mm) and math.isfinite(direct_y_mm):
        ax.scatter(
            [direct_x_mm],
            [direct_y_mm],
            marker="x",
            s=80,
            linewidths=2.0,
            color="cyan",
            zorder=5,
            label="Direct beam",
        )
        ax.annotate(
            "direct beam",
            xy=(direct_x_mm, direct_y_mm),
            xytext=(7, -9),
            textcoords="offset points",
            color="cyan",
            fontsize=8,
            ha="left",
            va="top",
            zorder=5,
        )
    ax.set_xlabel("Detector horizontal (mm)")
    ax.set_ylabel("Detector vertical (mm)")
    ax.set_title(_frame_title(angle_deg, default_prefix, title))
    buffer = BytesIO()
    fig.savefig(buffer, dpi=180, format="png")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB").copy()
    buffer.close()
    return image


def plot_screen_frame(
    result: RockingCurveResult,
    output_path: Path,
    *,
    angle_index: int | None = None,
    title: str | None = None,
) -> None:
    """Plot one detector frame from a rocking-curve result that includes screen images."""

    if result.screen_images is None or result.screen_config is None:
        raise ValueError("this rocking-curve result does not contain simulated screen images")
    stack = result.screen_images.detach().cpu()
    config = result.screen_config

    if angle_index is None:
        if config.reference_angle_deg is None:
            angle_index = 0
        else:
            angle_index = int(torch.argmin(torch.abs(result.angles_deg.detach().cpu() - config.reference_angle_deg)).item())
    if not 0 <= angle_index < stack.shape[0]:
        raise ValueError(f"angle_index {angle_index} is out of range for {stack.shape[0]} frames")

    frame = _scale_screen_image(stack[angle_index], config.display_scale).numpy()
    angle_deg = float(result.angles_deg.detach().cpu()[angle_index].item())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _render_screen_frame_image(
        frame,
        config=config,
        angle_deg=angle_deg,
        title=title,
        default_prefix="RHEED CTR screen",
    )
    try:
        image.save(output_path)
    finally:
        image.close()


def write_screen_gif(
    result: RockingCurveResult,
    output_path: Path,
    *,
    angle_min_deg: float | None = None,
    angle_max_deg: float | None = None,
    frame_step: int = 1,
    frame_duration_ms: int = 90,
    normalize: str = "global",
    boomerang: bool = True,
    title: str | None = None,
) -> None:
    """Write the full detector stack as an animated GIF."""

    if result.screen_images is None or result.screen_config is None:
        raise ValueError("this rocking-curve result does not contain simulated screen images")
    if frame_step < 1:
        raise ValueError("frame_step must be at least 1")
    if normalize not in {"global", "per-frame"}:
        raise ValueError(f"unsupported GIF normalization mode: {normalize}")

    stack = result.screen_images.detach().cpu()
    angles = result.angles_deg.detach().cpu()
    config = result.screen_config

    mask = torch.ones_like(angles, dtype=torch.bool)
    if angle_min_deg is not None:
        mask &= angles >= angle_min_deg - 1.0e-12
    if angle_max_deg is not None:
        mask &= angles <= angle_max_deg + 1.0e-12
    selected_indices = torch.nonzero(mask, as_tuple=False).flatten()
    if selected_indices.numel() == 0:
        raise ValueError("no detector frames fall within the requested GIF angle range")
    selected_indices = selected_indices[::frame_step]
    last_selected = torch.nonzero(mask, as_tuple=False).flatten()[-1]
    if int(selected_indices[-1].item()) != int(last_selected.item()):
        selected_indices = torch.cat([selected_indices, last_selected.view(1)], dim=0)

    maxima: list[float]
    if normalize == "global":
        global_max = float(torch.max(stack.index_select(0, selected_indices)).item())
        maxima = [global_max] * selected_indices.numel()
    else:
        maxima = [float(torch.max(stack[index]).item()) for index in selected_indices.tolist()]

    forward_frames: list[Image.Image] = []
    for local_index, angle_index in enumerate(selected_indices.tolist()):
        angle_deg = float(angles[angle_index].item())
        scaled = _scale_screen_image(stack[angle_index], config.display_scale, maxima[local_index]).numpy()
        forward_frames.append(
            _render_screen_frame_image(
                scaled,
                config=config,
                angle_deg=angle_deg,
                title=title,
                default_prefix="RHEED CTR screen",
            )
        )

    gif_frames = list(forward_frames)
    if boomerang and len(forward_frames) > 1:
        gif_frames.extend(frame.copy() for frame in forward_frames[-2::-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=frame_duration_ms,
            loop=0,
            disposal=2,
        )
    finally:
        for frame in gif_frames:
            frame.close()
