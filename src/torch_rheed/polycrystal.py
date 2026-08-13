"""Polycrystalline orientation sampling for standalone ``torch_rheed``."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .conversion import write_solver_inputs
from .inputs import load_bulk_input
from .models import (
    BulkInput,
    PolycrystalOrientation,
    PolycrystalResult,
    RockingCurvePairBatchResult,
    ScreenImageConfig,
    SurfaceInput,
)
from .screen import _scale_screen_image
from .simulation import simulate_pairs_batch


def _gcd_many(values: tuple[int, ...]) -> int:
    """Return the greatest common divisor of an integer tuple."""

    gcd_value = 0
    for value in values:
        gcd_value = math.gcd(gcd_value, abs(int(value)))
    return max(gcd_value, 1)


def _primitive_vector(vector: tuple[int, int, int]) -> tuple[int, int, int]:
    """Reduce one Miller-like integer vector to primitive form."""

    gcd_value = _gcd_many(vector)
    return tuple(int(component // gcd_value) for component in vector)


def _canonical_signed_vector(vector: tuple[int, int, int]) -> tuple[int, int, int]:
    """Choose a unique sign for an integer direction vector."""

    reduced = _primitive_vector(vector)
    for component in reduced:
        if component > 0:
            return reduced
        if component < 0:
            return tuple(-value for value in reduced)
    raise ValueError("zero vector is not a valid crystallographic direction")


def _enumerate_cubic_normals(max_miller_index: int) -> list[tuple[int, int, int]]:
    """Enumerate unique primitive cubic normals on one hemisphere."""

    if max_miller_index < 1:
        raise ValueError("max_miller_index must be at least 1")

    normals: set[tuple[int, int, int]] = set()
    for h_value in range(-max_miller_index, max_miller_index + 1):
        for k_value in range(-max_miller_index, max_miller_index + 1):
            for l_value in range(-max_miller_index, max_miller_index + 1):
                if h_value == 0 and k_value == 0 and l_value == 0:
                    continue
                normals.add(_canonical_signed_vector((h_value, k_value, l_value)))

    return sorted(normals, key=lambda normal: (normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2, normal))


def generate_beam_shell_indices(radius: int) -> list[tuple[int, int]]:
    """Generate a 2D integer-order beam shell ``h^2 + k^2 <= radius^2``."""

    if radius < 0:
        raise ValueError("beam-shell radius must be non-negative")

    beams = [
        (h_value, k_value)
        for h_value in range(-radius, radius + 1)
        for k_value in range(-radius, radius + 1)
        if h_value * h_value + k_value * k_value <= radius * radius
    ]
    return sorted(beams, key=lambda beam: (beam[0] * beam[0] + beam[1] * beam[1], beam[0], beam[1]))


def _azimuth_samples(start_deg: float, stop_deg: float, step_deg: float) -> list[float]:
    """Generate an inclusive azimuth grid in degrees."""

    if step_deg <= 0.0:
        raise ValueError("azimuth_step_deg must be positive")
    if stop_deg < start_deg:
        raise ValueError("azimuth_stop_deg must be greater than or equal to azimuth_start_deg")

    count = int(round((stop_deg - start_deg) / step_deg)) + 1
    samples = [start_deg + index * step_deg for index in range(count)]
    if not math.isclose(samples[-1], stop_deg, rel_tol=0.0, abs_tol=1.0e-9):
        samples.append(stop_deg)
    return samples


def _validate_cubic_bulk(base_bulk: BulkInput) -> None:
    """Reject base inputs that are not simple cubic single-domain bulks."""

    if base_bulk.ndom != 1:
        raise NotImplementedError("polycrystal mode currently supports only single-domain bulk inputs")
    if base_bulk.nsg not in (0, 1):
        raise NotImplementedError("polycrystal mode currently supports only p1 bulk symmetry")
    if abs(base_bulk.aa - base_bulk.bb) > 1.0e-8 or abs(base_bulk.aa - base_bulk.cc) > 1.0e-8:
        raise NotImplementedError("polycrystal mode currently requires aa = bb = cc for a cubic bulk cell")
    if abs(base_bulk.gam_deg - 90.0) > 1.0e-8:
        raise NotImplementedError("polycrystal mode currently requires a right-angle in-plane cell")


def _choose_in_plane_basis(normal_hkl: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Choose two primitive lattice vectors spanning the plane normal to ``normal_hkl``."""

    normal = np.array(normal_hkl, dtype=np.int64)
    axis_candidates = [
        np.array([1, 0, 0], dtype=np.int64),
        np.array([0, 1, 0], dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
    ]

    u_candidates: list[tuple[int, int, int]] = []
    for axis in axis_candidates:
        cross = np.cross(normal, axis)
        if np.any(cross):
            u_candidates.append(_primitive_vector((int(cross[0]), int(cross[1]), int(cross[2]))))

    if not u_candidates:
        raise RuntimeError(f"failed to construct an in-plane basis for normal {normal_hkl}")

    u_vector = min(u_candidates, key=lambda vec: (vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2, vec))
    v_cross = np.cross(normal, np.array(u_vector, dtype=np.int64))
    v_vector = _primitive_vector((int(v_cross[0]), int(v_cross[1]), int(v_cross[2])))

    basis_matrix = np.column_stack((np.array(u_vector, dtype=np.int64), np.array(v_vector, dtype=np.int64), normal))
    if round(np.linalg.det(basis_matrix)) < 0:
        v_vector = tuple(-value for value in v_vector)
    return u_vector, v_vector


def _oriented_bulk_atoms(
    base_bulk: BulkInput,
    *,
    normal_hkl: tuple[int, int, int],
) -> tuple[float, float, float, float, list[tuple[int, float, float, float, float]]]:
    """Build one oriented cubic bulk cell from the conventional cubic basis."""

    lattice_constant = base_bulk.aa
    u_vector, v_vector = _choose_in_plane_basis(normal_hkl)
    normal = np.array(normal_hkl, dtype=np.int64)
    matrix = np.column_stack(
        (
            np.array(u_vector, dtype=np.float64),
            np.array(v_vector, dtype=np.float64),
            np.array(normal, dtype=np.float64),
        )
    )
    inv_matrix = np.linalg.inv(matrix)
    determinant = int(round(abs(np.linalg.det(matrix))))
    if determinant < 1:
        raise RuntimeError(f"invalid oriented-cell determinant for normal {normal_hkl}")

    search_radius = int(np.max(np.sum(np.abs(matrix), axis=1))) + 2
    basis_fractional = np.column_stack(
        (
            np.array(base_bulk.x, dtype=np.float64),
            np.array(base_bulk.y, dtype=np.float64),
            np.array(base_bulk.z, dtype=np.float64) / base_bulk.cc,
        )
    )
    cc_length = lattice_constant * float(np.linalg.norm(normal))
    aa_length = lattice_constant * float(np.linalg.norm(np.array(u_vector, dtype=np.float64)))
    bb_length = lattice_constant * float(np.linalg.norm(np.array(v_vector, dtype=np.float64)))
    gamma_cos = float(
        np.dot(np.array(u_vector, dtype=np.float64), np.array(v_vector, dtype=np.float64))
        / (np.linalg.norm(np.array(u_vector, dtype=np.float64)) * np.linalg.norm(np.array(v_vector, dtype=np.float64)))
    )
    gamma_deg = math.degrees(math.acos(max(-1.0, min(1.0, gamma_cos))))

    tolerance = 1.0e-8
    bulk_records: dict[tuple[int, int, int, int, int], tuple[int, float, float, float, float]] = {}
    for atom_index in range(base_bulk.natm):
        fractional = basis_fractional[atom_index]
        found_any = False
        for tx in range(-search_radius, search_radius + 1):
            for ty in range(-search_radius, search_radius + 1):
                for tz in range(-search_radius, search_radius + 1):
                    translated = fractional + np.array([tx, ty, tz], dtype=np.float64)
                    oriented = inv_matrix @ translated
                    if np.any(oriented < -tolerance) or np.any(oriented >= 1.0 - tolerance):
                        continue
                    found_any = True
                    x_value = oriented[0] - math.floor(oriented[0] + 1.0e-4)
                    y_value = oriented[1] - math.floor(oriented[1] + 1.0e-4)
                    z_value = oriented[2] * cc_length
                    if abs(z_value) < 1.0e-8 or abs(z_value - cc_length) < 1.0e-8:
                        z_value = 0.0
                    key = (
                        base_bulk.ielm[atom_index],
                        round(base_bulk.ocr[atom_index] * 1.0e6),
                        round(x_value * 1.0e6),
                        round(y_value * 1.0e6),
                        round(z_value * 1.0e6),
                    )
                    bulk_records[key] = (
                        base_bulk.ielm[atom_index],
                        base_bulk.ocr[atom_index],
                        x_value,
                        y_value,
                        z_value,
                    )
        if not found_any:
            raise RuntimeError(
                f"failed to place atom {atom_index} from the cubic basis into the oriented cell for normal {normal_hkl}"
            )

    expected_atoms = determinant * base_bulk.natm
    if len(bulk_records) != expected_atoms:
        raise RuntimeError(
            f"oriented cell for normal {normal_hkl} should contain {expected_atoms} atoms but produced {len(bulk_records)}"
        )

    bulk_rows = sorted(bulk_records.values(), key=lambda row: (row[4], row[0], row[2], row[3]))
    return aa_length, bb_length, gamma_deg, cc_length, bulk_rows


def build_oriented_cubic_bulk_input(
    base_bulk: BulkInput,
    *,
    normal_hkl: tuple[int, int, int],
    azimuth_deg: float,
    beam_indices: list[tuple[int, int]],
) -> BulkInput:
    """Create one cubic oriented bulk cell with a new surface normal and azimuth."""

    _validate_cubic_bulk(base_bulk)
    aa_length, bb_length, gamma_deg, cc_length, bulk_rows = _oriented_bulk_atoms(base_bulk, normal_hkl=normal_hkl)
    beam_ih = [beam[0] for beam in beam_indices]
    beam_ik = [beam[1] for beam in beam_indices]
    return BulkInput(
        nh=base_bulk.nh,
        nk=base_bulk.nk,
        ndom=1,
        nb=[len(beam_indices)],
        rdom_deg=[0.0],
        ih=beam_ih,
        ik=beam_ik,
        be=base_bulk.be,
        azi_deg=azimuth_deg,
        azf_deg=azimuth_deg,
        daz_deg=0.0,
        gi_deg=base_bulk.gi_deg,
        gf_deg=base_bulk.gf_deg,
        dg_deg=base_bulk.dg_deg,
        dz_input=base_bulk.dz_input,
        ml=base_bulk.ml,
        nelm=base_bulk.nelm,
        iz=list(base_bulk.iz),
        da1=list(base_bulk.da1),
        sap=list(base_bulk.sap),
        bh=list(base_bulk.bh),
        bk=list(base_bulk.bk),
        bz=list(base_bulk.bz),
        nsg=base_bulk.nsg,
        aa=aa_length,
        bb=bb_length,
        gam_deg=gamma_deg,
        cc=cc_length,
        dx=0.0,
        dy=0.0,
        natm=len(bulk_rows),
        ielm=[row[0] for row in bulk_rows],
        ocr=[row[1] for row in bulk_rows],
        x=[row[2] for row in bulk_rows],
        y=[row[3] for row in bulk_rows],
        z=[row[4] for row in bulk_rows],
        source_path=None,
    )


def build_blank_surface_input(base_bulk: BulkInput, *, thickness_fraction: float = 0.5) -> SurfaceInput:
    """Create a nearly-empty surface file suitable for bulk-dominated polycrystal runs."""

    if thickness_fraction < 0.0:
        raise ValueError("thickness_fraction must be non-negative")
    return SurfaceInput(
        nelms=base_bulk.nelm,
        iz=list(base_bulk.iz),
        da1=list(base_bulk.da1),
        sap=list(base_bulk.sap),
        bh=list(base_bulk.bh),
        bk=list(base_bulk.bk),
        bz=list(base_bulk.bz),
        nsgs=1,
        msa=1,
        msb=0,
        nsa=0,
        nsb=1,
        dthick=thickness_fraction * base_bulk.cc,
        dxs=0.0,
        dys=0.0,
        natms=0,
        ielm=[],
        ocr=[],
        x=[],
        y=[],
        z=[],
        wdom=[1.0],
        source_path=None,
    )


def generate_polycrystal_input_pairs(
    base_bulk_path: Path,
    *,
    out_dir: Path,
    max_miller_index: int,
    azimuth_start_deg: float,
    azimuth_stop_deg: float,
    azimuth_step_deg: float,
    beam_indices: list[tuple[int, int]],
) -> tuple[list[PolycrystalOrientation], list[tuple[Path, Path]]]:
    """Generate oriented ``bulk.txt``/``surf.txt`` pairs for a cubic polycrystal."""

    base_bulk = load_bulk_input(base_bulk_path)
    normals = _enumerate_cubic_normals(max_miller_index)
    azimuth_values = _azimuth_samples(azimuth_start_deg, azimuth_stop_deg, azimuth_step_deg)
    inputs_dir = out_dir / "orientation_inputs"

    orientations: list[PolycrystalOrientation] = []
    input_pairs: list[tuple[Path, Path]] = []
    for normal_hkl in normals:
        for azimuth_deg in azimuth_values:
            oriented_bulk = build_oriented_cubic_bulk_input(
                base_bulk,
                normal_hkl=normal_hkl,
                azimuth_deg=azimuth_deg,
                beam_indices=beam_indices,
            )
            blank_surface = build_blank_surface_input(oriented_bulk)
            stem = f"h{normal_hkl[0]}_k{normal_hkl[1]}_l{normal_hkl[2]}_az{azimuth_deg:+07.2f}".replace(".", "p")
            pair_dir = inputs_dir / stem
            bulk_path = pair_dir / "bulk.txt"
            surface_path = pair_dir / "surf.txt"
            write_solver_inputs(oriented_bulk, blank_surface, bulk_path=bulk_path, surface_path=surface_path)
            orientations.append(
                PolycrystalOrientation(
                    normal_hkl=normal_hkl,
                    azimuth_deg=azimuth_deg,
                    bulk_path=bulk_path,
                    surface_path=surface_path,
                )
            )
            input_pairs.append((bulk_path, surface_path))

    return orientations, input_pairs


def simulate_polycrystal_from_bulk(
    base_bulk_path: Path,
    *,
    out_dir: Path,
    max_miller_index: int,
    azimuth_start_deg: float = 0.0,
    azimuth_stop_deg: float = 180.0,
    azimuth_step_deg: float = 15.0,
    beam_indices: list[tuple[int, int]] | None = None,
    beam_shell_radius: int = 4,
    device: str | torch.device | None = None,
    screen_config: ScreenImageConfig = ScreenImageConfig(frame_azimuth_deg=0.0),
) -> PolycrystalResult:
    """Generate many cubic grain orientations, simulate them, and integrate the detector stack."""

    resolved_beam_indices = beam_indices or generate_beam_shell_indices(beam_shell_radius)
    orientations, input_pairs = generate_polycrystal_input_pairs(
        base_bulk_path,
        out_dir=out_dir,
        max_miller_index=max_miller_index,
        azimuth_start_deg=azimuth_start_deg,
        azimuth_stop_deg=azimuth_stop_deg,
        azimuth_step_deg=azimuth_step_deg,
        beam_indices=resolved_beam_indices,
    )
    successful_pairs: list[tuple[Path, Path]] = []
    successful_results = []
    successful_orientations: list[PolycrystalOrientation] = []
    failed_orientations: list[PolycrystalOrientation] = []
    for orientation, input_pair in zip(orientations, input_pairs, strict=True):
        try:
            pair_result = simulate_pairs_batch([input_pair], device=device, screen_config=screen_config)
        except RuntimeError:
            failed_orientations.append(orientation)
            continue
        successful_pairs.append(input_pair)
        successful_results.append(pair_result.get_result(0))
        successful_orientations.append(orientation)

    if not successful_results:
        raise RuntimeError("every sampled grain orientation failed during the polycrystal solve")

    pair_batch = RockingCurvePairBatchResult(
        input_pairs=successful_pairs,
        results=successful_results,
    )
    batch_result = pair_batch.as_batch_result()
    if batch_result.screen_images is None or batch_result.screen_config is None:
        raise RuntimeError("polycrystal simulation requires detector image stacks to be enabled")

    batch_count = len(successful_orientations)
    weight_dtype = batch_result.intensities.dtype
    intensity_weights = torch.full((batch_count,), 1.0 / batch_count, dtype=weight_dtype, device=batch_result.intensities.device)
    screen_weights = intensity_weights.to(dtype=batch_result.screen_images.dtype, device=batch_result.screen_images.device)

    mean_beam_intensities = torch.sum(batch_result.intensities * intensity_weights[:, None, None], dim=0)
    integrated_screen_images = torch.sum(batch_result.screen_images * screen_weights[:, None, None, None], dim=0)
    total_detector_intensity = torch.sum(integrated_screen_images, dim=(-2, -1)).to(dtype=batch_result.intensities.dtype)

    return PolycrystalResult(
        orientations=successful_orientations,
        orientation_batch=batch_result,
        integrated_screen_images=integrated_screen_images,
        orientation_weights=intensity_weights,
        total_detector_intensity=total_detector_intensity,
        mean_beam_intensities=mean_beam_intensities,
        screen_config=batch_result.screen_config,
        failed_orientations=failed_orientations,
    )


def plot_polycrystal_screen_frame(
    result: PolycrystalResult,
    output_path: Path,
    *,
    angle_index: int | None = None,
    title: str | None = None,
) -> None:
    """Plot one integrated detector frame from a ``PolycrystalResult``."""

    config = result.screen_config
    stack = result.integrated_screen_images.detach().cpu()
    if angle_index is None:
        if config.reference_angle_deg is None:
            angle_index = 0
        else:
            angle_index = int(torch.argmin(torch.abs(result.angles_deg.detach().cpu() - config.reference_angle_deg)).item())
    if not 0 <= angle_index < stack.shape[0]:
        raise ValueError(f"angle_index {angle_index} is out of range for {stack.shape[0]} frames")

    frame = _scale_screen_image(stack[angle_index], config.display_scale).numpy()
    extent = (
        -0.5 * config.screen_width_mm,
        0.5 * config.screen_width_mm,
        -0.5 * config.screen_height_mm,
        0.5 * config.screen_height_mm,
    )

    figure_width = 8.0
    figure_height = max(4.5, figure_width * config.screen_height_mm / config.screen_width_mm)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
    ax.imshow(frame, cmap=config.colormap, origin="upper", extent=extent, aspect="equal")
    ax.set_xlabel("Detector horizontal (mm)")
    ax.set_ylabel("Detector vertical (mm)")
    angle_deg = float(result.angles_deg.detach().cpu()[angle_index].item())
    ax.set_title(title or f"Polycrystalline screen | angle {angle_deg:.2f} deg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_polycrystal_total_intensity(
    result: PolycrystalResult,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    """Plot total integrated detector intensity versus glancing angle."""

    angles_deg = result.angles_deg.detach().cpu()
    totals = result.total_detector_intensity.detach().cpu()
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.plot(angles_deg.tolist(), totals.tolist(), color="#f97316", linewidth=2.2)
    ax.set_xlabel("Glancing angle (deg)")
    ax.set_ylabel("Integrated detector intensity")
    ax.set_title(title or "Polycrystalline detector intensity")
    ax.grid(True, alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
