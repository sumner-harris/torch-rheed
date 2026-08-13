"""Public data models for the standalone torch_rheed package."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch


@dataclass
class BulkInput:
    """Parsed contents of a `bulk.txt` input file."""

    nh: int
    nk: int
    ndom: int
    nb: list[int]
    rdom_deg: list[float]
    ih: list[int]
    ik: list[int]
    be: float
    azi_deg: float
    azf_deg: float
    daz_deg: float
    gi_deg: float
    gf_deg: float
    dg_deg: float
    dz_input: float
    ml: int
    nelm: int
    iz: list[int]
    da1: list[float]
    sap: list[float]
    bh: list[float]
    bk: list[float]
    bz: list[float]
    nsg: int
    aa: float
    bb: float
    gam_deg: float
    cc: float
    dx: float
    dy: float
    natm: int
    ielm: list[int]
    ocr: list[float]
    x: list[float]
    y: list[float]
    z: list[float]
    source_path: Path | None = None


@dataclass
class SurfaceInput:
    """Parsed contents of a `surf.txt` input file."""

    nelms: int
    iz: list[int]
    da1: list[float]
    sap: list[float]
    bh: list[float]
    bk: list[float]
    bz: list[float]
    nsgs: int
    msa: int
    msb: int
    nsa: int
    nsb: int
    dthick: float
    dxs: float
    dys: float
    natms: int
    ielm: list[int]
    ocr: list[float]
    x: list[float]
    y: list[float]
    z: list[float]
    wdom: list[float]
    source_path: Path | None = None


@dataclass(frozen=True)
class StructureAtom:
    """One atom in a bulk+surface structure model used for plotting or export."""

    symbol: str
    atomic_number: int
    occupancy: float
    region: str
    x_angstrom: float
    y_angstrom: float
    z_angstrom: float


@dataclass
class StructureModel:
    """Expanded bulk+surface structure generated from ``bulk.txt`` and ``surf.txt``."""

    atoms: list[StructureAtom]
    a_vector_angstrom: tuple[float, float, float]
    b_vector_angstrom: tuple[float, float, float]
    bulk_layers: int
    a_units: int
    b_units: int
    bulk_source_path: Path | None = None
    surface_source_path: Path | None = None

    def _xy_transform(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the in-plane basis used by XYZ and CIF exports."""

        a_vec = (self.a_vector_angstrom[0], self.a_vector_angstrom[1])
        b_vec = (self.b_vector_angstrom[0], self.b_vector_angstrom[1])
        return a_vec, b_vec

    def select_regions(self, regions: str | list[str] | tuple[str, ...]) -> "StructureModel":
        """Return a copy containing only the requested ``bulk`` and/or ``surface`` atoms."""

        if isinstance(regions, str):
            selected_regions = {regions}
        else:
            selected_regions = set(regions)

        invalid_regions = selected_regions.difference({"bulk", "surface"})
        if invalid_regions:
            invalid = ", ".join(sorted(invalid_regions))
            raise ValueError(f"unsupported structure regions requested: {invalid}")

        return StructureModel(
            atoms=[atom for atom in self.atoms if atom.region in selected_regions],
            a_vector_angstrom=self.a_vector_angstrom,
            b_vector_angstrom=self.b_vector_angstrom,
            bulk_layers=self.bulk_layers,
            a_units=self.a_units,
            b_units=self.b_units,
            bulk_source_path=self.bulk_source_path,
            surface_source_path=self.surface_source_path,
        )

    def to_xyz_text(self) -> str:
        """Return the structure in simple XYZ format for external viewers."""

        comment = (
            f"bulk_layers={self.bulk_layers} a_units={self.a_units} b_units={self.b_units}"
        )
        lines = [str(len(self.atoms)), comment]
        for atom in self.atoms:
            lines.append(
                f"{atom.symbol} {atom.x_angstrom:.6f} {atom.y_angstrom:.6f} {atom.z_angstrom:.6f}"
            )
        return "\n".join(lines) + "\n"

    def write_xyz(self, path: Path) -> None:
        """Write the structure as an XYZ file for use in external viewers."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_xyz_text(), encoding="utf-8")

    def to_cif_text(self, *, vacuum_padding_angstrom: float = 6.0) -> str:
        """Return the structure as a simple P1 CIF slab for ASE or VESTA."""

        if vacuum_padding_angstrom <= 0.0:
            raise ValueError("vacuum_padding_angstrom must be positive")
        if not self.atoms:
            raise ValueError("structure contains no atoms to export")

        a_vec, b_vec = self._xy_transform()
        z_values = [atom.z_angstrom for atom in self.atoms]
        z_min = min(z_values)
        z_max = max(z_values)
        c_length = (z_max - z_min) + 2.0 * vacuum_padding_angstrom

        a_length = (a_vec[0] ** 2 + a_vec[1] ** 2) ** 0.5
        b_length = (b_vec[0] ** 2 + b_vec[1] ** 2) ** 0.5
        gamma_cos = (a_vec[0] * b_vec[0] + a_vec[1] * b_vec[1]) / (a_length * b_length)
        gamma_deg = math.degrees(math.acos(max(-1.0, min(1.0, gamma_cos))))

        det = a_vec[0] * b_vec[1] - a_vec[1] * b_vec[0]
        if abs(det) < 1e-12:
            raise ValueError("in-plane cell vectors are linearly dependent")

        lines = [
            "data_torch_rheed_structure",
            "_symmetry_space_group_name_H-M 'P 1'",
            "_symmetry_Int_Tables_number 1",
            f"_cell_length_a {a_length:.8f}",
            f"_cell_length_b {b_length:.8f}",
            f"_cell_length_c {c_length:.8f}",
            "_cell_angle_alpha 90.00000000",
            "_cell_angle_beta 90.00000000",
            f"_cell_angle_gamma {gamma_deg:.8f}",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        ]

        for index, atom in enumerate(self.atoms, start=1):
            x_frac = (atom.x_angstrom * b_vec[1] - atom.y_angstrom * b_vec[0]) / det
            y_frac = (-atom.x_angstrom * a_vec[1] + atom.y_angstrom * a_vec[0]) / det
            z_frac = (atom.z_angstrom - z_min + vacuum_padding_angstrom) / c_length
            lines.append(
                f"{atom.symbol}{index} {atom.symbol} "
                f"{x_frac:.8f} {y_frac:.8f} {z_frac:.8f} {atom.occupancy:.8f}"
            )
        return "\n".join(lines) + "\n"

    def write_cif(self, path: Path, *, vacuum_padding_angstrom: float = 6.0) -> None:
        """Write the structure as a CIF slab compatible with ASE and VESTA."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_cif_text(vacuum_padding_angstrom=vacuum_padding_angstrom), encoding="utf-8")


@dataclass
class BulkDomain:
    """In-memory bulk scattering data for one structural domain."""

    nb: int
    rdom_rad: float
    ih: list[int]
    ik: list[int]
    jorg: list[int]
    ngr: int
    nvb: int
    nv: int
    nbg: list[int]
    igh: list[int]
    igk: list[int]
    iv: list[list[int]]
    matrices: list[torch.Tensor]


@dataclass
class BulkSimulation:
    """Bulk-stage output consumed by the surface rocking-curve solver."""

    source: BulkInput
    device: torch.device
    inegpos: int
    idiag: int
    naz: int
    ng: int
    azi_rad: float
    daz_rad: float
    gi_rad: float
    dg_rad: float
    dz: float
    epsb: float
    gam_rad: float
    wn: float
    domains: list[BulkDomain]


@dataclass(frozen=True)
class ScreenImageConfig:
    """Detector geometry and CTR rasterization settings for synthetic RHEED screens."""

    plane_mode: str = "vertical"
    screen_distance_mm: float = 300.0
    screen_width_mm: float = 120.0
    screen_height_mm: float = 90.0
    pixels_x: int = 320
    pixels_y: int = 240
    correlation_length_angstrom: float = 1000.0
    rod_profile: str = "lorentzian"
    beam_intensity_floor: float = 0.0
    source_glancing_divergence_fwhm_deg: float = 0.0
    source_divergence_samples: int = 9
    reference_angle_deg: float | None = None
    frame_azimuth_deg: float | None = None
    display_scale: str = "sqrt"
    colormap: str = "inferno"


@dataclass
class RockingCurveResult:
    """Rocking-curve intensities for one simulated RHEED configuration.

    ``intensities`` is shaped ``(N, NB)``, where ``N`` is the number of
    glancing-angle samples and ``NB`` is the number of diffraction beams in
    ``beam_indices``. When present, ``screen_images`` is shaped ``(N, H, W)``.
    """

    beam_indices: list[tuple[int, int]]
    angles_deg: torch.Tensor
    intensities: torch.Tensor
    naz: int
    ng: int
    screen_images: torch.Tensor | None = None
    screen_config: ScreenImageConfig | None = None

    def _cpu_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return detached CPU tensors for text, CSV, and plotting helpers."""

        return self.angles_deg.detach().cpu(), self.intensities.detach().cpu()

    def screen_stack_cpu(self) -> torch.Tensor | None:
        """Return the optional screen-image stack on CPU."""

        if self.screen_images is None:
            return None
        return self.screen_images.detach().cpu()

    def to_surface_output_text(self) -> str:
        """Return the conventional `surf-bulkE.s` text representation."""

        pairs = "".join(f",{ih} {ik}" for ih, ik in self.beam_indices)
        header = "\n".join(
            [
                "#azimuths,g-angles,beams",
                f"{self.naz} {self.ng} {len(self.beam_indices)}",
                "#ih,ik",
                f"deg{pairs},",
            ]
        )
        angles_deg, intensities = self._cpu_views()
        lines: list[str] = []
        for row_idx in range(angles_deg.numel()):
            row = [angles_deg[row_idx].item(), *intensities[row_idx].tolist()]
            lines.append(",".join(f"{value:12.4E}" for value in row) + ",")
        return header + "\n" + "\n".join(lines) + "\n"

    def write_surface_output(self, path: Path) -> None:
        """Write the rocking curve in the traditional text format."""

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_surface_output_text(), encoding="utf-8")

    def write_csv(self, path: Path) -> None:
        """Write the rocking curve to CSV with one column per beam."""

        labels = [f"{ih} {ik}" for ih, ik in self.beam_indices]
        angles_deg, intensities = self._cpu_views()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("angle_deg," + ",".join(labels) + "\n")
            for row_idx in range(angles_deg.numel()):
                values = [f"{angles_deg[row_idx].item():.10g}"]
                values.extend(f"{value:.10g}" for value in intensities[row_idx].tolist())
                fh.write(",".join(values) + "\n")

    def write_screen_stack(self, path: Path) -> None:
        """Write the optional screen-image stack tensor with ``torch.save``."""

        if self.screen_images is None:
            raise ValueError("this rocking-curve result does not contain simulated screen images")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.screen_images.detach().cpu(), path)


@dataclass
class RockingCurveBatchResult:
    """Rocking-curve intensities for a batch of compatible RHEED structures.

    ``intensities`` is shaped ``(B, N, NB)``, where ``B`` is batch size,
    ``N`` is the number of glancing-angle samples, and ``NB`` is the number of
    diffraction beams in ``beam_indices``. When present, ``screen_images`` is
    shaped ``(B, N, H, W)``.
    """

    beam_indices: list[tuple[int, int]]
    angles_deg: torch.Tensor
    intensities: torch.Tensor
    naz: int
    ng: int
    surface_paths: list[Path | None]
    screen_images: torch.Tensor | None = None
    screen_config: ScreenImageConfig | None = None

    def __len__(self) -> int:
        """Return the number of structures in the batch."""

        return int(self.intensities.shape[0])

    def get_result(self, index: int) -> RockingCurveResult:
        """Extract one structure from the batch as a standard result object."""

        return RockingCurveResult(
            beam_indices=self.beam_indices,
            angles_deg=self.angles_deg.clone(),
            intensities=self.intensities[index].clone(),
            naz=self.naz,
            ng=self.ng,
            screen_images=None if self.screen_images is None else self.screen_images[index].clone(),
            screen_config=self.screen_config,
        )

    def split(self) -> list[RockingCurveResult]:
        """Split the batch result into per-structure rocking curves."""

        return [self.get_result(index) for index in range(len(self))]

    def screen_stack_cpu(self) -> torch.Tensor | None:
        """Return the optional batched screen-image stack on CPU."""

        if self.screen_images is None:
            return None
        return self.screen_images.detach().cpu()

    def write_screen_stack(self, path: Path) -> None:
        """Write the optional batched screen-image stack tensor with ``torch.save``."""

        if self.screen_images is None:
            raise ValueError("this batched rocking-curve result does not contain simulated screen images")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.screen_images.detach().cpu(), path)


@dataclass
class RockingCurvePairBatchResult:
    """Results for a high-level batch of ``(bulk.txt, surf.txt)`` simulation pairs.

    This container preserves input order even when the solver internally groups
    compatible pairs into smaller optimized tensor batches.
    """

    input_pairs: list[tuple[Path, Path]]
    results: list[RockingCurveResult]

    def __len__(self) -> int:
        """Return the number of simulated input pairs."""

        return len(self.results)

    def get_result(self, index: int) -> RockingCurveResult:
        """Return one result in the original pair order."""

        return self.results[index]

    def split(self) -> list[RockingCurveResult]:
        """Return the ordered per-pair rocking-curve results."""

        return list(self.results)

    def as_batch_result(self) -> RockingCurveBatchResult:
        """Restack fully compatible pair results into one tensor batch.

        This succeeds only when every pair produced the same angle grid, beam
        list, and screen-image configuration.
        """

        if not self.results:
            raise ValueError("cannot restack an empty pair batch")

        reference = self.results[0]
        reference_angles = reference.angles_deg.detach().cpu()
        screen_presence = reference.screen_images is not None

        for result in self.results[1:]:
            if result.beam_indices != reference.beam_indices:
                raise ValueError("pair batch contains incompatible beam lists and cannot be restacked")
            candidate_angles = result.angles_deg.detach().cpu()
            if candidate_angles.shape != reference_angles.shape or not torch.allclose(
                candidate_angles,
                reference_angles,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("pair batch contains incompatible angle grids and cannot be restacked")
            if result.naz != reference.naz or result.ng != reference.ng:
                raise ValueError("pair batch contains incompatible scan shapes and cannot be restacked")
            if (result.screen_images is not None) != screen_presence:
                raise ValueError("pair batch mixes results with and without screen-image stacks")
            if screen_presence:
                if result.screen_config != reference.screen_config:
                    raise ValueError("pair batch contains incompatible screen-image configurations")
                if result.screen_images.shape != reference.screen_images.shape:
                    raise ValueError("pair batch contains incompatible screen-image tensor shapes")

        intensities = torch.stack([result.intensities for result in self.results], dim=0)
        screen_images = None
        if screen_presence:
            screen_images = torch.stack([result.screen_images for result in self.results], dim=0)

        return RockingCurveBatchResult(
            beam_indices=reference.beam_indices,
            angles_deg=reference.angles_deg.clone(),
            intensities=intensities,
            naz=reference.naz,
            ng=reference.ng,
            surface_paths=[surface_path for _, surface_path in self.input_pairs],
            screen_images=screen_images,
            screen_config=reference.screen_config,
        )


@dataclass(frozen=True)
class PolycrystalOrientation:
    """One sampled grain orientation used in a polycrystalline approximation."""

    normal_hkl: tuple[int, int, int]
    azimuth_deg: float
    bulk_path: Path | None = None
    surface_path: Path | None = None


@dataclass
class PolycrystalResult:
    """Integrated detector result for a batch of discrete grain orientations.

    ``orientation_batch`` stores the raw per-orientation solver outputs, while
    ``integrated_screen_images`` stores the weighted average detector stack with
    shape ``(N, H, W)``. ``mean_beam_intensities`` is the same weighted average
    of the per-orientation beam intensities, shaped ``(N, NB)``.
    """

    orientations: list[PolycrystalOrientation]
    orientation_batch: RockingCurveBatchResult
    integrated_screen_images: torch.Tensor
    orientation_weights: torch.Tensor
    total_detector_intensity: torch.Tensor
    mean_beam_intensities: torch.Tensor
    screen_config: ScreenImageConfig
    failed_orientations: list[PolycrystalOrientation] | None = None

    @property
    def angles_deg(self) -> torch.Tensor:
        """Return the glancing-angle grid shared by every sampled orientation."""

        return self.orientation_batch.angles_deg

    def integrated_screen_stack_cpu(self) -> torch.Tensor:
        """Return the integrated detector stack on CPU."""

        return self.integrated_screen_images.detach().cpu()

    def orientation_screen_stack_cpu(self) -> torch.Tensor | None:
        """Return the raw per-orientation detector stack on CPU."""

        return self.orientation_batch.screen_stack_cpu()

    def write_integrated_screen_stack(self, path: Path) -> None:
        """Write the integrated detector stack with ``torch.save``."""

        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.integrated_screen_images.detach().cpu(), path)

    def write_orientation_screen_stack(self, path: Path) -> None:
        """Write the raw per-orientation detector stack with ``torch.save``."""

        self.orientation_batch.write_screen_stack(path)

    def write_total_detector_csv(self, path: Path) -> None:
        """Write the total integrated detector intensity versus angle to CSV."""

        angles = self.angles_deg.detach().cpu()
        totals = self.total_detector_intensity.detach().cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("angle_deg,total_detector_intensity\n")
            for angle_deg, total in zip(angles.tolist(), totals.tolist(), strict=True):
                fh.write(f"{angle_deg:.10g},{total:.10g}\n")

    def write_mean_beam_csv(self, path: Path) -> None:
        """Write the orientation-averaged beam intensities to CSV."""

        angles = self.angles_deg.detach().cpu()
        intensities = self.mean_beam_intensities.detach().cpu()
        labels = [f"{ih} {ik}" for ih, ik in self.orientation_batch.beam_indices]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("angle_deg," + ",".join(labels) + "\n")
            for row_idx in range(angles.numel()):
                values = [f"{angles[row_idx].item():.10g}"]
                values.extend(f"{value:.10g}" for value in intensities[row_idx].tolist())
                fh.write(",".join(values) + "\n")
