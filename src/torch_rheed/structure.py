"""Structure-building and 3D-plot utilities for standalone ``torch_rheed``."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .inputs import load_bulk_input, load_surface_input
from .models import BulkInput, StructureAtom, StructureModel, SurfaceInput


_ELEMENT_SYMBOLS = [
    "",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
]

_ELEMENT_COLORS = {
    "H": "#f7f7f7",
    "C": "#4d4d4d",
    "N": "#3b82f6",
    "O": "#ef4444",
    "Al": "#9ca3af",
    "Si": "#d9a441",
    "Ti": "#6b7280",
    "Ge": "#06b6d4",
    "Sr": "#22c55e",
}

_ELEMENT_RADII = {
    "H": 0.32,
    "C": 0.70,
    "N": 0.65,
    "O": 0.60,
    "Al": 1.18,
    "Si": 1.11,
    "Ti": 1.40,
    "Ge": 1.20,
    "Sr": 1.95,
}


def _require_supported_geometry(nsg: int, label: str) -> None:
    """Reject plane-group settings outside the current standalone scope."""

    if nsg not in (0, 1):
        raise NotImplementedError(
            f"torch_rheed structure utilities currently support only p1 symmetry for the {label} layer; got nsg={nsg}"
        )


def _reduce01(value: float) -> float:
    """Reduce a coordinate into the half-open interval ``[0, 1)``."""

    return value - math.floor(value + 1e-4)


def _element_symbol(atomic_number: int) -> str:
    """Return the periodic-table symbol used for XYZ export and legends."""

    if not 1 <= atomic_number < len(_ELEMENT_SYMBOLS):
        raise ValueError(f"atomic number must be between 1 and 98, got {atomic_number}")
    return _ELEMENT_SYMBOLS[atomic_number]


def _surface_transform(surface: SurfaceInput) -> np.ndarray:
    """Return the integer transform from surface-cell to bulk-cell coordinates."""

    return np.array(
        [[float(surface.msa), float(surface.nsa)], [float(surface.msb), float(surface.nsb)]],
        dtype=np.float64,
    )


def _principal_surface_coordinate(x: float, y: float, transform: np.ndarray) -> np.ndarray:
    """Map one in-plane coordinate into the principal surface unit cell."""

    surface_coord = np.linalg.solve(transform, np.array([x, y], dtype=np.float64))
    return np.array([_reduce01(float(surface_coord[0])), _reduce01(float(surface_coord[1]))], dtype=np.float64)


def _bulk_basis_vectors(bulk: BulkInput) -> tuple[np.ndarray, np.ndarray]:
    """Return the bulk ``a`` and ``b`` lattice vectors in Cartesian coordinates."""

    gamma_rad = math.radians(bulk.gam_deg)
    a_vec = np.array([bulk.aa, 0.0, 0.0], dtype=np.float64)
    b_vec = np.array([bulk.bb * math.cos(gamma_rad), bulk.bb * math.sin(gamma_rad), 0.0], dtype=np.float64)
    return a_vec, b_vec


def _bulk_to_cartesian(bulk_coord: np.ndarray, a_vec: np.ndarray, b_vec: np.ndarray, z_angstrom: float) -> tuple[float, float, float]:
    """Convert one bulk-basis in-plane coordinate into Cartesian Angstroms."""

    xy = bulk_coord[0] * a_vec[:2] + bulk_coord[1] * b_vec[:2]
    return float(xy[0]), float(xy[1]), float(z_angstrom)


def build_structure_model(
    bulk: BulkInput,
    surface: SurfaceInput,
    *,
    a_units: int = 1,
    b_units: int = 1,
    bulk_layers: int = 3,
) -> StructureModel:
    """Build the expanded bulk+surface structure used by the standalone solver."""

    if a_units < 1 or b_units < 1:
        raise ValueError("a_units and b_units must both be at least 1")
    if bulk_layers < 1:
        raise ValueError("bulk_layers must be at least 1")

    _require_supported_geometry(bulk.nsg, "bulk")
    _require_supported_geometry(surface.nsgs, "surface")

    a_vec, b_vec = _bulk_basis_vectors(bulk)
    transform = _surface_transform(surface)
    transform_inv = np.linalg.inv(transform)

    a_surface = surface.msa * a_vec + surface.msb * b_vec
    b_surface = surface.nsa * a_vec + surface.nsb * b_vec
    file_a = tuple(float(value) for value in (a_units * a_surface))
    file_b = tuple(float(value) for value in (b_units * b_surface))

    atoms: list[StructureAtom] = []

    for atom_index in range(surface.natms):
        atomic_number = surface.iz[surface.ielm[atom_index] - 1]
        symbol = _element_symbol(atomic_number)
        base_surface_coord = _principal_surface_coordinate(surface.x[atom_index], surface.y[atom_index], transform)
        for ia in range(a_units):
            for ib in range(b_units):
                bulk_coord = transform @ (base_surface_coord + np.array([ia, ib], dtype=np.float64))
                x_ang, y_ang, z_ang = _bulk_to_cartesian(bulk_coord, a_vec, b_vec, surface.z[atom_index])
                atoms.append(
                    StructureAtom(
                        symbol=symbol,
                        atomic_number=atomic_number,
                        occupancy=surface.ocr[atom_index],
                        region="surface",
                        x_angstrom=x_ang,
                        y_angstrom=y_ang,
                        z_angstrom=z_ang,
                    )
                )

    surface_corners = np.array(
        [
            [0.0, 0.0],
            [float(a_units), 0.0],
            [0.0, float(b_units)],
            [float(a_units), float(b_units)],
        ],
        dtype=np.float64,
    )
    bulk_corners = (transform @ surface_corners.T).T

    for layer_index in range(1, bulk_layers + 1):
        layer_shift = np.array([surface.dxs + bulk.dx * (layer_index - 1), surface.dys + bulk.dy * (layer_index - 1)], dtype=np.float64)
        for atom_index in range(bulk.natm):
            atomic_number = bulk.iz[bulk.ielm[atom_index] - 1]
            symbol = _element_symbol(atomic_number)
            base_bulk_coord = np.array([bulk.x[atom_index], bulk.y[atom_index]], dtype=np.float64) - layer_shift
            translate_min = np.floor(np.min(bulk_corners - base_bulk_coord, axis=0)).astype(int) - 1
            translate_max = np.ceil(np.max(bulk_corners - base_bulk_coord, axis=0)).astype(int) + 1

            for tx in range(int(translate_min[0]), int(translate_max[0]) + 1):
                for ty in range(int(translate_min[1]), int(translate_max[1]) + 1):
                    bulk_coord = base_bulk_coord + np.array([tx, ty], dtype=np.float64)
                    surface_coord = transform_inv @ bulk_coord
                    eps = 1.0e-9
                    if (
                        -eps <= surface_coord[0] < float(a_units) - eps
                        and -eps <= surface_coord[1] < float(b_units) - eps
                    ):
                        x_ang, y_ang, z_ang = _bulk_to_cartesian(bulk_coord, a_vec, b_vec, bulk.z[atom_index] - bulk.cc * layer_index)
                        atoms.append(
                            StructureAtom(
                                symbol=symbol,
                                atomic_number=atomic_number,
                                occupancy=bulk.ocr[atom_index],
                                region="bulk",
                                x_angstrom=x_ang,
                                y_angstrom=y_ang,
                                z_angstrom=z_ang,
                            )
                        )

    atoms.sort(key=lambda atom: (atom.z_angstrom, atom.symbol, atom.x_angstrom, atom.y_angstrom))
    return StructureModel(
        atoms=atoms,
        a_vector_angstrom=file_a,
        b_vector_angstrom=file_b,
        bulk_layers=bulk_layers,
        a_units=a_units,
        b_units=b_units,
        bulk_source_path=bulk.source_path,
        surface_source_path=surface.source_path,
    )


def build_structure_from_files(
    bulk_path: Path,
    surface_path: Path,
    *,
    a_units: int = 1,
    b_units: int = 1,
    bulk_layers: int = 3,
) -> StructureModel:
    """Load standalone inputs and build the corresponding expanded structure."""

    bulk = load_bulk_input(bulk_path)
    surface = load_surface_input(surface_path, bulk.ndom)
    return build_structure_model(
        bulk,
        surface,
        a_units=a_units,
        b_units=b_units,
        bulk_layers=bulk_layers,
    )


def _element_color(symbol: str, atomic_number: int) -> str:
    """Return a deterministic display color for one element."""

    if symbol in _ELEMENT_COLORS:
        return _ELEMENT_COLORS[symbol]
    return plt.cm.tab20((atomic_number % 20) / 19.0)


def _element_size(symbol: str) -> float:
    """Return a display size scaled from a lightweight covalent-radius table."""

    radius = _ELEMENT_RADII.get(symbol, 1.0)
    return 90.0 * radius * radius


def _set_equal_box_aspect(ax, x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray) -> None:
    """Keep the three axes visually balanced for structure plots."""

    x_span = max(float(x_values.max() - x_values.min()), 1.0)
    y_span = max(float(y_values.max() - y_values.min()), 1.0)
    z_span = max(float(z_values.max() - z_values.min()), 1.0)
    ax.set_box_aspect((x_span, y_span, z_span))


def _draw_cell_outline(ax, structure: StructureModel, z_plane: float, *, color: str = "#64748b", alpha: float = 0.75) -> None:
    """Draw one periodic-cell outline at the requested ``z`` position."""

    a_vec = np.array(structure.a_vector_angstrom, dtype=np.float64)
    b_vec = np.array(structure.b_vector_angstrom, dtype=np.float64)
    corners = np.array(
        [
            [0.0, 0.0, z_plane],
            [a_vec[0], a_vec[1], z_plane],
            [a_vec[0] + b_vec[0], a_vec[1] + b_vec[1], z_plane],
            [b_vec[0], b_vec[1], z_plane],
            [0.0, 0.0, z_plane],
        ],
        dtype=np.float64,
    )
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color, linewidth=1.3, alpha=alpha)


def _normalize_regions(regions: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize the optional region selector used by structure plotting helpers."""

    if regions is None:
        return ("bulk", "surface")
    if isinstance(regions, str):
        return (regions,)
    return tuple(regions)


def plot_structure_views(
    structure: StructureModel,
    output_path: Path,
    *,
    title: str | None = None,
    regions: str | list[str] | tuple[str, ...] | None = None,
) -> None:
    """Plot perspective, top, and side views of one expanded bulk+surface structure."""

    selected_regions = _normalize_regions(regions)
    display_structure = structure.select_regions(selected_regions)
    if not display_structure.atoms:
        joined = ", ".join(selected_regions)
        raise ValueError(f"structure contains no atoms for the requested regions: {joined}")

    x_values = np.array([atom.x_angstrom for atom in display_structure.atoms], dtype=np.float64)
    y_values = np.array([atom.y_angstrom for atom in display_structure.atoms], dtype=np.float64)
    z_values = np.array([atom.z_angstrom for atom in display_structure.atoms], dtype=np.float64)

    x_pad = max((x_values.max() - x_values.min()) * 0.08, 0.8)
    y_pad = max((y_values.max() - y_values.min()) * 0.08, 0.8)
    z_pad = max((z_values.max() - z_values.min()) * 0.08, 0.8)

    fig = plt.figure(figsize=(14, 5.4), constrained_layout=True)
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    view_settings = [
        ("Perspective", 22, -58),
        ("Top", 90, -90),
        ("Side", 8, -90),
    ]

    legend_handles: dict[str, object] = {}
    z_bottom = float(z_values.min())
    z_top = float(z_values.max())

    for ax, (label, elev, azim) in zip(axes, view_settings):
        for atom in [atom for atom in display_structure.atoms if atom.region == "bulk"]:
            handle = ax.scatter(
                atom.x_angstrom,
                atom.y_angstrom,
                atom.z_angstrom,
                s=_element_size(atom.symbol),
                c=[_element_color(atom.symbol, atom.atomic_number)],
                alpha=max(0.20, min(atom.occupancy, 0.85)),
                edgecolors="none",
            )
            legend_handles.setdefault(atom.symbol, handle)

        for atom in [atom for atom in display_structure.atoms if atom.region == "surface"]:
            handle = ax.scatter(
                atom.x_angstrom,
                atom.y_angstrom,
                atom.z_angstrom,
                s=_element_size(atom.symbol) * 1.08,
                c=[_element_color(atom.symbol, atom.atomic_number)],
                alpha=max(0.35, min(atom.occupancy, 1.0)),
                edgecolors="black",
                linewidths=0.4,
            )
            legend_handles.setdefault(atom.symbol, handle)

        _draw_cell_outline(ax, display_structure, z_top)
        _draw_cell_outline(ax, display_structure, z_bottom)
        a_vec = np.array(display_structure.a_vector_angstrom, dtype=np.float64)
        b_vec = np.array(display_structure.b_vector_angstrom, dtype=np.float64)
        for corner in ([0.0, 0.0], [a_vec[0], a_vec[1]], [b_vec[0], b_vec[1]], [a_vec[0] + b_vec[0], a_vec[1] + b_vec[1]]):
            ax.plot(
                [corner[0], corner[0]],
                [corner[1], corner[1]],
                [z_bottom, z_top],
                color="#94a3b8",
                linewidth=0.8,
                alpha=0.55,
            )

        ax.set_xlim(float(x_values.min() - x_pad), float(x_values.max() + x_pad))
        ax.set_ylim(float(y_values.min() - y_pad), float(y_values.max() + y_pad))
        ax.set_zlim(float(z_values.min() - z_pad), float(z_values.max() + z_pad))
        _set_equal_box_aspect(ax, x_values, y_values, z_values)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(label)
        ax.set_xlabel("x (A)")
        ax.set_ylabel("y (A)")
        ax.set_zlabel("z (A)")

    if title is None:
        if selected_regions == ("bulk", "surface"):
            plot_title = "Bulk+surface structure"
        elif selected_regions == ("bulk",):
            plot_title = "Bulk structure"
        elif selected_regions == ("surface",):
            plot_title = "Surface structure"
        else:
            plot_title = "Selected structure regions"
    else:
        plot_title = title

    fig.suptitle(plot_title, fontsize=14)
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="upper right",
        title="Elements",
        frameon=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
