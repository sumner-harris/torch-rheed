"""Importers and writers for converting slab structures into solver inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import shlex

import numpy as np

from .models import BulkInput, SurfaceInput


_LATTICE_RE = re.compile(r"""Lattice\s*=\s*["']([^"']+)["']""")
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
_ATOMIC_NUMBERS = {symbol: atomic_number for atomic_number, symbol in enumerate(_ELEMENT_SYMBOLS) if symbol}
_DEFAULT_ELEMENT_PARAMS = {
    "Sr": (1.0, 0.1, 0.5, 0.5, 0.5),
    "Ti": (1.0, 0.1, 0.5, 0.5, 0.5),
    "O": (1.0, 0.1, 0.6, 0.6, 0.6),
}


@dataclass(frozen=True)
class ImportedAtom:
    """One atom loaded from a CIF or XYZ slab structure."""

    symbol: str
    atomic_number: int
    occupancy: float
    x_angstrom: float
    y_angstrom: float
    z_angstrom: float


@dataclass
class ImportedStructure:
    """Cartesian slab structure plus periodic cell vectors."""

    atoms: list[ImportedAtom]
    a_vector_angstrom: tuple[float, float, float]
    b_vector_angstrom: tuple[float, float, float]
    c_vector_angstrom: tuple[float, float, float]
    source_path: Path | None = None


@dataclass(frozen=True)
class ElementParameters:
    """Numerical atomic-parameter block used by bulk.txt and surf.txt."""

    da1: float = 1.0
    sap: float = 0.1
    bh: float = 0.5
    bk: float = 0.5
    bz: float = 0.5


def _reduce01(value: float) -> float:
    return value - math.floor(value + 1e-4)


def _canonical_fraction(value: float, *, tol: float = 1.0e-5) -> float:
    """Reduce a periodic coordinate and snap values near the cell boundary to zero."""

    reduced = _reduce01(value)
    if abs(reduced) < tol or abs(reduced - 1.0) < tol:
        return 0.0
    return reduced


def _canonical_height(value: float, period: float | None = None, *, tol: float = 1.0e-5) -> float:
    """Snap near-zero and near-periodic heights to a clean canonical value."""

    if abs(value) < tol:
        return 0.0
    if period is not None and abs(value - period) < tol:
        return 0.0
    return value


def _parse_vector_text(text: str) -> tuple[float, float, float]:
    """Parse a comma-separated vector such as ``1.0,0.0,0.0``."""

    values = [float(token) for token in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"expected three comma-separated vector components, got: {text}")
    return values[0], values[1], values[2]


def _symbol_to_atomic_number(symbol: str) -> int:
    cleaned = symbol.strip().capitalize()
    if cleaned not in _ATOMIC_NUMBERS:
        raise ValueError(f"unsupported element symbol: {symbol}")
    return _ATOMIC_NUMBERS[cleaned]


def _build_cell_vectors(
    a_length: float,
    b_length: float,
    c_length: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Construct Cartesian cell vectors from the CIF cell constants."""

    alpha = math.radians(alpha_deg)
    beta = math.radians(beta_deg)
    gamma = math.radians(gamma_deg)
    sin_gamma = math.sin(gamma)
    if abs(sin_gamma) < 1e-12:
        raise ValueError("gamma angle leads to a singular in-plane cell")

    a_vec = (a_length, 0.0, 0.0)
    b_vec = (b_length * math.cos(gamma), b_length * math.sin(gamma), 0.0)
    c_x = c_length * math.cos(beta)
    c_y = c_length * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / sin_gamma
    c_z_sq = c_length * c_length - c_x * c_x - c_y * c_y
    c_z = math.sqrt(max(c_z_sq, 0.0))
    return a_vec, b_vec, (c_x, c_y, c_z)


def _parse_cif(path: Path) -> ImportedStructure:
    """Parse a simple slab CIF into Cartesian coordinates."""

    lines = path.read_text(encoding="utf-8").splitlines()
    scalars: dict[str, str] = {}
    atom_headers: list[str] = []
    atom_rows: list[list[str]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if stripped.startswith("_"):
            parts = stripped.split(maxsplit=1)
            scalars[parts[0]] = parts[1] if len(parts) > 1 else ""
            index += 1
            continue
        if stripped == "loop_":
            index += 1
            headers: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("_"):
                headers.append(lines[index].strip())
                index += 1
            rows: list[list[str]] = []
            while index < len(lines):
                row = lines[index].strip()
                if not row or row.startswith("#"):
                    index += 1
                    continue
                if row == "loop_" or row.startswith("_") or row.startswith("data_"):
                    break
                rows.append(shlex.split(row))
                index += 1
            if any(header.startswith("_atom_site_") for header in headers):
                atom_headers = headers
                atom_rows = rows
            continue
        index += 1

    required_scalars = [
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    ]
    for key in required_scalars:
        if key not in scalars:
            raise ValueError(f"missing required CIF scalar: {key}")
    if not atom_headers or not atom_rows:
        raise ValueError("no atom-site loop found in CIF file")

    header_index = {name: idx for idx, name in enumerate(atom_headers)}
    if "_atom_site_type_symbol" in header_index:
        symbol_column = "_atom_site_type_symbol"
    elif "_atom_site_label" in header_index:
        symbol_column = "_atom_site_label"
    else:
        raise ValueError("CIF atom loop must contain _atom_site_type_symbol or _atom_site_label")

    for key in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"):
        if key not in header_index:
            raise ValueError(f"missing required CIF atom column: {key}")

    a_vec, b_vec, c_vec = _build_cell_vectors(
        float(scalars["_cell_length_a"]),
        float(scalars["_cell_length_b"]),
        float(scalars["_cell_length_c"]),
        float(scalars["_cell_angle_alpha"]),
        float(scalars["_cell_angle_beta"]),
        float(scalars["_cell_angle_gamma"]),
    )
    lattice = np.array([a_vec, b_vec, c_vec], dtype=np.float64)

    atoms: list[ImportedAtom] = []
    for row in atom_rows:
        symbol_token = row[header_index[symbol_column]]
        symbol_match = re.match(r"[A-Za-z]+", symbol_token)
        if symbol_match is None:
            raise ValueError(f"could not infer element symbol from CIF token: {symbol_token}")
        symbol = symbol_match.group(0).capitalize()
        occupancy = 1.0
        if "_atom_site_occupancy" in header_index:
            occupancy = float(row[header_index["_atom_site_occupancy"]])

        fractional = np.array(
            [
                float(row[header_index["_atom_site_fract_x"]]),
                float(row[header_index["_atom_site_fract_y"]]),
                float(row[header_index["_atom_site_fract_z"]]),
            ],
            dtype=np.float64,
        )
        cartesian = fractional @ lattice
        atoms.append(
            ImportedAtom(
                symbol=symbol,
                atomic_number=_symbol_to_atomic_number(symbol),
                occupancy=occupancy,
                x_angstrom=float(cartesian[0]),
                y_angstrom=float(cartesian[1]),
                z_angstrom=float(cartesian[2]),
            )
        )

    return ImportedStructure(
        atoms=atoms,
        a_vector_angstrom=a_vec,
        b_vector_angstrom=b_vec,
        c_vector_angstrom=c_vec,
        source_path=path,
    )


def _parse_xyz(
    path: Path,
    *,
    a_vector_override: tuple[float, float, float] | None = None,
    b_vector_override: tuple[float, float, float] | None = None,
    c_vector_override: tuple[float, float, float] | None = None,
) -> ImportedStructure:
    """Parse an XYZ or extxyz slab structure into Cartesian coordinates."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError("XYZ file must contain at least two lines")
    atom_count = int(lines[0].strip())
    comment = lines[1].strip()

    a_vec = b_vec = c_vec = None
    lattice_match = _LATTICE_RE.search(comment)
    if lattice_match is not None:
        values = [float(token) for token in lattice_match.group(1).split()]
        if len(values) != 9:
            raise ValueError("extxyz Lattice field must contain 9 floating-point values")
        a_vec = (values[0], values[1], values[2])
        b_vec = (values[3], values[4], values[5])
        c_vec = (values[6], values[7], values[8])

    if a_vector_override is not None:
        a_vec = a_vector_override
    if b_vector_override is not None:
        b_vec = b_vector_override
    if c_vector_override is not None:
        c_vec = c_vector_override
    if a_vec is None or b_vec is None:
        raise ValueError("XYZ import requires either an extxyz Lattice field or explicit --a-vector and --b-vector overrides")
    if c_vec is None:
        c_vec = (0.0, 0.0, 0.0)

    atom_lines = lines[2 : 2 + atom_count]
    if len(atom_lines) != atom_count:
        raise ValueError("XYZ atom count does not match the number of coordinate rows")

    atoms: list[ImportedAtom] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"invalid XYZ atom row: {line}")
        symbol = parts[0].capitalize()
        atoms.append(
            ImportedAtom(
                symbol=symbol,
                atomic_number=_symbol_to_atomic_number(symbol),
                occupancy=1.0,
                x_angstrom=float(parts[1]),
                y_angstrom=float(parts[2]),
                z_angstrom=float(parts[3]),
            )
        )

    return ImportedStructure(
        atoms=atoms,
        a_vector_angstrom=a_vec,
        b_vector_angstrom=b_vec,
        c_vector_angstrom=c_vec,
        source_path=path,
    )


def load_imported_structure(
    path: Path,
    *,
    a_vector_override: tuple[float, float, float] | None = None,
    b_vector_override: tuple[float, float, float] | None = None,
    c_vector_override: tuple[float, float, float] | None = None,
) -> ImportedStructure:
    """Load a slab structure from a supported CIF or XYZ-like file."""

    suffix = path.suffix.lower()
    if suffix == ".cif":
        return _parse_cif(path)
    if suffix in {".xyz", ".extxyz"}:
        return _parse_xyz(
            path,
            a_vector_override=a_vector_override,
            b_vector_override=b_vector_override,
            c_vector_override=c_vector_override,
        )
    raise ValueError(f"unsupported structure-file extension: {path.suffix}")


def parse_beam_argument(text: str) -> tuple[int, int]:
    """Parse one beam token such as ``0,-1`` into solver beam indices."""

    tokens = [token.strip() for token in text.split(",")]
    if len(tokens) != 2:
        raise ValueError(f"beam specification must look like IH,IK; got: {text}")
    return int(tokens[0]), int(tokens[1])


def parse_element_param_argument(text: str) -> tuple[str, ElementParameters]:
    """Parse ``Element:da1,sap,bh,bk,bz`` overrides for imported structures."""

    if ":" not in text:
        raise ValueError(f"element override must look like Symbol:da1,sap,bh,bk,bz; got: {text}")
    symbol_text, values_text = text.split(":", maxsplit=1)
    values = [float(token) for token in values_text.split(",")]
    if len(values) != 5:
        raise ValueError(f"element override must provide five numeric values; got: {text}")
    symbol = symbol_text.strip().capitalize()
    return symbol, ElementParameters(*values)


def _cell_metrics(structure: ImportedStructure) -> tuple[float, float, float]:
    """Return ``aa``, ``bb``, and ``gamma`` from the imported in-plane cell."""

    a_xy = np.array(structure.a_vector_angstrom[:2], dtype=np.float64)
    b_xy = np.array(structure.b_vector_angstrom[:2], dtype=np.float64)
    aa = float(np.linalg.norm(a_xy))
    bb = float(np.linalg.norm(b_xy))
    if aa <= 0.0 or bb <= 0.0:
        raise ValueError("imported a and b cell vectors must have nonzero in-plane length")
    gamma_cos = float(np.dot(a_xy, b_xy) / (aa * bb))
    gamma_deg = math.degrees(math.acos(max(-1.0, min(1.0, gamma_cos))))
    return aa, bb, gamma_deg


def _fractional_xy(structure: ImportedStructure) -> np.ndarray:
    """Return the 2x2 matrix used to convert Cartesian x/y into in-plane fractions."""

    a_xy = np.array(structure.a_vector_angstrom[:2], dtype=np.float64)
    b_xy = np.array(structure.b_vector_angstrom[:2], dtype=np.float64)
    matrix = np.column_stack((a_xy, b_xy))
    if abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("imported in-plane cell vectors are linearly dependent")
    return matrix


def _parameter_for_symbol(
    symbol: str,
    overrides: dict[str, ElementParameters],
    defaults: ElementParameters,
) -> ElementParameters:
    if symbol in overrides:
        return overrides[symbol]
    if symbol in _DEFAULT_ELEMENT_PARAMS:
        values = _DEFAULT_ELEMENT_PARAMS[symbol]
        return ElementParameters(*values)
    return defaults


def _format_bulk_text(bulk: BulkInput) -> str:
    """Serialize one ``BulkInput`` object into a ``bulk.txt`` file."""

    lines = [
        f"{bulk.nh},{bulk.nk},{bulk.ndom}                                 ,NH,NK,NDOM",
        f"{sum(bulk.nb)}                                     ,NB",
        ",".join(f"{value:.10g}" for value in bulk.rdom_deg) + "                                     ,RDOM",
        ",".join(f"{ih},{ik}" for ih, ik in zip(bulk.ih, bulk.ik)) + "                ,(IH(I),IK(I))",
        f"{bulk.be:.10g},{bulk.azi_deg:.10g},{bulk.azf_deg:.10g},{bulk.daz_deg:.10g},{bulk.gi_deg:.10g},{bulk.gf_deg:.10g},{bulk.dg_deg:.10g}                  ,BE,AZI,AZF,DAZ,GI,GF,DG",
        f"{bulk.dz_input:.10g},{bulk.ml}                              ,DZ,ML",
        f"{bulk.nelm}                                     ,NELM",
    ]
    for iz, da1, sap, bh, bk, bz in zip(bulk.iz, bulk.da1, bulk.sap, bulk.bh, bulk.bk, bulk.bz):
        lines.append(f"{iz},{da1:.10g},{sap:.10g}                            ,Z,da1,sap")
        lines.append(f"{bh:.10g},{bk:.10g},{bz:.10g}                           ,BH,BK,BZ")
    lines.append(
        f"{bulk.nsg},{bulk.aa:.10g},{bulk.bb:.10g},{bulk.gam_deg:.10g},{bulk.cc:.10g},{bulk.dx:.10g},{bulk.dy:.10g}        ,NSG,AA,BB,GAM,CC,DX,DY"
    )
    lines.append(f"{bulk.natm}                                     ,NATM")
    for ielm, ocr, x, y, z in zip(bulk.ielm, bulk.ocr, bulk.x, bulk.y, bulk.z):
        lines.append(f"{ielm},{ocr:.10g},{x:.10g},{y:.10g},{z:.10g}            ,IELM,ocr,X,Y,Z")
    return "\n".join(lines) + "\n"


def _format_surface_text(surface: SurfaceInput) -> str:
    """Serialize one ``SurfaceInput`` object into a ``surf.txt`` file."""

    lines = [f"{surface.nelms}                                     ,NELMS"]
    for iz, da1, sap, bh, bk, bz in zip(surface.iz, surface.da1, surface.sap, surface.bh, surface.bk, surface.bz):
        lines.append(f"{iz},{da1:.10g},{sap:.10g}                            ,Z,da1,sap")
        lines.append(f"{bh:.10g},{bk:.10g},{bz:.10g}                           ,BH,BK,BZ")
    lines.append(
        f"{surface.nsgs},{surface.msa},{surface.msb},{surface.nsa},{surface.nsb},{surface.dthick:.10g},{surface.dxs:.10g},{surface.dys:.10g}                 ,NSGS,msa,msb,nsa,nsb,dthick,DXS,DYS"
    )
    lines.append(f"{surface.natms}                                     ,NATM")
    for ielm, ocr, x, y, z in zip(surface.ielm, surface.ocr, surface.x, surface.y, surface.z):
        lines.append(f"{ielm},{ocr:.10g},{x:.10g},{y:.10g},{z:.10g}            ,IELM,ocr,X,Y,Z")
    lines.append(",".join(f"{weight:.10g}" for weight in surface.wdom) + "                                     ,WDOM")
    return "\n".join(lines) + "\n"


def write_solver_inputs(
    bulk: BulkInput,
    surface: SurfaceInput,
    *,
    bulk_path: Path,
    surface_path: Path,
) -> None:
    """Write ``bulk.txt`` and ``surf.txt`` files created from an imported slab."""

    bulk_path.parent.mkdir(parents=True, exist_ok=True)
    surface_path.parent.mkdir(parents=True, exist_ok=True)
    bulk_path.write_text(_format_bulk_text(bulk), encoding="utf-8")
    surface_path.write_text(_format_surface_text(surface), encoding="utf-8")


def build_solver_inputs_from_structure(
    structure: ImportedStructure,
    *,
    surface_z_min: float,
    bulk_c_length: float,
    surface_thickness: float | None = None,
    beam_indices: list[tuple[int, int]],
    be: float = 15.0,
    azi_deg: float = 0.0,
    azf_deg: float = 0.0,
    daz_deg: float = 0.0,
    gi_deg: float = 0.1,
    gf_deg: float = 7.0,
    dg_deg: float = 0.1,
    dz_input: float = 0.05,
    ml: int = 200,
    nh: int = 1,
    nk: int = 1,
    bulk_dx: float = 0.0,
    bulk_dy: float = 0.0,
    surface_dxs: float = 0.0,
    surface_dys: float = 0.0,
    nsg: int = 1,
    nsgs: int = 1,
    msa: int = 1,
    msb: int = 0,
    nsa: int = 0,
    nsb: int = 1,
    default_element_parameters: ElementParameters = ElementParameters(),
    element_parameter_overrides: dict[str, ElementParameters] | None = None,
) -> tuple[BulkInput, SurfaceInput]:
    """Convert one imported slab structure into ``bulk.txt`` and ``surf.txt`` inputs."""

    if not structure.atoms:
        raise ValueError("imported structure contains no atoms")
    if bulk_c_length <= 0.0:
        raise ValueError("bulk_c_length must be positive")
    if not beam_indices:
        raise ValueError("at least one beam index must be provided")

    overrides = element_parameter_overrides or {}
    in_plane_matrix = _fractional_xy(structure)
    aa, bb, gamma_deg = _cell_metrics(structure)
    surface_transform = np.array([[float(msa), float(nsa)], [float(msb), float(nsb)]], dtype=np.float64)
    if abs(np.linalg.det(surface_transform)) < 1e-12:
        raise ValueError("surface cell transform is singular")
    surface_transform_inv = np.linalg.inv(surface_transform)

    split_tol = 1.0e-5
    surface_atoms = [atom for atom in structure.atoms if atom.z_angstrom >= surface_z_min - split_tol]
    bulk_atoms_all = [atom for atom in structure.atoms if atom.z_angstrom < surface_z_min - split_tol]
    if not surface_atoms:
        raise ValueError("surface_z_min leaves no atoms in the surface region")
    if not bulk_atoms_all:
        raise ValueError("surface_z_min leaves no atoms in the bulk region")

    bulk_origin_z = surface_z_min - bulk_c_length
    unique_symbols = sorted({atom.symbol for atom in structure.atoms}, key=lambda symbol: _symbol_to_atomic_number(symbol), reverse=True)
    unique_atomic_numbers = [_symbol_to_atomic_number(symbol) for symbol in unique_symbols]
    symbol_to_ielm = {symbol: index + 1 for index, symbol in enumerate(unique_symbols)}

    def dedupe_key(symbol: str, frac_x: float, frac_y: float, z_value: float) -> tuple[str, int, int, int]:
        return symbol, round(frac_x * 1.0e5), round(frac_y * 1.0e5), round(z_value * 1.0e5)

    bulk_records: dict[tuple[str, int, int, int], tuple[int, float, float, float, float]] = {}
    for atom in bulk_atoms_all:
        frac_xy = np.linalg.solve(in_plane_matrix, np.array([atom.x_angstrom, atom.y_angstrom], dtype=np.float64))
        frac_x = _canonical_fraction(float(frac_xy[0]))
        frac_y = _canonical_fraction(float(frac_xy[1]))
        z_value = _canonical_height((atom.z_angstrom - bulk_origin_z) % bulk_c_length, bulk_c_length)
        key = dedupe_key(atom.symbol, frac_x, frac_y, z_value)
        bulk_records[key] = (symbol_to_ielm[atom.symbol], atom.occupancy, frac_x, frac_y, z_value)

    surface_records: dict[tuple[str, int, int, int], tuple[int, float, float, float, float]] = {}
    for atom in surface_atoms:
        frac_xy = np.linalg.solve(in_plane_matrix, np.array([atom.x_angstrom, atom.y_angstrom], dtype=np.float64))
        surface_frac = surface_transform_inv @ frac_xy
        surface_frac_x = _canonical_fraction(float(surface_frac[0]))
        surface_frac_y = _canonical_fraction(float(surface_frac[1]))
        bulk_frac = surface_transform @ np.array([surface_frac_x, surface_frac_y], dtype=np.float64)
        x_value = _canonical_fraction(float(bulk_frac[0]))
        y_value = _canonical_fraction(float(bulk_frac[1]))
        z_value = _canonical_height(atom.z_angstrom - surface_z_min)
        key = dedupe_key(atom.symbol, x_value, y_value, z_value)
        surface_records[key] = (symbol_to_ielm[atom.symbol], atom.occupancy, x_value, y_value, z_value)

    bulk_rows = sorted(bulk_records.values(), key=lambda row: (row[4], row[0], row[2], row[3]))
    surface_rows = sorted(surface_records.values(), key=lambda row: (row[4], row[0], row[2], row[3]))
    if not bulk_rows or not surface_rows:
        raise RuntimeError("failed to construct non-empty bulk and surface atom lists from the imported slab")

    da1_list: list[float] = []
    sap_list: list[float] = []
    bh_list: list[float] = []
    bk_list: list[float] = []
    bz_list: list[float] = []
    for symbol in unique_symbols:
        params = _parameter_for_symbol(symbol, overrides, default_element_parameters)
        da1_list.append(params.da1)
        sap_list.append(params.sap)
        bh_list.append(params.bh)
        bk_list.append(params.bk)
        bz_list.append(params.bz)

    beam_ih = [beam[0] for beam in beam_indices]
    beam_ik = [beam[1] for beam in beam_indices]
    if surface_thickness is None:
        estimated_thickness = max(row[4] for row in surface_rows) - min(row[4] for row in surface_rows) if surface_rows else 0.0
        dthick = max(estimated_thickness, 0.5 * bulk_c_length)
    else:
        if surface_thickness < 0.0:
            raise ValueError("surface_thickness must be non-negative")
        dthick = surface_thickness

    bulk_input = BulkInput(
        nh=nh,
        nk=nk,
        ndom=1,
        nb=[len(beam_indices)],
        rdom_deg=[0.0],
        ih=beam_ih,
        ik=beam_ik,
        be=be,
        azi_deg=azi_deg,
        azf_deg=azf_deg,
        daz_deg=daz_deg,
        gi_deg=gi_deg,
        gf_deg=gf_deg,
        dg_deg=dg_deg,
        dz_input=dz_input,
        ml=ml,
        nelm=len(unique_symbols),
        iz=unique_atomic_numbers,
        da1=da1_list,
        sap=sap_list,
        bh=bh_list,
        bk=bk_list,
        bz=bz_list,
        nsg=nsg,
        aa=aa,
        bb=bb,
        gam_deg=gamma_deg,
        cc=bulk_c_length,
        dx=bulk_dx,
        dy=bulk_dy,
        natm=len(bulk_rows),
        ielm=[row[0] for row in bulk_rows],
        ocr=[row[1] for row in bulk_rows],
        x=[row[2] for row in bulk_rows],
        y=[row[3] for row in bulk_rows],
        z=[row[4] for row in bulk_rows],
        source_path=structure.source_path,
    )
    surface_input = SurfaceInput(
        nelms=len(unique_symbols),
        iz=unique_atomic_numbers,
        da1=da1_list,
        sap=sap_list,
        bh=bh_list,
        bk=bk_list,
        bz=bz_list,
        nsgs=nsgs,
        msa=msa,
        msb=msb,
        nsa=nsa,
        nsb=nsb,
        dthick=dthick,
        dxs=surface_dxs,
        dys=surface_dys,
        natms=len(surface_rows),
        ielm=[row[0] for row in surface_rows],
        ocr=[row[1] for row in surface_rows],
        x=[row[2] for row in surface_rows],
        y=[row[3] for row in surface_rows],
        z=[row[4] for row in surface_rows],
        wdom=[1.0],
        source_path=structure.source_path,
    )
    return bulk_input, surface_input
