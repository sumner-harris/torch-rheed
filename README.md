# torch-rheed

`torch-rheed` is a Python-native, PyTorch implementation of the RHEED forward
calculation described in the
[sim-trhepd-rheed paper](https://doi.org/10.1016/j.cpc.2022.108371). It reads
the conventional `bulk.txt` and `surf.txt` structure files, solves the
dynamical diffraction problem without calling the original Fortran
executables, and returns rocking curves and optional synthetic detector
images.

The package currently implements the electron/RHEED, single-domain, `p1`
subset of sim-trhepd-rheed. It is not an official release of the upstream
project and does not yet reproduce every TRHEPD/RHEED input mode or plane
group.

## Features

- Original sim-trhepd-rheed-style `bulk.txt` and `surf.txt` input parsing
- Dynamical bulk and surface diffraction calculations in PyTorch
- CPU and CUDA execution through the same Python API
- Sixth-order SP6 surface solver and a conventional multislice solver
- Numerically stable scattering-matrix composition for evanescent beams
- Single calculations and batched surface-structure calculations
- Rocking-curve text, CSV, and plot outputs
- Synthetic crystal-truncation-rod (CTR) detector images, previews, and GIFs
- Configurable source divergence and detector point-spread broadening
- CIF/XYZ import, structure visualization, and CIF/XYZ export
- Approximate polycrystalline screens from orientation averaging

## Installation

Python 3.13 or newer is required. With [`uv`](https://docs.astral.sh/uv/):

```powershell
uv sync
```

Or install the package into an existing environment:

```powershell
python -m pip install -e .
```

The command-line entry point is `torch-rheed`. The equivalent module form is
`python -m torch_rheed`.

## Quick start

Run a RHEED rocking-curve calculation and place all generated files under the
ignored local `data/` directory:

```powershell
uv run torch-rheed simulate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --out-dir data\my_run `
  --device cpu
```

This produces:

```text
data/my_run/
├── surf-bulkE.s
├── rocking_curve.csv
├── rocking_curve.png
├── screen_stack.pt
└── screen_preview.png
```

Everything under `data/` is ignored by Git. Use explicit output paths if you
want results elsewhere.

## Calculation method

The implementation follows the forward model used by sim-trhepd-rheed: the
three-dimensional wavefunction is expanded in in-plane reciprocal-lattice
beams, reducing the stationary dynamical diffraction equation to coupled
equations along the surface-normal direction, `z`.

### 1. Scattering potential

For each reciprocal-lattice component, `torch-rheed` builds complex Fourier
coefficients of the crystal potential from:

- atomic numbers and tabulated electron scattering-factor coefficients;
- the beam energy and relativistically corrected electron wave number;
- atomic coordinates and occupancies;
- in-plane and surface-normal Debye-Waller parameters;
- the `da1` scattering-factor correction; and
- the `sap` absorptive-potential parameter.

The bulk and surface potentials are sampled along `z`. In-plane coupling is
computed for the diffraction beams listed in `bulk.txt`.

### 2. Semi-infinite bulk

The repeated bulk unit is divided into slices using `DZ`. Each constant-potential
slice is solved by matrix eigendecomposition and converted from a transfer
matrix to bounded two-port scattering blocks. Adjacent slices are composed
with the Redheffer star product, avoiding the exponentially growing amplitudes
that make direct transfer-matrix multiplication unstable for evanescent
beams.

The completed unit-cell scattering matrix is terminated recursively against
additional bulk repeats until the reflected intensity converges or the `ML`
repeat limit is reached. The resulting reflection matrix supplies the lower
boundary condition for the surface calculation.

### 3. Surface propagation

Two surface solvers are available:

| Solver | Description |
| --- | --- |
| `sp6` | Default. Sixth-order, 11-stage symmetric `SRKN^b_11` BAB splitting with the right-hand-side transformation (RHST), following Kudo, Yamamoto, and Hoshi. It avoids a matrix eigensolve at every surface step. |
| `multislice` | Conventional piecewise-constant multislice propagation using a matrix eigendecomposition for each surface slice. Useful as a reference calculation. |

For SP6, `--integration-step` specifies the requested step in angstroms. The
actual step is adjusted slightly so the fixed surface boundary is reached
exactly. If the option is omitted, the requested step is ten times the
effective bulk slice spacing. `--rhst-threshold` controls when the RHST
normalization is applied and defaults to `1000`, as in the reference method.

For convergence-sensitive work, repeat the calculation with half the SP6
integration step and compare the rocking curves.

### 4. Rocking-curve intensity

At every requested azimuth/glancing-angle point, the surface reflection matrix
is joined to the converged bulk reflection condition. Beam intensity is
calculated from the reflected amplitude with the outgoing-to-incident
surface-normal flux factor. Non-propagating outgoing beams are assigned zero
measured intensity.

### 5. Synthetic detector images

The optional detector renderer maps each calculated beam to its reciprocal-space
crystal truncation rod and intersects those rods with a fixed detector plane.
The rocking-curve intensity is interpolated along each rod; a Lorentzian or
Gaussian transverse rod profile represents finite in-plane correlation
length. The renderer can also apply:

- Gaussian quadrature over glancing-angle source divergence;
- an isotropic detector point-spread function specified by physical FWHM;
- vertical or specular-normal detector geometry; and
- a hard mask for detector pixels hidden by the substrate.

These screen images are a visualization derived from the calculated rocking
curves. They are an extension of the core forward calculation, not a claim of
full experimental detector or Kikuchi-feature simulation.

## Inputs

The parser consumes the numeric fields of the original comma-delimited input
format. Descriptive text after the numeric fields on a line is allowed. Lengths
are in angstroms, beam energy is in keV, and angles are in degrees.

### `bulk.txt`

`bulk.txt` defines the incident beam, scan, diffraction-beam basis, bulk unit
cell, atomic potential parameters, and numerical bulk controls.

| Field group | Contents |
| --- | --- |
| Beam indexing | `NH`, `NK`, number of domains, beam count, domain rotation, and each `(IH, IK)` diffraction index |
| Scan | Beam energy `BE`; initial/final/step azimuth `AZI`, `AZF`, `DAZ`; initial/final/step glancing angle `GI`, `GF`, `DG` |
| Bulk numerics | Requested slice spacing `DZ` and maximum repeat count `ML` |
| Element parameters | Atomic number `IZ`, scattering correction `DA1`, absorption `SAP`, and Debye-Waller terms `BH`, `BK`, `BZ` |
| Unit cell | Plane-group code `NSG`, lengths `AA`, `BB`, angle `GAM`, normal repeat `CC`, and inter-repeat shifts `DX`, `DY` |
| Atoms | Element index `IELM`, occupancy `OCR`, fractional in-plane coordinates `X`, `Y`, and normal coordinate `Z` in angstroms |

The current implementation requires one domain and `NSG` equal to `0` or `1`
(`p1`). The `(0, 0)` beam must be included.

### `surf.txt`

`surf.txt` defines the non-periodic surface region placed above the bulk.

| Field group | Contents |
| --- | --- |
| Element parameters | Surface `NELMS` followed by `IZ`, `DA1`, `SAP`, `BH`, `BK`, and `BZ` for each parameter set |
| Surface cell | `NSGS`; integer cell transform `MSA`, `MSB`, `NSA`, `NSB`; vacuum-tail thickness `DTHICK`; and surface shifts `DXS`, `DYS` |
| Atoms | Surface `NATMS` followed by `IELM`, `OCR`, `X`, `Y`, and `Z` for each independent atom |
| Domain weights | Optional `WDOM` values; omitted values default to one |

The current implementation requires `NSGS` equal to `0` or `1` (`p1`). For
batched surface calculations that share one bulk solve, all surfaces must also
use identical element-parameter blocks.

### Structure-file conversion

A slab in CIF, XYZ, or extended XYZ format can be split into compatible input
files:

```powershell
uv run torch-rheed import-structure `
  --input C:\path\to\slab.cif `
  --surface-z-min 21.62 `
  --bulk-c-length 3.905 `
  --surface-thickness 2.0 `
  --beam 0,-2 --beam 0,-1 --beam 0,0 --beam 0,1 --beam 0,2 `
  --out-dir data\generated_inputs
```

`surface-z-min` selects the slab atoms assigned to the surface region,
`bulk-c-length` defines one repeated bulk unit along `z`, and
`surface-thickness` becomes `DTHICK`. Plain XYZ files require explicit in-plane
cell vectors unless a `Lattice` record is available.

## Outputs

| Output | Contents |
| --- | --- |
| `surf-bulkE.s` | Conventional sim-trhepd-rheed-style electron rocking-curve text: scan metadata, beam indices, angles, and intensities |
| `rocking_curve.csv` | `angle_deg` plus one intensity column per `(IH, IK)` beam |
| `rocking_curve.png` | Plot of the calculated beam intensities versus scan angle |
| `screen_stack.pt` | PyTorch detector tensor with shape `(N, H, W)` |
| `screen_preview.png` | One detector frame near the selected reference angle |
| Optional GIF | Animated detector frames written when `--screen-gif-output` is supplied |

`N` is the number of scan points and `H × W` is the requested detector raster.
The Python batch API returns intensity tensors shaped `(B, N, NB)` and detector
stacks shaped `(B, N, H, W)`, where `B` is the number of structures and `NB` is
the number of diffraction beams.

Unlike the original executable workflow, the bulk intermediate state remains
in memory; no `bulkE.b` file is required.

## Command-line examples

### Select the surface solver

```powershell
uv run torch-rheed simulate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --solver sp6 `
  --integration-step 0.2 `
  --out-dir data\sp6_run
```

Use `--solver multislice` to run the conventional surface solver.

### Configure detector rendering

```powershell
uv run torch-rheed simulate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --out-dir data\detector_run `
  --screen-distance-mm 200 `
  --screen-width-mm 140 `
  --screen-height-mm 100 `
  --screen-pixels-x 512 `
  --screen-pixels-y 384 `
  --correlation-length-angstrom 1000 `
  --instrument-broadening-fwhm-mm 1.4634 `
  --screen-gif-output data\detector_run\screen.gif
```

Detector rendering currently requires a fixed azimuth scan and at least two
glancing-angle samples.

### Validate against a reference curve

```powershell
uv run torch-rheed validate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --reference C:\path\to\reference\surf-bulkE.s `
  --comparison-plot data\validation\comparison.png
```

### Plot an existing output

```powershell
uv run torch-rheed plot `
  --input C:\path\to\surf-bulkE.s `
  --output data\rocking_curve.png
```

### Visualize the input structure

```powershell
uv run torch-rheed structure `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --a-units 2 `
  --b-units 2 `
  --bulk-layers 4 `
  --out-dir data\structure
```

This writes combined, bulk-only, and surface-only plots plus viewer-friendly
XYZ and P1 CIF files.

Run `uv run torch-rheed --help` or `uv run torch-rheed <command> --help` for
the complete option list.

## Python API

```python
from pathlib import Path

from torch_rheed import ScreenImageConfig, plot_rocking_curve, simulate_from_files

result = simulate_from_files(
    Path("inputs/bulk.txt"),
    Path("inputs/surf.txt"),
    device="cpu",                 # or "cuda:0"
    solver="sp6",
    integration_step=0.2,
    screen_config=ScreenImageConfig(
        screen_distance_mm=200.0,
        screen_width_mm=140.0,
        screen_height_mm=100.0,
        pixels_x=512,
        pixels_y=384,
        instrument_broadening_fwhm_mm=1.4634,
    ),
)

result.write_surface_output(Path("data/run/surf-bulkE.s"))
result.write_csv(Path("data/run/rocking_curve.csv"))
result.write_screen_stack(Path("data/run/screen_stack.pt"))
plot_rocking_curve(result, Path("data/run/rocking_curve.png"))
```

For parameter sweeps, `simulate_from_files_batch(...)` reuses one bulk
calculation across compatible `surf.txt` files. `simulate_pairs_batch(...)`
accepts arbitrary `(bulk.txt, surf.txt)` pairs, groups compatible work, and
preserves input order.

## References

The physical model and legacy input/output conventions come from:

- T. Hanada, Y. Motoyama, K. Yoshimi, and T. Hoshi,
  “sim-trhepd-rheed – Open-source simulator of total-reflection high-energy
  positron diffraction (TRHEPD) and reflection high-energy electron diffraction
  (RHEED),” *Computer Physics Communications* **277**, 108371 (2022).
  [DOI](https://doi.org/10.1016/j.cpc.2022.108371) ·
  [arXiv](https://arxiv.org/abs/2110.09477) ·
  [reference implementation](https://github.com/sim-trhepd-rheed/sim-trhepd-rheed)

The default SP6 surface integrator follows:

- S. Kudo, Y. Yamamoto, and T. Hoshi, “A fast and accurate computation method
  for reflective diffraction simulations,” *Computer Physics Communications*
  **296**, 109029 (2024).
  [DOI](https://doi.org/10.1016/j.cpc.2023.109029) ·
  [arXiv](https://arxiv.org/abs/2306.00271)

When publishing calculations performed with this package, cite the relevant
method papers above and state the solver, integration step, beam set, and
convergence checks used.
