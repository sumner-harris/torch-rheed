# torch_rheed

`torch_rheed` is a standalone PyTorch package for simulating RHEED rocking
curves from the original `sim-trhepd-rheed` text inputs. The package reads
`bulk.txt` and `surf.txt`, computes the bulk and surface scattering in Python,
and writes rocking-curve outputs in the conventional `surf-bulkE.s` text
format, CSV, plot form, and synthetic detector-image stacks.

## Current scope

- RHEED / electron-mode calculations
- single-domain inputs
- `p1` symmetry inputs
- rocking-curve simulation from `bulk.txt` + `surf.txt`
- synthetic RHEED screen-image generation alongside each rocking-curve run
- polycrystalline screen approximation by averaging many oriented cubic grains
- plotting utilities for simulated or saved rocking-curve outputs
- bulk+surface structure visualization and export to `XYZ` and `CIF`

The package does not require Fortran executables or intermediate Fortran output
files at runtime.

## Installation

Create a local `uv` virtual environment and install the package in editable
mode:

```powershell
uv venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

## Command-line usage

Run a simulation and write the conventional text output, CSV, and a plot:

```powershell
torch-rheed simulate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --device cpu `
  --out-dir C:\path\to\results
```

Use the sixth-order SP6 surface solver from Kudo, Yamamoto, and Hoshi with:

```powershell
torch-rheed simulate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --solver sp6 `
  --integration-step 0.2 `
  --device cpu `
  --out-dir C:\path\to\results_sp6
```

`--integration-step` is the requested surface integration step in Angstrom.
The solver adjusts it slightly so that the existing integration boundary is
covered exactly. If it is omitted, SP6 requests ten times the effective `DZ`
from `bulk.txt`. SP6 is the default surface solver. The original
eigendecomposition-based solver remains available with `--solver multislice`.

SP6 implements the paper's sixth-order, 11-stage symmetric `SRKN^b_11` BAB
splitting and the right-hand-side transformation (RHST). The RHST threshold
defaults to `1000`, matching the paper and original Fortran implementation;
it can be changed with `--rhst-threshold` for numerical experiments.

Implementation references:

- Kudo, Yamamoto, and Hoshi, *Computer Physics Communications* 296 (2024),
  [doi:10.1016/j.cpc.2023.109029](https://doi.org/10.1016/j.cpc.2023.109029)
- Original Fortran [`surf_prkn.f90`](https://github.com/shuheikudo/trhepd-opt/blob/main/sim-trhepd-rheed/src/surf_prkn.f90)
  and [`matcomp.f90`](https://github.com/shuheikudo/trhepd-opt/blob/main/sim-trhepd-rheed/src/matcomp.f90)

For a new material or beam set, verify convergence by repeating the SP6 run
with half the integration step. A representative 19-beam, 691-angle SrTiO3
case in this repository took 3.1 s at `0.2` Angstrom versus 89.1 s for the
legacy surface solve on the same CPU (28.6x faster); the maximum intensity
change between SP6 steps `0.2` and `0.05` Angstrom was below 5e-6 of the peak
intensity. Performance varies with beam count, angle count, device, and input.

With `--out-dir`, `simulate` also writes:

- `screen_stack.pt`: raw screen tensor shaped `(N, H, W)` for one structure
- `screen_preview.png`: one detector frame near the reference angle

You can customize detector rendering directly on the same call, for example:

```powershell
torch-rheed simulate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --out-dir C:\path\to\results `
  --screen-distance-mm 200 `
  --screen-width-mm 140 `
  --screen-height-mm 100 `
  --screen-pixels-x 512 `
  --screen-pixels-y 384 `
  --instrument-broadening-fwhm-mm 1.4634 `
  --plane-mode vertical
```

Plot an existing `surf-bulkE.s` or `surf-bulkP.s` file:

```powershell
torch-rheed plot `
  --input C:\path\to\surf-bulkE.s `
  --output C:\path\to\rocking_curve.png
```

Build a bulk+surface structure view and export files that ASE or VESTA can
read directly:

```powershell
torch-rheed structure `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --a-units 2 `
  --b-units 2 `
  --bulk-layers 4 `
  --bulk-plot-output C:\path\to\structure_bulk.png `
  --surface-plot-output C:\path\to\structure_surface.png `
  --out-dir C:\path\to\structure_outputs
```

Convert a slab `cif` or `xyz` file into `bulk.txt` and `surf.txt`:

```powershell
torch-rheed import-structure `
  --input C:\path\to\slab.cif `
  --surface-z-min 21.62 `
  --bulk-c-length 3.905 `
  --surface-thickness 2.0 `
  --beam 0,-2 `
  --beam 0,-1 `
  --beam 0,0 `
  --beam 0,1 `
  --beam 0,2 `
  --out-dir C:\path\to\imported_inputs
```

Validate a simulation against a saved reference output:

```powershell
torch-rheed validate `
  --bulk C:\path\to\bulk.txt `
  --surf C:\path\to\surf.txt `
  --reference C:\path\to\surf-bulkE.s `
  --comparison-plot C:\path\to\comparison.png
```

Approximate a polycrystalline screen by generating many cubic grain
orientations, pairing each one with a nearly-empty `surf.txt`, and averaging
their detector stacks in a fixed laboratory frame:

```powershell
torch-rheed polycrystal `
  --bulk C:\path\to\bulk.txt `
  --out-dir C:\path\to\polycrystal_results `
  --max-miller-index 2 `
  --azimuth-step-deg 90 `
  --beam-shell-radius 2 `
  --screen-reference-angle-deg 2.3 `
  --screen-frame-azimuth-deg 0
```

With `polycrystal`, `--out-dir` contains:

- `orientation_inputs\...`: generated `bulk.txt` / `surf.txt` pairs for every sampled grain orientation
- `screen_stack_batch.pt`: raw orientation batch stack shaped `(B, N, H, W)`
- `screen_stack_polycrystalline.pt`: integrated screen stack shaped `(N, H, W)`
- `screen_preview_polycrystalline.png`: preview frame near the reference angle
- `total_detector_intensity_vs_angle.csv` and `.png`
- `summary.txt`

If `simulate` is run without output-path arguments, the command writes the
conventional `surf-bulkE.s` text representation to standard output.

## Python API

```python
from pathlib import Path

from torch_rheed import ScreenImageConfig, plot_rocking_curve, plot_screen_frame, simulate_from_files, simulate_from_files_batch, simulate_pairs_batch
from torch_rheed import plot_polycrystal_screen_frame, plot_polycrystal_total_intensity, simulate_polycrystal_from_bulk
from torch_rheed import build_structure_from_files, plot_structure_views
from torch_rheed import build_solver_inputs_from_structure, load_imported_structure, write_solver_inputs

result = simulate_from_files(
    Path(r"C:\path\to\bulk.txt"),
    Path(r"C:\path\to\surf.txt"),
    device="cpu",
    screen_config=ScreenImageConfig(
        screen_distance_mm=200.0,
        pixels_x=512,
        pixels_y=384,
        instrument_broadening_fwhm_mm=1.4634,
    ),
)
result.write_surface_output(Path(r"C:\path\to\surf-bulkE.s"))
result.write_csv(Path(r"C:\path\to\rocking_curve.csv"))
result.write_screen_stack(Path(r"C:\path\to\screen_stack.pt"))
plot_rocking_curve(result, Path(r"C:\path\to\rocking_curve.png"))
plot_screen_frame(result, Path(r"C:\path\to\screen_preview.png"))

sp6_result = simulate_from_files(
    Path(r"C:\path\to\bulk.txt"),
    Path(r"C:\path\to\surf.txt"),
    device="cpu",
    screen_config=None,
    solver="sp6",
    integration_step=0.2,
)

batch = simulate_from_files_batch(
    Path(r"C:\path\to\bulk.txt"),
    [
        Path(r"C:\path\to\surf_a.txt"),
        Path(r"C:\path\to\surf_b.txt"),
        Path(r"C:\path\to\surf_c.txt"),
    ],
    device="cpu",
)
second_result = batch.get_result(1)
batch.write_screen_stack(Path(r"C:\path\to\screen_stack_batch.pt"))

pair_batch = simulate_pairs_batch(
    [
        (Path(r"C:\path\to\bulk_a.txt"), Path(r"C:\path\to\surf_a.txt")),
        (Path(r"C:\path\to\bulk_b.txt"), Path(r"C:\path\to\surf_b.txt")),
        (Path(r"C:\path\to\bulk_c.txt"), Path(r"C:\path\to\surf_c.txt")),
    ],
    device="cpu",
)
compatible_batch = pair_batch.as_batch_result()

poly = simulate_polycrystal_from_bulk(
    Path(r"C:\path\to\bulk.txt"),
    out_dir=Path(r"C:\path\to\polycrystal_results"),
    max_miller_index=2,
    azimuth_step_deg=90.0,
    beam_shell_radius=2,
    screen_config=ScreenImageConfig(
        screen_distance_mm=200.0,
        reference_angle_deg=2.3,
        frame_azimuth_deg=0.0,
    ),
)
poly.write_orientation_screen_stack(Path(r"C:\path\to\screen_stack_batch.pt"))
poly.write_integrated_screen_stack(Path(r"C:\path\to\screen_stack_polycrystalline.pt"))
poly.write_total_detector_csv(Path(r"C:\path\to\total_detector_intensity.csv"))
plot_polycrystal_screen_frame(poly, Path(r"C:\path\to\screen_preview_polycrystalline.png"))
plot_polycrystal_total_intensity(poly, Path(r"C:\path\to\total_detector_intensity.png"))

structure = build_structure_from_files(
    Path(r"C:\path\to\bulk.txt"),
    Path(r"C:\path\to\surf.txt"),
    a_units=2,
    b_units=2,
    bulk_layers=4,
)
structure.write_xyz(Path(r"C:\path\to\structure.xyz"))
structure.write_cif(Path(r"C:\path\to\structure.cif"))
plot_structure_views(structure, Path(r"C:\path\to\structure_views.png"))
plot_structure_views(structure, Path(r"C:\path\to\structure_bulk.png"), regions="bulk")
plot_structure_views(structure, Path(r"C:\path\to\structure_surface.png"), regions="surface")

imported = load_imported_structure(Path(r"C:\path\to\slab.cif"))
bulk_input, surface_input = build_solver_inputs_from_structure(
    imported,
    surface_z_min=21.62,
    bulk_c_length=3.905,
    surface_thickness=2.0,
    beam_indices=[(0, -2), (0, -1), (0, 0), (0, 1), (0, 2)],
)
write_solver_inputs(
    bulk_input,
    surface_input,
    bulk_path=Path(r"C:\path\to\bulk.txt"),
    surface_path=Path(r"C:\path\to\surf.txt"),
)
```

Any simulation entrypoint that accepts `device=` can target CPU or CUDA, for
example `device="cuda:0"`, as long as the local PyTorch install has that device
available.

Simulation entrypoints now return detector images together with the rocking
curve:

- `RockingCurveResult.intensities` has shape `(N, NB)`
- `RockingCurveBatchResult.intensities` has shape `(B, N, NB)`
- `RockingCurveResult.screen_images` has shape `(N, H, W)`
- `RockingCurveBatchResult.screen_images` has shape `(B, N, H, W)`

where `B` is batch size, `N` is the number of glancing-angle samples, and
`NB` is the number of simulated diffraction beams.

There are two batch entrypoints:

- `simulate_from_files_batch(...)` is the low-level fast path for many
  `surf.txt` files that share one `bulk.txt`
- `simulate_pairs_batch(...)` is the high-level scheduler that accepts a list
  of `(bulk.txt, surf.txt)` pairs, groups compatible jobs automatically, and
  preserves the original pair order in the returned results

The polycrystalline helper builds on that scheduler:

- `simulate_polycrystal_from_bulk(...)` samples many cubic grain orientations,
  generates the corresponding `bulk.txt` / nearly-empty `surf.txt` pairs, runs
  them through the high-level pair scheduler, and returns both the raw
  orientation batch stack and the integrated detector stack

The `structure` exports are intentionally viewer-friendly:

- `structure.xyz` opens in ASE and VESTA for quick inspection
- `structure.cif` includes the in-plane periodic cell plus a configurable slab
  vacuum padding, which is often more convenient in VESTA
- the structure plotting helpers can render the full slab, just the `bulk`
  atoms, or just the `surface` atoms

The `import-structure` workflow is intentionally explicit about the slab split:

- `surface_z_min` defines where the surface region begins in the imported
  coordinates
- `bulk_c_length` defines the repeated bulk-unit thickness along `z`
- `surface_thickness` controls the `dthick` value written into `surf.txt`
- plain `xyz` import needs explicit `--a-vector` and `--b-vector` values unless
  the file is in `extxyz` form with a `Lattice="..."` comment

## Validation cases in this repository

The package is validated against the SrTiO3 TiO2-terminated STO reference cases
in:

- `srtio3_001_tio2_beam_convergence_15keV/nb_001`
- `srtio3_001_tio2_beam_convergence_15keV/nb_003`
- `srtio3_001_tio2_beam_convergence_15keV/nb_005`

You can regenerate the validation overlays with:

```powershell
.\.venv\Scripts\python.exe -m torch_rheed.plot_validation
```
