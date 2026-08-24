"""Command-line interface for the standalone ``torch_rheed`` package."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import ScreenImageConfig
from .conversion import (
    ElementParameters,
    build_solver_inputs_from_structure,
    load_imported_structure,
    parse_beam_argument,
    parse_element_param_argument,
    write_solver_inputs,
)
from .outputs import load_surface_output
from .plotting import plot_reference_comparison, plot_rocking_curve, plot_screen_frame
from .polycrystal import (
    generate_beam_shell_indices,
    plot_polycrystal_screen_frame,
    plot_polycrystal_total_intensity,
    simulate_polycrystal_from_bulk,
)
from .screen import write_screen_gif
from .simulation import simulate_from_files
from .structure import build_structure_from_files, plot_structure_views
from .validation import compare_to_reference


def _resolve_outputs(
    out_dir: Path | None,
    surface_output: Path | None,
    csv_output: Path | None,
    plot_output: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    if out_dir is None:
        return surface_output, csv_output, plot_output

    return (
        surface_output or (out_dir / "surf-bulkE.s"),
        csv_output or (out_dir / "rocking_curve.csv"),
        plot_output or (out_dir / "rocking_curve.png"),
    )


def _resolve_screen_outputs(
    out_dir: Path | None,
    screen_stack_output: Path | None,
    screen_preview_output: Path | None,
    screen_gif_output: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    if out_dir is None:
        return screen_stack_output, screen_preview_output, screen_gif_output
    return (
        screen_stack_output or (out_dir / "screen_stack.pt"),
        screen_preview_output or (out_dir / "screen_preview.png"),
        screen_gif_output,
    )


def _resolve_structure_outputs(
    out_dir: Path | None,
    plot_output: Path | None,
    bulk_plot_output: Path | None,
    surface_plot_output: Path | None,
    xyz_output: Path | None,
    cif_output: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None]:
    if out_dir is None:
        return plot_output, bulk_plot_output, surface_plot_output, xyz_output, cif_output

    return (
        plot_output or (out_dir / "structure_views.png"),
        bulk_plot_output or (out_dir / "structure_bulk.png"),
        surface_plot_output or (out_dir / "structure_surface.png"),
        xyz_output or (out_dir / "structure.xyz"),
        cif_output or (out_dir / "structure.cif"),
    )


def _resolve_polycrystal_outputs(
    out_dir: Path | None,
    integrated_stack_output: Path | None,
    batch_stack_output: Path | None,
    preview_output: Path | None,
    total_intensity_csv_output: Path | None,
    total_intensity_plot_output: Path | None,
    mean_beam_csv_output: Path | None,
    summary_output: Path | None,
) -> tuple[Path | None, Path | None, Path | None, Path | None, Path | None, Path | None, Path | None]:
    if out_dir is None:
        return (
            integrated_stack_output,
            batch_stack_output,
            preview_output,
            total_intensity_csv_output,
            total_intensity_plot_output,
            mean_beam_csv_output,
            summary_output,
        )
    return (
        integrated_stack_output or (out_dir / "screen_stack_polycrystalline.pt"),
        batch_stack_output or (out_dir / "screen_stack_batch.pt"),
        preview_output or (out_dir / "screen_preview_polycrystalline.png"),
        total_intensity_csv_output or (out_dir / "total_detector_intensity_vs_angle.csv"),
        total_intensity_plot_output or (out_dir / "total_detector_intensity_vs_angle.png"),
        mean_beam_csv_output or (out_dir / "mean_beam_intensity.csv"),
        summary_output or (out_dir / "summary.txt"),
    )


def _write_requested_outputs(
    *,
    result,
    surface_output: Path | None,
    csv_output: Path | None,
    plot_output: Path | None,
    screen_stack_output: Path | None,
    screen_preview_output: Path | None,
    screen_gif_output: Path | None,
    screen_gif_frame_step: int,
    screen_gif_frame_duration_ms: int,
    screen_gif_normalize: str,
    title: str | None,
) -> None:
    if surface_output is not None:
        result.write_surface_output(surface_output)
    if csv_output is not None:
        result.write_csv(csv_output)
    if plot_output is not None:
        plot_rocking_curve(result, plot_output, title=title)
    if screen_stack_output is not None:
        result.write_screen_stack(screen_stack_output)
    if screen_preview_output is not None:
        plot_screen_frame(result, screen_preview_output, title=title or "Simulated RHEED CTR screen")
    if screen_gif_output is not None:
        write_screen_gif(
            result,
            screen_gif_output,
            frame_step=screen_gif_frame_step,
            frame_duration_ms=screen_gif_frame_duration_ms,
            normalize=screen_gif_normalize,
            boomerang=True,
            title=title or "Simulated RHEED CTR screen",
        )


def _print_result_summary(result) -> None:
    beams = ", ".join(f"({ih},{ik})" for ih, ik in result.beam_indices)
    print(f"angles={result.angles_deg.numel()} beams={len(result.beam_indices)}")
    print(f"beam_indices={beams}")


def _add_common_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bulk", required=True, type=Path, help="Path to the bulk.txt input file.")
    parser.add_argument("--surf", required=True, type=Path, help="Path to the surf.txt input file.")
    parser.add_argument(
        "--solver",
        choices=("multislice", "sp6"),
        default="sp6",
        help="Surface propagator. Defaults to the sixth-order SP6 SRKN^b_11 splitting method with RHST.",
    )
    parser.add_argument(
        "--integration-step",
        type=float,
        help="Requested SP6 surface step in Angstrom. Defaults to 10 times the bulk.txt DZ.",
    )
    parser.add_argument(
        "--rhst-threshold",
        type=float,
        default=1000.0,
        help="Gershgorin condition estimate that triggers SP6 right-hand-side stabilization.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch execution device, for example 'cpu', 'cuda', or 'cuda:0'. Defaults to 'cpu'.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional output directory. When provided, default output filenames are created inside it.",
    )
    parser.add_argument("--surface-output", type=Path, help="Write the conventional surf-bulkE.s text output.")
    parser.add_argument("--csv-output", type=Path, help="Write the rocking-curve data as CSV.")
    parser.add_argument("--plot-output", type=Path, help="Write a rocking-curve plot image.")
    parser.add_argument("--screen-stack-output", type=Path, help="Write the simulated detector stack as a torch tensor file.")
    parser.add_argument("--screen-preview-output", type=Path, help="Write a detector-frame preview image near the reference angle.")
    parser.add_argument("--screen-gif-output", type=Path, help="Write the detector stack as an animated GIF.")
    parser.add_argument(
        "--plane-mode",
        default="vertical",
        choices=("vertical", "specular-normal"),
        help="Detector plane geometry used for synthetic screen rendering.",
    )
    parser.add_argument("--screen-distance-mm", type=float, default=300.0, help="Sample-to-screen distance in millimeters.")
    parser.add_argument("--screen-width-mm", type=float, default=120.0, help="Detector width in millimeters.")
    parser.add_argument("--screen-height-mm", type=float, default=90.0, help="Detector height in millimeters.")
    parser.add_argument("--screen-pixels-x", type=int, default=320, help="Detector raster width in pixels.")
    parser.add_argument("--screen-pixels-y", type=int, default=240, help="Detector raster height in pixels.")
    parser.add_argument(
        "--correlation-length-angstrom",
        type=float,
        default=1000.0,
        help="In-plane coherence/correlation length that sets the CTR rod width on the detector.",
    )
    parser.add_argument(
        "--rod-profile",
        default="lorentzian",
        choices=("lorentzian", "lorentzian2"),
        help="Reciprocal-space rod cross-section used by the CTR screen renderer.",
    )
    parser.add_argument(
        "--beam-intensity-floor",
        type=float,
        default=0.0,
        help="Clip beam intensities below this value before building the CTR I(qz) profiles.",
    )
    parser.add_argument(
        "--instrument-broadening-fwhm-mm",
        type=float,
        default=0.0,
        help="Gaussian instrument-broadening FWHM on the detector in millimeters.",
    )
    parser.add_argument(
        "--source-glancing-divergence-fwhm-deg",
        type=float,
        default=0.0,
        help=(
            "Gaussian full width at half maximum, in degrees, for the incident-beam "
            "glancing-angle divergence integrated into each CTR detector frame."
        ),
    )
    parser.add_argument(
        "--source-divergence-samples",
        type=int,
        default=9,
        help="Number of quadrature samples used for the glancing-angle source-divergence integration.",
    )
    parser.add_argument("--screen-reference-angle-deg", type=float, help="Optional detector-centering angle in degrees. Defaults to the 00 peak.")
    parser.add_argument(
        "--screen-frame-azimuth-deg",
        type=float,
        help="Optional fixed lab-frame azimuth, in degrees, used to define the detector plane while the diffraction beams follow each run's actual azimuth.",
    )
    parser.add_argument(
        "--screen-display-scale",
        default="sqrt",
        choices=("linear", "sqrt", "log"),
        help="Display scaling used for preview images.",
    )
    parser.add_argument("--screen-colormap", default="inferno", help="Matplotlib colormap used for preview images.")
    parser.add_argument(
        "--screen-gif-frame-step",
        type=int,
        default=1,
        help="Use every Nth detector frame in the GIF output.",
    )
    parser.add_argument(
        "--screen-gif-frame-duration-ms",
        type=int,
        default=90,
        help="Frame duration for --screen-gif-output.",
    )
    parser.add_argument(
        "--screen-gif-normalize",
        default="global",
        choices=("global", "per-frame"),
        help="Use one shared intensity scale across the GIF or renormalize each frame independently.",
    )
    parser.add_argument("--title", help="Optional plot title.")


def _screen_config_from_args(args: argparse.Namespace) -> ScreenImageConfig:
    """Build the detector-image configuration from common CLI arguments."""

    return ScreenImageConfig(
        plane_mode=args.plane_mode,
        screen_distance_mm=args.screen_distance_mm,
        screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm,
        pixels_x=args.screen_pixels_x,
        pixels_y=args.screen_pixels_y,
        correlation_length_angstrom=args.correlation_length_angstrom,
        rod_profile=args.rod_profile,
        beam_intensity_floor=args.beam_intensity_floor,
        instrument_broadening_fwhm_mm=args.instrument_broadening_fwhm_mm,
        source_glancing_divergence_fwhm_deg=args.source_glancing_divergence_fwhm_deg,
        source_divergence_samples=args.source_divergence_samples,
        reference_angle_deg=args.screen_reference_angle_deg,
        frame_azimuth_deg=args.screen_frame_azimuth_deg,
        display_scale=args.screen_display_scale,
        colormap=args.screen_colormap,
    )


def _command_simulate(args: argparse.Namespace) -> int:
    result = simulate_from_files(
        args.bulk,
        args.surf,
        device=args.device,
        screen_config=_screen_config_from_args(args),
        solver=args.solver,
        integration_step=args.integration_step,
        rhst_threshold=args.rhst_threshold,
    )
    surface_output, csv_output, plot_output = _resolve_outputs(
        args.out_dir,
        args.surface_output,
        args.csv_output,
        args.plot_output,
    )
    screen_stack_output, screen_preview_output, screen_gif_output = _resolve_screen_outputs(
        args.out_dir,
        args.screen_stack_output,
        args.screen_preview_output,
        args.screen_gif_output,
    )
    _write_requested_outputs(
        result=result,
        surface_output=surface_output,
        csv_output=csv_output,
        plot_output=plot_output,
        screen_stack_output=screen_stack_output,
        screen_preview_output=screen_preview_output,
        screen_gif_output=screen_gif_output,
        screen_gif_frame_step=args.screen_gif_frame_step,
        screen_gif_frame_duration_ms=args.screen_gif_frame_duration_ms,
        screen_gif_normalize=args.screen_gif_normalize,
        title=args.title,
    )

    if (
        surface_output is None
        and csv_output is None
        and plot_output is None
        and screen_stack_output is None
        and screen_preview_output is None
        and screen_gif_output is None
    ):
        print(result.to_surface_output_text(), end="")
    else:
        _print_result_summary(result)
    return 0


def _command_plot(args: argparse.Namespace) -> int:
    result = load_surface_output(args.input)
    plot_rocking_curve(result, args.output, title=args.title)
    print(args.output)
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    result = simulate_from_files(
        args.bulk,
        args.surf,
        device=args.device,
        screen_config=_screen_config_from_args(args),
        solver=args.solver,
        integration_step=args.integration_step,
        rhst_threshold=args.rhst_threshold,
    )
    surface_output, csv_output, plot_output = _resolve_outputs(
        args.out_dir,
        args.surface_output,
        args.csv_output,
        args.plot_output,
    )
    screen_stack_output, screen_preview_output, screen_gif_output = _resolve_screen_outputs(
        args.out_dir,
        args.screen_stack_output,
        args.screen_preview_output,
        args.screen_gif_output,
    )
    _write_requested_outputs(
        result=result,
        surface_output=surface_output,
        csv_output=csv_output,
        plot_output=plot_output,
        screen_stack_output=screen_stack_output,
        screen_preview_output=screen_preview_output,
        screen_gif_output=screen_gif_output,
        screen_gif_frame_step=args.screen_gif_frame_step,
        screen_gif_frame_duration_ms=args.screen_gif_frame_duration_ms,
        screen_gif_normalize=args.screen_gif_normalize,
        title=args.title,
    )

    comparison = compare_to_reference(result, args.reference)
    print(f"textual_match={comparison.textual_match}")
    print(f"max_abs_diff={comparison.max_abs_diff:.6e}")

    if args.comparison_plot is not None:
        plot_reference_comparison(
            load_surface_output(args.reference),
            result,
            args.comparison_plot,
            title=args.title or "Reference comparison",
        )
        print(args.comparison_plot)
    return 0


def _command_structure(args: argparse.Namespace) -> int:
    structure = build_structure_from_files(
        args.bulk,
        args.surf,
        a_units=args.a_units,
        b_units=args.b_units,
        bulk_layers=args.bulk_layers,
    )
    plot_output, bulk_plot_output, surface_plot_output, xyz_output, cif_output = _resolve_structure_outputs(
        args.out_dir,
        args.plot_output,
        args.bulk_plot_output,
        args.surface_plot_output,
        args.xyz_output,
        args.cif_output,
    )
    if plot_output is None and bulk_plot_output is None and surface_plot_output is None and xyz_output is None and cif_output is None:
        raise ValueError(
            "structure output requires at least one of --out-dir, --plot-output, --bulk-plot-output, --surface-plot-output, --xyz-output, or --cif-output"
        )

    if plot_output is not None:
        plot_structure_views(structure, plot_output, title=args.title)
        print(plot_output)
    if bulk_plot_output is not None:
        bulk_title = args.bulk_title or (f"{args.title} (Bulk)" if args.title else None)
        plot_structure_views(structure, bulk_plot_output, title=bulk_title, regions="bulk")
        print(bulk_plot_output)
    if surface_plot_output is not None:
        surface_title = args.surface_title or (f"{args.title} (Surface)" if args.title else None)
        plot_structure_views(structure, surface_plot_output, title=surface_title, regions="surface")
        print(surface_plot_output)
    if xyz_output is not None:
        structure.write_xyz(xyz_output)
        print(xyz_output)
    if cif_output is not None:
        structure.write_cif(cif_output, vacuum_padding_angstrom=args.vacuum_padding)
        print(cif_output)
    return 0


def _command_import_structure(args: argparse.Namespace) -> int:
    element_overrides = dict(parse_element_param_argument(text) for text in args.element_param)
    structure = load_imported_structure(
        args.input,
        a_vector_override=_parse_optional_vector(args.a_vector),
        b_vector_override=_parse_optional_vector(args.b_vector),
        c_vector_override=_parse_optional_vector(args.c_vector),
    )
    bulk_input, surface_input = build_solver_inputs_from_structure(
        structure,
        surface_z_min=args.surface_z_min,
        bulk_c_length=args.bulk_c_length,
        surface_thickness=args.surface_thickness,
        beam_indices=[parse_beam_argument(text) for text in args.beam],
        be=args.be,
        azi_deg=args.azi_deg,
        azf_deg=args.azf_deg,
        daz_deg=args.daz_deg,
        gi_deg=args.gi_deg,
        gf_deg=args.gf_deg,
        dg_deg=args.dg_deg,
        dz_input=args.dz_input,
        ml=args.ml,
        nh=args.nh,
        nk=args.nk,
        bulk_dx=args.bulk_dx,
        bulk_dy=args.bulk_dy,
        surface_dxs=args.surface_dxs,
        surface_dys=args.surface_dys,
        nsg=args.nsg,
        nsgs=args.nsgs,
        msa=args.msa,
        msb=args.msb,
        nsa=args.nsa,
        nsb=args.nsb,
        default_element_parameters=ElementParameters(
            da1=args.default_da1,
            sap=args.default_sap,
            bh=args.default_bh,
            bk=args.default_bk,
            bz=args.default_bz,
        ),
        element_parameter_overrides=element_overrides,
    )
    bulk_output, surface_output = _resolve_import_outputs(args.out_dir, args.bulk_output, args.surface_output)
    if bulk_output is None or surface_output is None:
        raise ValueError("import-structure requires either --out-dir or both --bulk-output and --surface-output")
    write_solver_inputs(bulk_input, surface_input, bulk_path=bulk_output, surface_path=surface_output)
    print(bulk_output)
    print(surface_output)
    return 0


def _command_polycrystal(args: argparse.Namespace) -> int:
    beam_indices = [parse_beam_argument(text) for text in args.beam] if args.beam else generate_beam_shell_indices(args.beam_shell_radius)
    result = simulate_polycrystal_from_bulk(
        args.bulk,
        out_dir=args.out_dir or Path.cwd() / "torch_rheed_polycrystal",
        max_miller_index=args.max_miller_index,
        azimuth_start_deg=args.azimuth_start_deg,
        azimuth_stop_deg=args.azimuth_stop_deg,
        azimuth_step_deg=args.azimuth_step_deg,
        beam_indices=beam_indices,
        beam_shell_radius=args.beam_shell_radius,
        device=args.device,
        screen_config=_screen_config_from_args(args),
        solver=args.solver,
        integration_step=args.integration_step,
        rhst_threshold=args.rhst_threshold,
    )

    (
        integrated_stack_output,
        batch_stack_output,
        preview_output,
        total_intensity_csv_output,
        total_intensity_plot_output,
        mean_beam_csv_output,
        summary_output,
    ) = _resolve_polycrystal_outputs(
        args.out_dir,
        args.integrated_stack_output,
        args.batch_stack_output,
        args.preview_output,
        args.total_intensity_csv_output,
        args.total_intensity_plot_output,
        args.mean_beam_csv_output,
        args.summary_output,
    )

    if integrated_stack_output is not None:
        result.write_integrated_screen_stack(integrated_stack_output)
    if batch_stack_output is not None:
        result.write_orientation_screen_stack(batch_stack_output)
    if preview_output is not None:
        plot_polycrystal_screen_frame(result, preview_output, title=args.title or "Polycrystalline RHEED CTR screen")
    if total_intensity_csv_output is not None:
        result.write_total_detector_csv(total_intensity_csv_output)
    if total_intensity_plot_output is not None:
        plot_polycrystal_total_intensity(result, total_intensity_plot_output, title=args.total_intensity_title)
    if mean_beam_csv_output is not None:
        result.write_mean_beam_csv(mean_beam_csv_output)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        normals = sorted({orientation.normal_hkl for orientation in result.orientations})
        summary_output.write_text(
            "\n".join(
                [
                    "Polycrystalline torch_rheed approximation",
                    "",
                    f"Base bulk: {args.bulk}",
                    f"Unique cubic normals: {len(normals)}",
                    f"Successful orientation count: {len(result.orientations)}",
                    f"Failed orientation count: {0 if result.failed_orientations is None else len(result.failed_orientations)}",
                    f"Azimuth range: {args.azimuth_start_deg} deg to {args.azimuth_stop_deg} deg",
                    f"Azimuth step: {args.azimuth_step_deg} deg",
                    f"Beam count per orientation: {len(result.orientation_batch.beam_indices)}",
                    f"Integrated screen shape: {tuple(result.integrated_screen_images.shape)}",
                    f"Orientation batch shape: {tuple(result.orientation_batch.screen_images.shape) if result.orientation_batch.screen_images is not None else 'none'}",
                    f"Detector reference angle (deg): {result.screen_config.reference_angle_deg}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    print(
        f"orientations={len(result.orientations)} "
        f"failed={0 if result.failed_orientations is None else len(result.failed_orientations)} "
        f"normals={len(set(orientation.normal_hkl for orientation in result.orientations))}"
    )
    print(f"angles={result.angles_deg.numel()} beams={len(result.orientation_batch.beam_indices)}")
    return 0


def _parse_optional_vector(text: str | None) -> tuple[float, float, float] | None:
    if text is None:
        return None
    values = [float(token) for token in text.split(",")]
    if len(values) != 3:
        raise ValueError(f"expected three comma-separated vector components, got: {text}")
    return values[0], values[1], values[2]


def _resolve_import_outputs(
    out_dir: Path | None,
    bulk_output: Path | None,
    surface_output: Path | None,
) -> tuple[Path | None, Path | None]:
    if out_dir is None:
        return bulk_output, surface_output
    return bulk_output or (out_dir / "bulk.txt"), surface_output or (out_dir / "surf.txt")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="torch-rheed",
        description="Standalone PyTorch RHEED rocking-curve simulator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run a standalone rocking-curve simulation from bulk.txt and surf.txt.",
    )
    _add_common_simulation_arguments(simulate_parser)
    simulate_parser.set_defaults(func=_command_simulate)

    plot_parser = subparsers.add_parser(
        "plot",
        help="Plot a saved surf-bulkE.s or surf-bulkP.s text output.",
    )
    plot_parser.add_argument("--input", required=True, type=Path, help="Input surface-output text file.")
    plot_parser.add_argument("--output", required=True, type=Path, help="Output plot path.")
    plot_parser.add_argument("--title", help="Optional plot title.")
    plot_parser.set_defaults(func=_command_plot)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run a simulation and compare it against a reference surface-output text file.",
    )
    _add_common_simulation_arguments(validate_parser)
    validate_parser.add_argument("--reference", required=True, type=Path, help="Reference surface-output text file.")
    validate_parser.add_argument(
        "--comparison-plot",
        type=Path,
        help="Optional overlay plot comparing the generated and reference curves.",
    )
    validate_parser.set_defaults(func=_command_validate)

    structure_parser = subparsers.add_parser(
        "structure",
        help="Build a bulk+surface structure view and export viewer-friendly structure files.",
    )
    structure_parser.add_argument("--bulk", required=True, type=Path, help="Path to the bulk.txt input file.")
    structure_parser.add_argument("--surf", required=True, type=Path, help="Path to the surf.txt input file.")
    structure_parser.add_argument("--a-units", type=int, default=1, help="Number of surface-cell repeats along a_s.")
    structure_parser.add_argument("--b-units", type=int, default=1, help="Number of surface-cell repeats along b_s.")
    structure_parser.add_argument("--bulk-layers", type=int, default=3, help="Number of bulk unit repetitions below the surface.")
    structure_parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional output directory. When provided, default structure plot, XYZ, and CIF files are created inside it.",
    )
    structure_parser.add_argument("--plot-output", type=Path, help="Output path for the multi-view structure plot.")
    structure_parser.add_argument("--bulk-plot-output", type=Path, help="Output path for a bulk-only structure plot.")
    structure_parser.add_argument("--surface-plot-output", type=Path, help="Output path for a surface-only structure plot.")
    structure_parser.add_argument("--xyz-output", type=Path, help="Output path for the XYZ structure export.")
    structure_parser.add_argument("--cif-output", type=Path, help="Output path for the CIF structure export.")
    structure_parser.add_argument(
        "--vacuum-padding",
        type=float,
        default=6.0,
        help="Vacuum padding, in Angstrom, added above and below the slab in the CIF export.",
    )
    structure_parser.add_argument("--title", help="Optional plot title.")
    structure_parser.add_argument("--bulk-title", help="Optional title override for the bulk-only plot.")
    structure_parser.add_argument("--surface-title", help="Optional title override for the surface-only plot.")
    structure_parser.set_defaults(func=_command_structure)

    import_parser = subparsers.add_parser(
        "import-structure",
        help="Convert a slab CIF or XYZ file into bulk.txt and surf.txt inputs.",
    )
    import_parser.add_argument("--input", required=True, type=Path, help="Input slab structure file (.cif, .xyz, or .extxyz).")
    import_parser.add_argument("--out-dir", type=Path, help="Optional directory for generated bulk.txt and surf.txt outputs.")
    import_parser.add_argument("--bulk-output", type=Path, help="Optional explicit output path for bulk.txt.")
    import_parser.add_argument("--surface-output", type=Path, help="Optional explicit output path for surf.txt.")
    import_parser.add_argument("--surface-z-min", required=True, type=float, help="Absolute z coordinate where the surface region begins.")
    import_parser.add_argument("--bulk-c-length", required=True, type=float, help="Thickness of one repeated bulk unit along z, in Angstrom.")
    import_parser.add_argument(
        "--surface-thickness",
        type=float,
        help="Optional dthick value to write into surf.txt. If omitted, torch_rheed estimates it from the imported slab.",
    )
    import_parser.add_argument("--beam", action="append", required=True, help="Beam index pair written as IH,IK. Repeat the option for multiple beams.")
    import_parser.add_argument("--be", type=float, default=15.0, help="Beam energy in keV.")
    import_parser.add_argument("--azi-deg", type=float, default=0.0, help="Initial azimuth angle in degrees.")
    import_parser.add_argument("--azf-deg", type=float, default=0.0, help="Final azimuth angle in degrees.")
    import_parser.add_argument("--daz-deg", type=float, default=0.0, help="Azimuth-angle step in degrees.")
    import_parser.add_argument("--gi-deg", type=float, default=0.1, help="Initial glancing angle in degrees.")
    import_parser.add_argument("--gf-deg", type=float, default=7.0, help="Final glancing angle in degrees.")
    import_parser.add_argument("--dg-deg", type=float, default=0.1, help="Glancing-angle step in degrees.")
    import_parser.add_argument("--dz-input", type=float, default=0.05, help="Requested slice spacing DZ for the generated bulk.txt file.")
    import_parser.add_argument("--ml", type=int, default=200, help="Bulk repeat count written into the generated bulk.txt file.")
    import_parser.add_argument("--nh", type=int, default=1, help="NH beam-index denominator for bulk.txt.")
    import_parser.add_argument("--nk", type=int, default=1, help="NK beam-index denominator for bulk.txt.")
    import_parser.add_argument("--nsg", type=int, default=1, help="Bulk plane-group code. Current standalone support is p1 only.")
    import_parser.add_argument("--nsgs", type=int, default=1, help="Surface plane-group code. Current standalone support is p1 only.")
    import_parser.add_argument("--msa", type=int, default=1, help="Surface cell transform coefficient msa.")
    import_parser.add_argument("--msb", type=int, default=0, help="Surface cell transform coefficient msb.")
    import_parser.add_argument("--nsa", type=int, default=0, help="Surface cell transform coefficient nsa.")
    import_parser.add_argument("--nsb", type=int, default=1, help="Surface cell transform coefficient nsb.")
    import_parser.add_argument("--bulk-dx", type=float, default=0.0, help="Bulk inter-repeat shift DX.")
    import_parser.add_argument("--bulk-dy", type=float, default=0.0, help="Bulk inter-repeat shift DY.")
    import_parser.add_argument("--surface-dxs", type=float, default=0.0, help="Surface offset DXS.")
    import_parser.add_argument("--surface-dys", type=float, default=0.0, help="Surface offset DYS.")
    import_parser.add_argument("--default-da1", type=float, default=1.0, help="Default da1 value for elements without explicit overrides.")
    import_parser.add_argument("--default-sap", type=float, default=0.1, help="Default sap value for elements without explicit overrides.")
    import_parser.add_argument("--default-bh", type=float, default=0.5, help="Default BH value for elements without explicit overrides.")
    import_parser.add_argument("--default-bk", type=float, default=0.5, help="Default BK value for elements without explicit overrides.")
    import_parser.add_argument("--default-bz", type=float, default=0.5, help="Default BZ value for elements without explicit overrides.")
    import_parser.add_argument(
        "--element-param",
        action="append",
        default=[],
        help="Per-element override written as Symbol:da1,sap,bh,bk,bz. Repeat the option as needed.",
    )
    import_parser.add_argument("--a-vector", help="Override for the XYZ in-plane a vector as ax,ay,az.")
    import_parser.add_argument("--b-vector", help="Override for the XYZ in-plane b vector as bx,by,bz.")
    import_parser.add_argument("--c-vector", help="Optional override for the XYZ c vector as cx,cy,cz.")
    import_parser.set_defaults(func=_command_import_structure)

    poly_parser = subparsers.add_parser(
        "polycrystal",
        help="Approximate a polycrystalline RHEED screen by averaging many oriented cubic bulk grains.",
    )
    poly_parser.add_argument("--bulk", required=True, type=Path, help="Base cubic bulk.txt file used as the crystallographic template.")
    poly_parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for generated orientation inputs and integrated detector outputs.",
    )
    poly_parser.add_argument("--device", default="cpu", help="Torch execution device, for example 'cpu' or 'cuda:0'.")
    poly_parser.add_argument("--solver", choices=("multislice", "sp6"), default="sp6", help="Surface propagator used for each grain orientation (default: sp6).")
    poly_parser.add_argument("--integration-step", type=float, help="Requested SP6 surface step in Angstrom.")
    poly_parser.add_argument("--rhst-threshold", type=float, default=1000.0, help="SP6 RHST condition threshold.")
    poly_parser.add_argument("--max-miller-index", type=int, default=2, help="Maximum Miller index used when enumerating unique cubic surface normals.")
    poly_parser.add_argument("--azimuth-start-deg", type=float, default=0.0, help="Starting in-plane azimuth angle for each normal.")
    poly_parser.add_argument("--azimuth-stop-deg", type=float, default=180.0, help="Final in-plane azimuth angle for each normal.")
    poly_parser.add_argument("--azimuth-step-deg", type=float, default=15.0, help="Azimuth increment used for each sampled normal.")
    poly_parser.add_argument(
        "--beam",
        action="append",
        help="Optional explicit beam index pair written as IH,IK. Repeat the option for multiple beams.",
    )
    poly_parser.add_argument(
        "--beam-shell-radius",
        type=int,
        default=4,
        help="If --beam is omitted, use every integer-order beam with h^2 + k^2 <= radius^2.",
    )
    poly_parser.add_argument("--integrated-stack-output", type=Path, help="Output path for the integrated detector stack tensor.")
    poly_parser.add_argument("--batch-stack-output", type=Path, help="Output path for the raw per-orientation detector stack tensor.")
    poly_parser.add_argument("--preview-output", type=Path, help="Output path for an integrated detector preview image.")
    poly_parser.add_argument("--total-intensity-csv-output", type=Path, help="Output path for total detector intensity versus angle as CSV.")
    poly_parser.add_argument("--total-intensity-plot-output", type=Path, help="Output path for a total detector intensity plot.")
    poly_parser.add_argument("--mean-beam-csv-output", type=Path, help="Output path for the orientation-averaged beam intensities as CSV.")
    poly_parser.add_argument("--summary-output", type=Path, help="Output path for a text summary of the sampled orientations.")
    poly_parser.add_argument("--title", help="Optional detector preview title.")
    poly_parser.add_argument("--total-intensity-title", help="Optional title for the detector-intensity plot.")
    poly_parser.add_argument(
        "--plane-mode",
        default="vertical",
        choices=("vertical", "specular-normal"),
        help="Detector plane geometry used for synthetic screen rendering.",
    )
    poly_parser.add_argument("--screen-distance-mm", type=float, default=300.0, help="Sample-to-screen distance in millimeters.")
    poly_parser.add_argument("--screen-width-mm", type=float, default=120.0, help="Detector width in millimeters.")
    poly_parser.add_argument("--screen-height-mm", type=float, default=90.0, help="Detector height in millimeters.")
    poly_parser.add_argument("--screen-pixels-x", type=int, default=320, help="Detector raster width in pixels.")
    poly_parser.add_argument("--screen-pixels-y", type=int, default=240, help="Detector raster height in pixels.")
    poly_parser.add_argument(
        "--correlation-length-angstrom",
        type=float,
        default=1000.0,
        help="In-plane coherence/correlation length that sets the CTR rod width on the detector.",
    )
    poly_parser.add_argument(
        "--rod-profile",
        default="lorentzian",
        choices=("lorentzian", "lorentzian2"),
        help="Reciprocal-space rod cross-section used by the CTR screen renderer.",
    )
    poly_parser.add_argument(
        "--beam-intensity-floor",
        type=float,
        default=0.0,
        help="Clip beam intensities below this value before building the CTR I(qz) profiles.",
    )
    poly_parser.add_argument(
        "--instrument-broadening-fwhm-mm",
        type=float,
        default=0.0,
        help="Gaussian instrument-broadening FWHM on the detector in millimeters.",
    )
    poly_parser.set_defaults(
        source_glancing_divergence_fwhm_deg=0.0,
        source_divergence_samples=9,
    )
    poly_parser.add_argument("--screen-reference-angle-deg", type=float, help="Optional detector-centering angle in degrees. Defaults to the 00 peak.")
    poly_parser.add_argument(
        "--screen-frame-azimuth-deg",
        type=float,
        default=0.0,
        help="Fixed laboratory azimuth used to define the detector plane while grain orientations rotate beneath it.",
    )
    poly_parser.add_argument(
        "--screen-display-scale",
        default="sqrt",
        choices=("linear", "sqrt", "log"),
        help="Display scaling used for preview images.",
    )
    poly_parser.add_argument("--screen-colormap", default="inferno", help="Matplotlib colormap used for preview images.")
    poly_parser.set_defaults(func=_command_polycrystal)

    return parser


def main() -> int:
    """Run the ``torch_rheed`` command-line interface."""

    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
