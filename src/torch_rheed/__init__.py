"""Standalone PyTorch tools for RHEED rocking-curve simulation."""

from .conversion import (
    ElementParameters,
    build_solver_inputs_from_structure,
    load_imported_structure,
    parse_beam_argument,
    parse_element_param_argument,
    write_solver_inputs,
)
from .inputs import load_bulk_input, load_surface_input
from .models import (
    BulkInput,
    BulkSimulation,
    PolycrystalOrientation,
    PolycrystalResult,
    RockingCurveBatchResult,
    RockingCurvePairBatchResult,
    RockingCurveResult,
    ScreenImageConfig,
    StructureAtom,
    StructureModel,
    SurfaceInput,
)
from .outputs import load_surface_output
from .plotting import plot_reference_comparison, plot_rocking_curve, plot_screen_frame, plot_surface_output_file
from .polycrystal import (
    build_blank_surface_input,
    build_oriented_cubic_bulk_input,
    generate_beam_shell_indices,
    generate_polycrystal_input_pairs,
    plot_polycrystal_screen_frame,
    plot_polycrystal_total_intensity,
    simulate_polycrystal_from_bulk,
)
from .screen import write_screen_gif
from .simulation import (
    simulate_bulk,
    simulate_from_files,
    simulate_from_files_batch,
    simulate_pairs_batch,
    simulate_rocking_curve,
    simulate_rocking_curve_batch,
)
from .structure import build_structure_from_files, build_structure_model, plot_structure_views
from .validation import ValidationResult, compare_to_reference

__all__ = [
    "BulkInput",
    "BulkSimulation",
    "ElementParameters",
    "PolycrystalOrientation",
    "PolycrystalResult",
    "RockingCurveBatchResult",
    "RockingCurvePairBatchResult",
    "RockingCurveResult",
    "ScreenImageConfig",
    "StructureAtom",
    "StructureModel",
    "SurfaceInput",
    "ValidationResult",
    "build_structure_from_files",
    "build_structure_model",
    "build_solver_inputs_from_structure",
    "build_blank_surface_input",
    "build_oriented_cubic_bulk_input",
    "compare_to_reference",
    "generate_beam_shell_indices",
    "generate_polycrystal_input_pairs",
    "load_imported_structure",
    "load_bulk_input",
    "load_surface_input",
    "load_surface_output",
    "parse_beam_argument",
    "parse_element_param_argument",
    "plot_reference_comparison",
    "plot_rocking_curve",
    "plot_polycrystal_screen_frame",
    "plot_polycrystal_total_intensity",
    "plot_screen_frame",
    "plot_structure_views",
    "plot_surface_output_file",
    "simulate_bulk",
    "simulate_from_files",
    "simulate_from_files_batch",
    "simulate_pairs_batch",
    "simulate_polycrystal_from_bulk",
    "simulate_rocking_curve",
    "simulate_rocking_curve_batch",
    "write_screen_gif",
    "write_solver_inputs",
]
