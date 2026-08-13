from __future__ import annotations

from pathlib import Path

from .outputs import load_surface_output
from .plotting import plot_reference_comparison
from .simulation import simulate_from_files


def main() -> None:
    package_root = Path(__file__).resolve().parents[2]
    repo_root = package_root.parent
    output_dir = package_root / "validation_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "1-beam": (
            repo_root / "srtio3_001_tio2_beam_convergence_15keV" / "nb_001" / "surf-bulkE.s",
            package_root / "validation_outputs" / "nb_001_surf-bulkE.s",
            output_dir / "sto_validation_1beam.png",
        ),
        "3-beam": (
            repo_root / "srtio3_001_tio2_beam_convergence_15keV" / "nb_003" / "surf-bulkE.s",
            package_root / "validation_outputs" / "nb_003_surf-bulkE.s",
            output_dir / "sto_validation_3beam.png",
        ),
        "5-beam": (
            repo_root / "srtio3_001_tio2_beam_convergence_15keV" / "nb_005" / "surf-bulkE.s",
            package_root / "validation_outputs" / "nb_005_surf-bulkE.s",
            output_dir / "sto_validation_5beam.png",
        ),
    }

    for case_name, (ref_path, torch_path, out_path) in cases.items():
        result = simulate_from_files(ref_path.parent / "bulk.txt", ref_path.parent / "surf.txt")
        result.write_surface_output(torch_path)
        plot_reference_comparison(
            load_surface_output(ref_path),
            result,
            out_path,
            title=f"{case_name} STO validation",
        )
        print(out_path)


if __name__ == "__main__":
    main()
