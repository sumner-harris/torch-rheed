from __future__ import annotations

import inspect
import unittest

from torch_rheed.cli import build_parser
from torch_rheed.polycrystal import simulate_polycrystal_from_bulk
from torch_rheed.simulation import (
    SURFACE_SOLVERS,
    simulate_from_files,
    simulate_from_files_batch,
    simulate_pairs_batch,
    simulate_rocking_curve,
    simulate_rocking_curve_batch,
)


class SolverDefaultTests(unittest.TestCase):
    def test_sp6_is_the_first_supported_solver(self) -> None:
        self.assertEqual(SURFACE_SOLVERS[0], "sp6")

    def test_public_python_entrypoints_default_to_sp6(self) -> None:
        entrypoints = (
            simulate_rocking_curve,
            simulate_rocking_curve_batch,
            simulate_from_files,
            simulate_from_files_batch,
            simulate_pairs_batch,
            simulate_polycrystal_from_bulk,
        )

        for entrypoint in entrypoints:
            with self.subTest(entrypoint=entrypoint.__name__):
                self.assertEqual(inspect.signature(entrypoint).parameters["solver"].default, "sp6")

    def test_simulate_cli_defaults_to_sp6(self) -> None:
        args = build_parser().parse_args(["simulate", "--bulk", "bulk.txt", "--surf", "surf.txt"])

        self.assertEqual(args.solver, "sp6")

    def test_polycrystal_cli_defaults_to_sp6(self) -> None:
        args = build_parser().parse_args(["polycrystal", "--bulk", "bulk.txt", "--out-dir", "out"])

        self.assertEqual(args.solver, "sp6")


if __name__ == "__main__":
    unittest.main()
