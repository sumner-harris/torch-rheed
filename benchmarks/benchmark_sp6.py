"""Benchmark SP6 against the legacy surface solver on one input pair."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from torch_rheed import load_bulk_input, load_surface_input, simulate_bulk, simulate_rocking_curve


def _run_surface(bulk, surface, *, solver: str, integration_step: float | None = None):
    started = perf_counter()
    result = simulate_rocking_curve(
        bulk,
        surface,
        screen_config=None,
        solver=solver,
        integration_step=integration_step,
    )
    return result, perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk", required=True, type=Path)
    parser.add_argument("--surf", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--integration-step", type=float, default=0.2)
    args = parser.parse_args()

    bulk_input = load_bulk_input(args.bulk)
    surface_input = load_surface_input(args.surf, bulk_input.ndom)
    started = perf_counter()
    bulk = simulate_bulk(bulk_input, device=args.device)
    bulk_seconds = perf_counter() - started

    legacy, legacy_seconds = _run_surface(bulk, surface_input, solver="multislice")
    sp6, sp6_seconds = _run_surface(
        bulk,
        surface_input,
        solver="sp6",
        integration_step=args.integration_step,
    )
    delta = (sp6.intensities - legacy.intensities).abs()
    peak = legacy.intensities.abs().amax()

    print(f"device={args.device}")
    print(f"angles={legacy.angles_deg.numel()} beams={len(legacy.beam_indices)}")
    print(f"bulk_seconds={bulk_seconds:.6f}")
    print(f"legacy_surface_seconds={legacy_seconds:.6f}")
    print(f"sp6_surface_seconds={sp6_seconds:.6f}")
    print(f"surface_speedup={legacy_seconds / sp6_seconds:.3f}x")
    print(f"max_abs_intensity_difference={float(delta.amax()):.12e}")
    print(f"max_difference_over_legacy_peak={float(delta.amax() / peak):.12e}")
    print(f"rms_intensity_difference={float(torch.sqrt(torch.mean(delta.square()))):.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
