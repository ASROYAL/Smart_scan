#!/usr/bin/env python3
"""Run a scheduler against a recorded complex64 IQ file.

The recording has no ground-truth labels, so truth-based metrics (PD / PFA /
intercept ratio / delay) are unavailable (0); detection-side metrics are real.

Example:
    python scripts/generate_dataset.py --scenario periodic --duration 2.0 --center 730e6
    python scripts/recorded_experiment.py --iq data/recordings/periodic.c64 --scheduler adaptive
"""

from __future__ import annotations

import argparse

from smartscan.core.config import load_config
from smartscan.evaluation.experiment import build_and_run_recorded
from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES


def main() -> None:
    ap = argparse.ArgumentParser(description="Run scheduler on a recorded IQ file")
    ap.add_argument("--iq", required=True, help="path to the complex64 .c64 IQ file")
    ap.add_argument("--meta", default=None, help="JSON sidecar (defaults to <iq>.json)")
    ap.add_argument("--scheduler", default="adaptive",
                    choices=[s.value for s in ALL_SCHEDULER_TYPES])
    ap.add_argument("--detector", default="energy",
                    choices=["energy", "matched_filter", "cyclostationary"])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--num-bands", type=int, default=50)
    ap.add_argument("--ml-predictor", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.simulation.num_steps = args.steps
    cfg.environment.num_bands = args.num_bands
    cfg.detector.detector_type = args.detector

    out = build_and_run_recorded(cfg, args.scheduler, args.iq, args.meta,
                                 use_ml_predictor=args.ml_predictor)
    r = out.result
    print(f"\nRecorded run — {args.iq}")
    print(f"  Scheduler:          {r.scheduler_name} ({args.detector} detector)")
    print(f"  Steps:              {r.num_steps}")
    print(f"  Scan hit rate:      {r.scan_hit_rate:.3f}")
    print(f"  Scan efficiency*:   {r.scan_efficiency:.3f}   (*needs truth; 0 for real data)")
    print(f"  Band coverage:      {r.band_coverage:.3f}")
    print(f"  Prediction acc.:    {r.prediction_accuracy:.3f}")
    print(f"  Avg reward:         {r.avg_reward:.3f}")
    print("  (PD / PFA / intercept ratio require ground truth — unavailable for recordings)")


if __name__ == "__main__":
    main()
