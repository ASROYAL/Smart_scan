#!/usr/bin/env python3
"""Run a scheduler against radar ESM Pulse-Descriptor-Word data.

Uses either seeded synthetic radars or a PDW CSV (e.g. an export of the Turing
Synthetic Radar dataset, Scan Mode). The DSP → detector → scheduler → evaluator
pipeline is identical to the simulated-IQ path.

Examples:
    python scripts/pdw_experiment.py --scheduler adaptive --steps 1500
    python scripts/pdw_experiment.py --pdw-csv data/recordings/turing_scan.csv --scheduler adaptive
    python scripts/pdw_experiment.py --compare --steps 1500
"""

from __future__ import annotations

import argparse

from smartscan.core.config import load_config
from smartscan.evaluation.experiment import build_and_run_pdw
from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES
from smartscan.simulation.pdw import generate_synthetic_pdws, load_pdws_csv


def _print(r) -> None:
    print(f"  {r.scheduler_name:14} "
          f"hit={r.scan_hit_rate:.3f}  disc_ratio={r.activity_discovery_ratio:.3f}  "
          f"eff={r.scan_efficiency:.3f}  PD={r.probability_of_detection:.3f}  "
          f"PFA={r.probability_of_false_alarm:.3f}  "
          f"avg_intercept={r.avg_discovery_delay*1000:.0f}ms  "
          f"censored={r.censored_avg_intercept_time*1000:.0f}ms  "
          f"missed={r.missed_event_rate:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run scheduler on radar PDW data")
    ap.add_argument("--scheduler", default="adaptive",
                    choices=[s.value for s in ALL_SCHEDULER_TYPES])
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--pdw-csv", default=None, help="PDW CSV (Turing-style); omit for synthetic")
    ap.add_argument("--emitters", type=int, default=6, help="synthetic emitter count")
    ap.add_argument("--duration", type=float, default=20.0, help="synthetic PDW duration (s)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--num-bands", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", action="store_true", help="run all schedulers")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.simulation.num_steps = args.steps
    cfg.environment.num_bands = args.num_bands
    cfg.simulation.seed = args.seed

    if args.pdw_csv:
        pdws = load_pdws_csv(args.pdw_csv)
        print(f"Loaded {len(pdws)} PDWs from {args.pdw_csv}")
    else:
        pdws = generate_synthetic_pdws(
            num_emitters=args.emitters, duration=args.duration, seed=args.seed,
            total_bw=cfg.environment.total_bandwidth, center=cfg.environment.center_frequency)
        print(f"Generated {len(pdws)} synthetic radar PDWs "
              f"({args.emitters} emitters, {args.duration:.0f}s)")

    schedulers = [s.value for s in ALL_SCHEDULER_TYPES] if args.compare else [args.scheduler]
    print(f"\nRadar ESM intercept — {args.num_bands} bands, {args.steps} scans, seed {args.seed}")
    for sch in schedulers:
        _print(build_and_run_pdw(cfg, sch, pdws, seed=args.seed).result)


if __name__ == "__main__":
    main()
