#!/usr/bin/env python3
"""Full radar-ESM campaign: every scheduler across many PDW scenarios.

Runs each scheduler over a set of PDW pulse trains (seeded synthetic radars, or a
directory of real Turing-dataset Scan-Mode CSVs), averages the intercept metrics,
writes a summary CSV, and prints a ranking. This is the "run full campaigns" step.

Examples:
    python scripts/pdw_campaign.py --seeds 1 2 3 --emitters 6 --steps 1500
    python scripts/pdw_campaign.py --pdw-dir data/recordings/turing_scan/ --steps 2000
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from smartscan.core.config import load_config
from smartscan.evaluation.experiment import build_and_run_pdw
from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES
from smartscan.simulation.pdw import generate_synthetic_pdws, load_pdws_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Radar ESM intercept campaign")
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--pdw-dir", default=None,
                    help="directory of PDW CSVs (real Turing data); omit for synthetic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--emitters", type=int, default=6)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--num-bands", type=int, default=40)
    ap.add_argument("--out", default="data/results/pdw_campaign.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.simulation.num_steps = args.steps
    cfg.environment.num_bands = args.num_bands

    # Build the list of (name, pdws) pulse trains
    trains: list[tuple[str, list]] = []
    if args.pdw_dir:
        trains = [(p.stem, load_pdws_csv(p))
                  for p in sorted(Path(args.pdw_dir).glob("*.csv"))]
        if not trains:
            raise SystemExit(f"No .csv PDW files found in {args.pdw_dir}")
        print(f"Loaded {len(trains)} real PDW pulse trains from {args.pdw_dir}")
    else:
        trains = [(f"synuth_seed{s}", generate_synthetic_pdws(
                   num_emitters=args.emitters, duration=args.duration, seed=s,
                   total_bw=cfg.environment.total_bandwidth,
                   center=cfg.environment.center_frequency))
                  for s in args.seeds]
        print(f"Generated {len(trains)} synthetic pulse trains "
              f"({args.emitters} radars, {args.duration:.0f}s each)")

    rows = []
    for train_name, pdws in trains:
        for sch in [s.value for s in ALL_SCHEDULER_TYPES]:
            seed = int.from_bytes(hashlib.sha256(train_name.encode()).digest()[:4], "big") % 10000
            r = build_and_run_pdw(cfg, sch, pdws, seed=seed, scenario_name=train_name).result
            rows.append({
                "train": train_name, "scheduler": r.scheduler_name,
                "intercept_ratio": r.activity_discovery_ratio,
                "intercept_rate": r.scan_hit_rate,
                "avg_intercept_ms": r.avg_discovery_delay * 1000,
                "scan_efficiency": r.scan_efficiency,
                "PD": r.probability_of_detection, "PFA": r.probability_of_false_alarm,
                "prediction_accuracy": r.prediction_accuracy,
            })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (df.groupby("scheduler")
               .agg(intercept_ratio=("intercept_ratio", "mean"),
                    intercept_rate=("intercept_rate", "mean"),
                    avg_intercept_ms=("avg_intercept_ms", "mean"),
                    scan_efficiency=("scan_efficiency", "mean"))
               .sort_values("scan_efficiency", ascending=False))

    print("\n=== CAMPAIGN RANKING (averaged over all pulse trains) ===")
    print(summary.round(3).to_string())
    print(f"\nPer-run results written to {out}")


if __name__ == "__main__":
    main()
