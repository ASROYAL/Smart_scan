#!/usr/bin/env python3
"""Run a single scheduler against a single scenario and report metrics.

Usage:
    python scripts/run_experiment.py --scheduler adaptive --scenario periodic
    python scripts/run_experiment.py --scheduler round_robin --scenario sparse --steps 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from smartscan.core.config import load_config
from smartscan.evaluation.experiment import build_and_run
from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES
from smartscan.simulation.scenarios import ALL_SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single spectrum-scan experiment")
    parser.add_argument("--scheduler", default="adaptive",
                        choices=[s.value for s in ALL_SCHEDULER_TYPES])
    parser.add_argument("--scenario", default="periodic", choices=ALL_SCENARIOS)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override number of scan steps")
    parser.add_argument("--num-bands", type=int, default=None)
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--export-logs", action="store_true", help="Export scan logs to CSV/JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.steps is not None:
        config.simulation.num_steps = args.steps
    if args.num_bands is not None:
        config.environment.num_bands = args.num_bands
    if args.seed is not None:
        config.simulation.seed = args.seed

    print(f"Running scheduler='{args.scheduler}' scenario='{args.scenario}' "
          f"bands={config.environment.num_bands} seed={config.simulation.seed}")

    outcome = build_and_run(config, args.scheduler, args.scenario, seed=args.seed)
    r = outcome.result

    print("\n=== Results ===")
    print(f"  Steps run:              {r.num_steps}")
    print(f"  Simulated duration:     {r.duration:.3f} s")
    print(f"  Probability of Detect:  {r.probability_of_detection:.3f}")
    print(f"  Probability False Alarm:{r.probability_of_false_alarm:.3f}")
    print(f"  Scan hit rate:          {r.scan_hit_rate:.3f}")
    print(f"  Activity discovery:     {r.activity_discovery_ratio:.3f}")
    print(f"  Avg discovery delay:    {r.avg_discovery_delay*1000:.1f} ms")
    print(f"  Median discovery delay: {r.median_discovery_delay*1000:.1f} ms")
    print(f"  P95 discovery delay:    {r.p95_discovery_delay*1000:.1f} ms")
    print(f"  Scan efficiency:        {r.scan_efficiency:.3f}")
    print(f"  Band coverage:          {r.band_coverage:.3f}")
    print(f"  Starvation rate:        {r.starvation_rate:.3f}")
    print(f"  Average reward:         {r.avg_reward:.3f}")
    print(f"  Prediction accuracy:    {r.prediction_accuracy:.3f}")
    print(f"  Avg intercept-time err: {r.avg_intercept_time_error*1000:.1f} ms")
    print(f"  Wall-clock:             {r.wall_clock_seconds:.2f} s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always export the result summary (reproducibility)
    result_path = out_dir / f"result_{args.scheduler}_{args.scenario}_seed{r.seed}.json"
    with open(result_path, "w") as f:
        json.dump(r.model_dump(), f, indent=2)
    print(f"\nResult summary written to {result_path}")

    if args.export_logs:
        csv_path = out_dir / f"log_{args.scheduler}_{args.scenario}_seed{r.seed}.csv"
        json_path = out_dir / f"log_{args.scheduler}_{args.scenario}_seed{r.seed}.json"
        outcome.artifacts.logger.export_csv(csv_path)
        outcome.artifacts.logger.export_json(json_path)
        print(f"Scan logs written to {csv_path} and {json_path}")

    # Timing summary
    print("\n" + outcome.artifacts.timing.summary())


if __name__ == "__main__":
    main()
