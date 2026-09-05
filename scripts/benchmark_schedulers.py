#!/usr/bin/env python3
"""Run all schedulers against all scenarios and produce comparison tables.

Usage:
    python scripts/benchmark_schedulers.py
    python scripts/benchmark_schedulers.py --scenarios sparse periodic --seeds 1 2 3
    python scripts/benchmark_schedulers.py --num-bands 100 --steps 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smartscan.core.config import load_config
from smartscan.evaluation.benchmark import run_benchmark, scheduler_ranking, summarize
from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES
from smartscan.simulation.scenarios import ALL_SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all schedulers across scenarios")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--scenarios", nargs="*", default=None, choices=ALL_SCENARIOS)
    parser.add_argument("--schedulers", nargs="*", default=None,
                        choices=[s.value for s in ALL_SCHEDULER_TYPES])
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--num-bands", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.num_bands is not None:
        config.environment.num_bands = args.num_bands
    if args.steps is not None:
        config.simulation.num_steps = args.steps

    scheduler_types = None
    if args.schedulers:
        from smartscan.core.models import SchedulerType
        scheduler_types = [SchedulerType(s) for s in args.schedulers]

    print("Running benchmark grid...")
    df = run_benchmark(
        config,
        scheduler_types=scheduler_types,
        scenarios=args.scenarios,
        seeds=args.seeds,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "benchmark_raw.csv"
    df.to_csv(raw_path, index=False)

    summary = summarize(df)
    summary_path = out_dir / "benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)

    ranking = scheduler_ranking(df)
    ranking_path = out_dir / "benchmark_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    import pandas as pd
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print("\n" + "=" * 80)
    print("PER-SCENARIO SUMMARY (averaged over seeds)")
    print("=" * 80)
    print(summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("OVERALL SCHEDULER RANKING (averaged over all scenarios)")
    print("=" * 80)
    print(ranking.to_string(index=False))

    print(f"\nRaw results:     {raw_path}")
    print(f"Summary table:   {summary_path}")
    print(f"Ranking table:   {ranking_path}")


if __name__ == "__main__":
    main()
