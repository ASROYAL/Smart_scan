"""Benchmark harness — run every scheduler against every scenario fairly.

All schedulers face identical environments (same seed, emitters, noise) so the
comparison is fair. Results are collected into a pandas DataFrame for tables and
charts.
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from smartscan.core.config import SmartScanConfig
from smartscan.core.models import ExperimentResult
from smartscan.evaluation.experiment import build_and_run
from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES
from smartscan.simulation.scenarios import ALL_SCENARIOS


def run_benchmark(
    config: SmartScanConfig,
    scheduler_types: list | None = None,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Run the full benchmark grid.

    Args:
        config: Base configuration.
        scheduler_types: Schedulers to test (defaults to all).
        scenarios: Scenarios to test (defaults to all).
        seeds: Seeds to average over (defaults to [config.simulation.seed]).
        progress: Print progress lines.

    Returns:
        DataFrame with one row per (scheduler, scenario, seed).
    """
    scheduler_types = scheduler_types or ALL_SCHEDULER_TYPES
    scenarios = scenarios or ALL_SCENARIOS
    seeds = seeds or [config.simulation.seed]

    rows: list[dict] = []
    total = len(scheduler_types) * len(scenarios) * len(seeds)
    done = 0

    for scenario_name in scenarios:
        for sched_type in scheduler_types:
            for seed in seeds:
                outcome = build_and_run(config, sched_type, scenario_name, seed=seed)
                rows.append(_result_to_row(outcome.result))
                done += 1
                if progress:
                    r = outcome.result
                    print(
                        f"[{done}/{total}] {scenario_name:18s} {r.scheduler_name:16s} "
                        f"seed={seed} PD={r.probability_of_detection:.2f} "
                        f"discovery_ratio={r.activity_discovery_ratio:.2f} "
                        f"avg_delay={r.avg_discovery_delay*1000:.1f}ms"
                    )

    return pd.DataFrame(rows)


def _result_to_row(r: ExperimentResult) -> dict:
    return {
        "scheduler": r.scheduler_name,
        "scenario": r.scenario_name,
        "seed": r.seed,
        "num_steps": r.num_steps,
        "PD": r.probability_of_detection,
        "PFA": r.probability_of_false_alarm,
        "hit_rate": r.scan_hit_rate,
        "discovery_ratio": r.activity_discovery_ratio,
        "avg_delay_s": r.avg_discovery_delay,
        "median_delay_s": r.median_discovery_delay,
        "p95_delay_s": r.p95_discovery_delay,
        "scan_efficiency": r.scan_efficiency,
        "coverage": r.band_coverage,
        "starvation_rate": r.starvation_rate,
        "avg_reward": r.avg_reward,
        "wall_clock_s": r.wall_clock_seconds,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across seeds, grouped by scheduler and scenario."""
    metric_cols = [
        "PD", "PFA", "hit_rate", "discovery_ratio", "avg_delay_s",
        "median_delay_s", "p95_delay_s", "scan_efficiency", "coverage",
        "starvation_rate", "avg_reward",
    ]
    # groupby(...).mean() returns a DataFrame here; cast past the pandas stubs'
    # DataFrame|Series|scalar union so .reset_index() type-checks.
    means = cast(pd.DataFrame, df.groupby(["scenario", "scheduler"])[metric_cols].mean())
    return means.reset_index()


def scheduler_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Average each scheduler's metrics across all scenarios for an overall ranking."""
    metric_cols = [
        "PD", "discovery_ratio", "avg_delay_s", "scan_efficiency",
        "coverage", "avg_reward",
    ]
    means = cast(pd.DataFrame, df.groupby("scheduler")[metric_cols].mean())
    return means.sort_values("discovery_ratio", ascending=False).reset_index()
