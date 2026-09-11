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
        "missed_event_rate": r.missed_event_rate,
        "censored_avg_intercept_time": r.censored_avg_intercept_time,
        "wall_clock_s": r.wall_clock_seconds,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across seeds, grouped by scheduler and scenario."""
    metric_cols = [
        "PD", "PFA", "hit_rate", "discovery_ratio", "avg_delay_s",
        "median_delay_s", "p95_delay_s", "scan_efficiency", "coverage",
        "starvation_rate", "avg_reward",
        "missed_event_rate", "censored_avg_intercept_time",
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


def bootstrap_confidence_intervals(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Seed-resampled 95% confidence intervals for each scheduler and metric."""

    import numpy as np

    metrics = metrics or ["discovery_ratio", "censored_avg_intercept_time", "scan_efficiency"]
    rng = np.random.default_rng(seed)
    rows = []
    for scheduler, group in df.groupby("scheduler"):
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
            rows.append({
                "scheduler": scheduler,
                "metric": metric,
                "mean": float(values.mean()),
                "ci_low": float(np.percentile(samples, 2.5)),
                "ci_high": float(np.percentile(samples, 97.5)),
            })
    return pd.DataFrame(rows)


def paired_seed_differences(
    df: pd.DataFrame, candidate: str, baseline: str = "round_robin",
) -> pd.DataFrame:
    """Paired candidate-minus-baseline differences on identical scenario/seed runs."""

    metrics = ["discovery_ratio", "censored_avg_intercept_time", "scan_efficiency"]
    indexed = df.set_index(["scenario", "seed", "scheduler"])
    rows = []
    pairs = df[["scenario", "seed"]].drop_duplicates().itertuples(index=False)
    for scenario, seed in pairs:
        try:
            cand = indexed.loc[(scenario, seed, candidate)]
            base = indexed.loc[(scenario, seed, baseline)]
        except KeyError:
            continue
        row = {"scenario": scenario, "seed": seed}
        row.update({metric: float(cand[metric] - base[metric]) for metric in metrics})
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """Mark nondominated runs across discovery, efficiency and censored delay."""

    result = df.copy()
    points = result[["discovery_ratio", "scan_efficiency", "censored_avg_intercept_time"]]
    flags = []
    for index, point in points.iterrows():
        dominated = (
            (points["discovery_ratio"] >= point["discovery_ratio"])
            & (points["scan_efficiency"] >= point["scan_efficiency"])
            & (points["censored_avg_intercept_time"] <= point["censored_avg_intercept_time"])
            & (
                (points["discovery_ratio"] > point["discovery_ratio"])
                | (points["scan_efficiency"] > point["scan_efficiency"])
                | (points["censored_avg_intercept_time"] < point["censored_avg_intercept_time"])
            )
        ).any()
        flags.append(not bool(dominated))
    result["pareto_optimal"] = flags
    return result
