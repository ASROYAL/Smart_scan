"""Benchmark scenario definitions.

Each scenario returns a list of EmitterConfig plus metadata. Scenarios are
reproducible given a seed. All schedulers run against identical scenarios for
fair comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from smartscan.core.models import EmitterConfig, EmitterType
from smartscan.telemetry.phone import quality_to_snr


@dataclass
class Scenario:
    name: str
    description: str
    emitters: list[EmitterConfig]
    noise_power_dbm: float = -100.0
    duration: float = 10.0


def _band_center(band_idx: int, num_bands: int, total_bw: float, center: float) -> float:
    """Center frequency of a given band index."""
    band_width = total_bw / num_bands
    start = center - total_bw / 2
    return start + (band_idx + 0.5) * band_width


def scenario_sparse(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Few bands transmit continuously."""
    rng = np.random.default_rng(seed)
    active_bands = rng.choice(num_bands, size=3, replace=False)
    emitters = [
        EmitterConfig(
            emitter_id=i, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=_band_center(b, num_bands, total_bw, center),
            bandwidth=5e6, amplitude=1.0, snr_db=25.0,
        )
        for i, b in enumerate(active_bands)
    ]
    return Scenario("sparse", "Few bands transmit continuously", emitters, duration=10.0)


def scenario_dense(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Many bands transmit."""
    rng = np.random.default_rng(seed)
    active_bands = rng.choice(num_bands, size=20, replace=False)
    emitters = [
        EmitterConfig(
            emitter_id=i, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=_band_center(b, num_bands, total_bw, center),
            bandwidth=5e6, amplitude=1.0, snr_db=20.0,
        )
        for i, b in enumerate(active_bands)
    ]
    return Scenario("dense", "Many bands transmit", emitters, duration=10.0)


def scenario_periodic(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Periodic sources with distinct periods."""
    rng = np.random.default_rng(seed)
    active_bands = rng.choice(num_bands, size=5, replace=False)
    periods = [0.5, 1.0, 1.5, 2.0, 2.5]
    emitters = [
        EmitterConfig(
            emitter_id=i, emitter_type=EmitterType.PERIODIC_BURST,
            center_frequency=_band_center(b, num_bands, total_bw, center),
            bandwidth=5e6, amplitude=1.0, snr_db=25.0,
            period=periods[i % len(periods)], duty_cycle=0.3,
        )
        for i, b in enumerate(active_bands)
    ]
    return Scenario("periodic", "Periodic sources", emitters, duration=15.0)


def scenario_random_burst(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Random burst sources with stochastic timing."""
    rng = np.random.default_rng(seed)
    active_bands = rng.choice(num_bands, size=5, replace=False)
    emitters = [
        EmitterConfig(
            emitter_id=i, emitter_type=EmitterType.RANDOM_BURST,
            center_frequency=_band_center(b, num_bands, total_bw, center),
            bandwidth=5e6, amplitude=1.0, snr_db=25.0,
            burst_rate=2.0, burst_duration=0.2,
        )
        for i, b in enumerate(active_bands)
    ]
    return Scenario("random_burst", "Random burst sources", emitters, duration=15.0)


def scenario_frequency_hopping(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Frequency-changing sources that hop between bands."""
    rng = np.random.default_rng(seed)
    hop_sets = []
    for _ in range(2):
        bands = rng.choice(num_bands, size=4, replace=False)
        hop_sets.append([_band_center(b, num_bands, total_bw, center) for b in bands])
    emitters = [
        EmitterConfig(
            emitter_id=i, emitter_type=EmitterType.FREQUENCY_HOPPING,
            center_frequency=hop_sets[i][0],
            bandwidth=5e6, amplitude=1.0, snr_db=25.0,
            hop_frequencies=hop_sets[i], hop_interval=1.0,
        )
        for i in range(len(hop_sets))
    ]
    return Scenario("frequency_hopping", "Frequency-changing sources", emitters, duration=15.0)


def scenario_changing(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Environment changes halfway: one set of emitters turns off, another turns on."""
    rng = np.random.default_rng(seed)
    first = rng.choice(num_bands, size=3, replace=False)
    second = rng.choice(num_bands, size=3, replace=False)
    half = 7.5
    emitters = []
    eid = 0
    for b in first:
        emitters.append(EmitterConfig(
            emitter_id=eid, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=_band_center(b, num_bands, total_bw, center),
            bandwidth=5e6, amplitude=1.0, snr_db=25.0,
            start_time=0.0, end_time=half,
        ))
        eid += 1
    for b in second:
        emitters.append(EmitterConfig(
            emitter_id=eid, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=_band_center(b, num_bands, total_bw, center),
            bandwidth=5e6, amplitude=1.0, snr_db=25.0,
            start_time=half, end_time=None,
        ))
        eid += 1
    return Scenario("changing", "Activity statistics change midway", emitters, duration=15.0)


def scenario_low_snr(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Low-SNR sources near the detection limit."""
    rng = np.random.default_rng(seed)
    active_bands = rng.choice(num_bands, size=4, replace=False)
    emitters = [
        EmitterConfig(
            emitter_id=i, emitter_type=EmitterType.CONTINUOUS,
            center_frequency=_band_center(b, num_bands, total_bw, center),
            bandwidth=5e6, amplitude=1.0, snr_db=8.0,  # low SNR
        )
        for i, b in enumerate(active_bands)
    ]
    return Scenario("low_snr", "Low-SNR sources", emitters, duration=10.0)


def scenario_phone_training(
    signal_quality: int,
    seed: int = 42,
    num_bands: int = 50,
    total_bw: float = 1000e6,
    center: float = 500e6,
) -> Scenario:
    """Consented phone-link strength driving a synthetic training emitter.

    The link score configures the simulated signal power. It is never passed to
    a scheduler; schedulers observe only IQ produced by the RF environment.
    """

    rng = np.random.default_rng(seed)
    target_band = int(rng.integers(max(1, num_bands // 5), max(2, 4 * num_bands // 5)))
    decoy_bands = [int(b) for b in rng.choice(num_bands, size=min(3, num_bands), replace=False)]
    target = EmitterConfig(
        emitter_id=0,
        emitter_type=EmitterType.PERIODIC_BURST,
        center_frequency=_band_center(target_band, num_bands, total_bw, center),
        bandwidth=5e6,
        amplitude=1.0,
        snr_db=quality_to_snr(signal_quality),
        period=0.8,
        duty_cycle=0.35,
    )
    decoys = [
        EmitterConfig(
            emitter_id=index + 1,
            emitter_type=EmitterType.RANDOM_BURST,
            center_frequency=_band_center(band, num_bands, total_bw, center),
            bandwidth=5e6,
            amplitude=1.0,
            snr_db=8.0 + index * 3.0,
            burst_rate=1.0 + 0.25 * index,
            burst_duration=0.12,
        )
        for index, band in enumerate(decoy_bands)
        if band != target_band
    ]
    return Scenario(
        "phone_training",
        f"Phone-link training beacon at {signal_quality}% strength with background activity",
        [target, *decoys],
        duration=12.0,
    )


def scenario_mixed(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Mixture of all activity types."""
    rng = np.random.default_rng(seed)
    bands = rng.choice(num_bands, size=8, replace=False)
    bc = lambda b: _band_center(b, num_bands, total_bw, center)
    emitters = [
        EmitterConfig(emitter_id=0, emitter_type=EmitterType.CONTINUOUS,
                       center_frequency=bc(bands[0]), bandwidth=5e6,
                       amplitude=1.0, snr_db=25.0),
        EmitterConfig(emitter_id=1, emitter_type=EmitterType.PERIODIC_BURST,
                       center_frequency=bc(bands[1]), bandwidth=5e6,
                       amplitude=1.0, snr_db=25.0, period=1.0, duty_cycle=0.3),
        EmitterConfig(emitter_id=2, emitter_type=EmitterType.PERIODIC_BURST,
                       center_frequency=bc(bands[2]), bandwidth=5e6,
                       amplitude=1.0, snr_db=22.0, period=2.0, duty_cycle=0.25),
        EmitterConfig(emitter_id=3, emitter_type=EmitterType.RANDOM_BURST,
                       center_frequency=bc(bands[3]), bandwidth=5e6,
                       amplitude=1.0, snr_db=25.0, burst_rate=1.5, burst_duration=0.2),
        EmitterConfig(emitter_id=4, emitter_type=EmitterType.FREQUENCY_HOPPING,
                       center_frequency=bc(bands[4]), bandwidth=5e6,
                       amplitude=1.0, snr_db=25.0,
                       hop_frequencies=[bc(bands[4]), bc(bands[5])], hop_interval=1.5),
        EmitterConfig(emitter_id=5, emitter_type=EmitterType.CONTINUOUS,
                       center_frequency=bc(bands[6]), bandwidth=5e6,
                       amplitude=1.0, snr_db=10.0),  # low SNR
        EmitterConfig(emitter_id=6, emitter_type=EmitterType.PERIODIC_BURST,
                       center_frequency=bc(bands[7]), bandwidth=5e6,
                       amplitude=1.0, snr_db=25.0, period=3.0, duty_cycle=0.1),  # rare
    ]
    return Scenario("mixed", "Mixture of activity types", emitters, duration=15.0)


def scenario_radar_scan(
    seed: int = 42, num_bands: int = 50, total_bw: float = 1000e6, center: float = 500e6,
) -> Scenario:
    """Rotating-antenna (spatially scanning) radars, some frequency-agile.

    Models the classic ES intercept problem: each radar illuminates the receiver
    only during a brief main-beam dwell once per antenna rotation, so a naive
    sweep easily misses it between visits.
    """
    rng = np.random.default_rng(seed)
    active_bands = rng.choice(num_bands, size=4, replace=False)
    bc = lambda b: _band_center(b, num_bands, total_bw, center)
    scan_periods = [2.0, 3.0, 4.0, 5.0]     # antenna rotation periods (s)
    beam_dwells = [0.2, 0.25, 0.3, 0.2]      # main-beam illumination (s)
    emitters = []
    for i, b in enumerate(active_bands):
        # make the last radar frequency-agile across two bands
        agile = i == len(active_bands) - 1
        hop = [bc(b), bc(active_bands[0])] if agile else None
        emitters.append(EmitterConfig(
            emitter_id=i, emitter_type=EmitterType.RADAR_SCAN,
            center_frequency=bc(b), bandwidth=5e6, amplitude=1.0, snr_db=25.0,
            scan_period=scan_periods[i % len(scan_periods)],
            beam_dwell=beam_dwells[i % len(beam_dwells)],
            hop_frequencies=hop,
        ))
    return Scenario("radar_scan", "Rotating-antenna scanning radars", emitters, duration=20.0)


SCENARIO_FACTORIES = {
    "sparse": scenario_sparse,
    "dense": scenario_dense,
    "periodic": scenario_periodic,
    "random_burst": scenario_random_burst,
    "frequency_hopping": scenario_frequency_hopping,
    "changing": scenario_changing,
    "low_snr": scenario_low_snr,
    "mixed": scenario_mixed,
    "radar_scan": scenario_radar_scan,
}


def get_scenario(name: str, **kwargs) -> Scenario:
    """Construct a scenario by name."""
    if name not in SCENARIO_FACTORIES:
        raise ValueError(f"Unknown scenario: {name}. Options: {list(SCENARIO_FACTORIES)}")
    return SCENARIO_FACTORIES[name](**kwargs)


ALL_SCENARIOS = list(SCENARIO_FACTORIES.keys())
