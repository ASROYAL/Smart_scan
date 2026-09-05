#!/usr/bin/env python3
"""Generate a recorded IQ dataset from a scenario, saved as complex64 files.

Produces a continuous IQ recording of one frequency window over time, which can
later be replayed through the SAME DSP pipeline via RecordedIQSource.

Usage:
    python scripts/generate_dataset.py --scenario periodic --duration 2.0
    python scripts/generate_dataset.py --scenario sparse --center 500e6 --output data/recordings/sparse.c64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from smartscan.acquisition.file_source import write_iq_file
from smartscan.core.config import load_config
from smartscan.simulation.environment import RFEnvironment
from smartscan.simulation.scenarios import get_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a recorded IQ dataset")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--scenario", default="periodic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=1.0, help="seconds")
    parser.add_argument("--center", type=float, default=None,
                        help="Center frequency to record (Hz). Defaults to config center.")
    parser.add_argument("--output", default=None,
                        help="Output .c64 path (default data/recordings/<scenario>.c64)")
    parser.add_argument("--chunk", type=float, default=0.01,
                        help="Chunk duration in seconds for time-varying generation")
    args = parser.parse_args()

    config = load_config(args.config)
    sample_rate = config.environment.sample_rate
    center = args.center if args.center is not None else config.environment.center_frequency
    bandwidth = config.receiver.instantaneous_bandwidth

    scenario = get_scenario(
        args.scenario, seed=args.seed,
        num_bands=config.environment.num_bands,
        total_bw=config.environment.total_bandwidth,
        center=config.environment.center_frequency,
    )

    env = RFEnvironment(
        emitter_configs=scenario.emitters,
        noise_power_dbm=scenario.noise_power_dbm,
        sample_rate=sample_rate,
        seed=args.seed,
    )

    # Generate the recording chunk-by-chunk so time-varying emitters are captured
    chunk_samples = int(sample_rate * args.chunk)
    num_chunks = max(1, int(args.duration / args.chunk))

    chunks = []
    for i in range(num_chunks):
        t = i * args.chunk
        env.set_time(t)
        chunk = env.generate_samples(center, bandwidth, chunk_samples, time=t)
        chunks.append(chunk)

    iq = np.concatenate(chunks)

    output = args.output or f"data/recordings/{args.scenario}.c64"
    write_iq_file(iq, output, sample_rate=sample_rate,
                  center_frequency=center, start_time=0.0)

    out_path = Path(output)
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(iq)} IQ samples ({size_mb:.1f} MB) to {out_path}")
    print(f"Metadata sidecar: {out_path.with_suffix('.json')}")
    print(f"  sample_rate={sample_rate:.0f} center={center:.0f} duration={args.duration}s")


if __name__ == "__main__":
    main()
