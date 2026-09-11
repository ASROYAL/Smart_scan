#!/usr/bin/env python3
"""Receiver sensitivity sweep: probability of detection vs SNR.

Characterises the detector's sensitivity — one of the EW figures of merit —
by injecting a tone at a range of SNRs into noise and measuring PD, plus the
noise-only false-alarm rate (PFA).

Example:
    python scripts/sensitivity_sweep.py
    python scripts/sensitivity_sweep.py --margin 4 --trials 100
"""

from __future__ import annotations

import argparse

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta
from smartscan.dsp.detector import EnergyDetector
from smartscan.evaluation.metrics import sensitivity_curve


def main() -> None:
    ap = argparse.ArgumentParser(description="Detector sensitivity sweep (PD vs SNR)")
    ap.add_argument("--margin", type=float, default=6.0, help="threshold margin (dB)")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--fft", type=int, default=1024)
    ap.add_argument("--samples", type=int, default=4096)
    args = ap.parse_args()

    det = EnergyDetector(DetectorConfig(fft_size=args.fft, threshold_margin_db=args.margin))

    def make_meta(n):
        return AcquisitionMeta(center_frequency=100e6, sample_rate=20e6, bandwidth=20e6,
                               num_samples=n, timestamp=0.0, dwell_time=n / 20e6)

    # swept low because a coherent tone gains ~10*log10(fft_size) dB of FFT
    # processing gain, so the detection knee sits well below 0 dB wideband SNR
    snrs = [-45, -42, -39, -36, -33, -30, -27, -24, -21, -18, -15, -12, -9, -6, 0]
    curve = sensitivity_curve(det, make_meta, snr_values=snrs, trials=args.trials,
                              num_samples=args.samples, seed=0)

    print(f"Detector sensitivity  (margin {args.margin} dB, {args.trials} trials/pt)")
    print(f"{'SNR (dB)':>9} | {'PD':>6} | {'PFA':>6}")
    print("-" * 28)
    for snr, pd, pfa in curve:
        bar = "#" * int(pd * 20)
        print(f"{snr:>9.0f} | {pd:>6.2f} | {pfa:>6.2f}  {bar}")
    # crude sensitivity: lowest SNR reaching PD >= 0.9
    thr = next((snr for snr, pd, _ in curve if pd >= 0.9), None)
    if thr is not None:
        print(f"\nSensitivity (PD>=0.9) reached at ~{thr:.0f} dB SNR")


if __name__ == "__main__":
    main()
