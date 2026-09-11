#!/usr/bin/env python3
"""Self-check: run a battery of functional tests and print a PASS/FAIL checklist.

A quick "is everything working?" tool — separate from the full pytest suite.

    python scripts/selfcheck.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, detail))
        except AssertionError as e:
            RESULTS.append((name, False, f"assertion: {e}"))
        except Exception as e:  # noqa: BLE001 - a check harness must survive any failure
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return deco


def _cfg(steps=300, bands=40, detector="energy"):
    from smartscan.core.config import SmartScanConfig
    c = SmartScanConfig()
    c.simulation.num_steps = steps
    c.environment.num_bands = bands
    c.detector.detector_type = detector
    return c


@check("Package imports")
def _():
    import smartscan
    return smartscan.__file__


@check("All 3 detectors detect a strong tone and reject noise")
def _():
    from smartscan.core.config import DetectorConfig
    from smartscan.core.models import AcquisitionMeta
    from smartscan.dsp.detector import create_detector
    from smartscan.simulation.noise import generate_awgn, power_for_snr
    from smartscan.simulation.waveform import generate_tone
    meta = AcquisitionMeta(center_frequency=100e6, sample_rate=20e6, bandwidth=20e6,
                           num_samples=4096, timestamp=0.0, dwell_time=4096/20e6)
    out = []
    for dt in ("energy", "matched_filter", "cyclostationary"):
        det = create_detector(DetectorConfig(detector_type=dt))
        rng = np.random.default_rng(0)
        sig = generate_awgn(4096, -100, rng) + generate_tone(4096, 20e6, 2e6, power_dbm=power_for_snr(-100, 25))
        assert det.detect(sig, meta).detected, f"{dt} missed a strong tone"
        fa = sum(det.detect(generate_awgn(4096, -100, np.random.default_rng(s)), meta).detected
                 for s in range(15))
        assert fa <= 4, f"{dt} false-alarm rate too high ({fa}/15)"
        out.append(dt)
    return " / ".join(out)


@check("All 7 schedulers run and produce valid metrics")
def _():
    from smartscan.evaluation.experiment import build_and_run
    from smartscan.schedulers.factory import ALL_SCHEDULER_TYPES
    names = []
    for s in ALL_SCHEDULER_TYPES:
        r = build_and_run(_cfg(steps=200), s, "mixed", seed=1).result
        assert 0.0 <= r.probability_of_detection <= 1.0
        assert 0.0 <= r.activity_discovery_ratio <= 1.0
        names.append(s.value)
    return f"{len(names)} schedulers ok"


@check("Learning beats open-loop baseline (dense scenario)")
def _():
    from smartscan.evaluation.experiment import build_and_run
    a = build_and_run(_cfg(steps=800), "adaptive", "dense", seed=1).result
    rr = build_and_run(_cfg(steps=800), "round_robin", "dense", seed=1).result
    assert a.scan_efficiency > rr.scan_efficiency, \
        f"adaptive {a.scan_efficiency:.2f} !> round_robin {rr.scan_efficiency:.2f}"
    return f"adaptive eff {a.scan_efficiency:.2f} > round_robin {rr.scan_efficiency:.2f}"


@check("Information wall — scheduler cannot see ground truth")
def _():
    from smartscan.core.config import ReceiverConfig, SchedulerConfig
    from smartscan.core.models import BandState
    from smartscan.schedulers.factory import create_scheduler
    s = create_scheduler("adaptive", ReceiverConfig(), SchedulerConfig(), 10, seed=1)
    # BandState (the scheduler's only input) must carry no ground-truth fields
    forbidden = {"truly_active", "emitter_id", "ground_truth", "is_active"}
    assert set(BandState.model_fields).isdisjoint(forbidden)
    # scheduler holds no environment/emitter reference
    blob = " ".join(type(s).__dict__.keys()).lower()
    assert "environment" not in blob and "emitter" not in blob
    return "no ground-truth fields on BandState; no env reference"


@check("Radar PDW pipeline — learning schedulers intercept far more efficiently")
def _():
    from smartscan.evaluation.experiment import build_and_run_pdw
    from smartscan.simulation.pdw import generate_synthetic_pdws
    cfg = _cfg(steps=1200)
    pdws = generate_synthetic_pdws(num_emitters=6, duration=20.0, seed=1,
                                   total_bw=cfg.environment.total_bandwidth,
                                   center=cfg.environment.center_frequency)
    ucb = build_and_run_pdw(cfg, "bandit_ucb", pdws, seed=1).result
    rr = build_and_run_pdw(cfg, "round_robin", pdws, seed=1).result
    # a learning scheduler intercepts several-fold more per scan than a blind sweep
    assert ucb.scan_hit_rate > 3 * rr.scan_hit_rate
    return (f"bandit_ucb hit-rate {ucb.scan_hit_rate:.3f} vs round_robin {rr.scan_hit_rate:.3f} "
            f"({ucb.scan_hit_rate/max(rr.scan_hit_rate,1e-9):.1f}x)")


@check("Sensitivity — weak signals rejected, strong signals detected (proper S-curve)")
def _():
    from smartscan.core.config import DetectorConfig
    from smartscan.core.models import AcquisitionMeta
    from smartscan.dsp.detector import EnergyDetector
    from smartscan.evaluation.metrics import sensitivity_curve
    det = EnergyDetector(DetectorConfig(fft_size=1024))
    mk = lambda n: AcquisitionMeta(center_frequency=100e6, sample_rate=20e6, bandwidth=20e6,
                                   num_samples=n, timestamp=0.0, dwell_time=n/20e6)
    # 4096-sample frames give enough Welch segments for a low false-alarm rate;
    # the real receiver dwells on ~200k samples, so this is conservative.
    curve = sensitivity_curve(det, mk, [-40.0, -20.0, 0.0], trials=30, num_samples=4096)
    pd_low, pd_high = curve[0][1], curve[-1][1]
    assert pd_low < 0.3, f"weak -40dB signal should rarely fire, got PD={pd_low:.2f}"
    assert pd_high > 0.8, f"strong 0dB signal should fire, got PD={pd_high:.2f}"
    return f"PD @ -40dB={pd_low:.2f} (low)  ->  @0dB={pd_high:.2f} (high)"


@check("Better predictions — periodicity is estimated online")
def _():
    from smartscan.evaluation.experiment import build_and_run
    out = build_and_run(_cfg(steps=1500, bands=40), "adaptive", "periodic", seed=1)
    got = [s.estimated_period for s in out.state_manager.all_states() if s.estimated_period]
    assert got, "no band ever got an estimated period"
    return f"{len(got)} band(s) locked a period, e.g. {got[0]:.2f}s"


@check("Reproducibility — same seed gives identical results")
def _():
    from smartscan.evaluation.experiment import build_and_run
    r1 = build_and_run(_cfg(steps=300), "adaptive", "mixed", seed=7).result
    r2 = build_and_run(_cfg(steps=300), "adaptive", "mixed", seed=7).result
    assert abs(r1.scan_efficiency - r2.scan_efficiency) < 1e-9
    return f"scan_efficiency identical ({r1.scan_efficiency:.4f})"


def main() -> int:
    print("\n" + "=" * 66)
    print("  SMART SPECTRUM SCAN — SELF CHECK")
    print("=" * 66)
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        tag = "\033[92m[PASS]\033[0m" if ok else "\033[91m[FAIL]\033[0m"
        print(f"  {tag}  {name}")
        if detail:
            print(f"         └─ {detail}")
    print("-" * 66)
    print(f"  {passed}/{len(RESULTS)} checks passed")
    print("=" * 66 + "\n")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - top-level guard: print traceback, exit non-zero
        traceback.print_exc()
        sys.exit(2)
