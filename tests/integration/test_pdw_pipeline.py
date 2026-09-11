"""Integration: full scan loop runs end-to-end on a PDW (radar ESM) environment."""

from smartscan.core.config import SmartScanConfig
from smartscan.evaluation.experiment import build_and_run_pdw
from smartscan.simulation.pdw import generate_synthetic_pdws


def _config(steps=1500, num_bands=40):
    cfg = SmartScanConfig()
    cfg.simulation.num_steps = steps
    cfg.simulation.seed = 1
    cfg.environment.num_bands = num_bands
    return cfg


def test_pdw_pipeline_runs_and_scores():
    cfg = _config()
    pdws = generate_synthetic_pdws(num_emitters=6, duration=20.0, seed=1,
                                   total_bw=cfg.environment.total_bandwidth,
                                   center=cfg.environment.center_frequency)
    outcome = build_and_run_pdw(cfg, "adaptive", pdws, seed=1)
    r = outcome.result
    assert r.num_steps == 1500
    assert 0.0 <= r.probability_of_detection <= 1.0
    assert 0.0 <= r.activity_discovery_ratio <= 1.0
    # the loop actually detected some radar activity
    assert r.scan_hit_rate > 0.0


def test_pdw_scheduler_beats_random_on_efficiency():
    """A learning scheduler should be no worse than random at finding radar pulses."""
    cfg = _config(steps=1200)
    pdws = generate_synthetic_pdws(num_emitters=6, duration=20.0, seed=2,
                                   total_bw=cfg.environment.total_bandwidth,
                                   center=cfg.environment.center_frequency)
    adaptive = build_and_run_pdw(cfg, "adaptive", pdws, seed=2).result
    rnd = build_and_run_pdw(cfg, "random", pdws, seed=2).result
    assert adaptive.scan_efficiency >= rnd.scan_efficiency


def test_pdw_information_wall():
    """The PDW source exposes no ground truth to the scheduler path."""
    from smartscan.acquisition.pdw_source import PDWRFSource
    from smartscan.simulation.pdw import PDWEnvironment

    env = PDWEnvironment(generate_synthetic_pdws(num_emitters=3, duration=5.0, seed=1))
    src = PDWRFSource(env)
    # source has no method returning emitter identity / truth
    public = [m for m in dir(src) if not m.startswith("_")]
    assert "read_samples" in public
    assert not any("truth" in m or "emitter" in m or "label" in m for m in public)
