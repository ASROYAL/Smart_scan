# Smart Adaptive Spectrum Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, modular system for adaptive RF spectrum scanning that learns from observations to minimize signal discovery delay.

**Architecture:** RF environment simulator generates IQ samples → receiver model observes limited bandwidth → DSP pipeline detects activity → band state tracker maintains history → scheduler selects next band using exploration/exploitation. Ground truth is strictly separated from the scheduler; only the evaluator can access it.

**Tech Stack:** Python 3.12+, numpy, scipy, pandas, scikit-learn, pydantic, matplotlib, plotly, streamlit, SQLite, pytest, hypothesis

---

## Phase 1 — Repository Foundation

### Task 1: Project Configuration Files

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "smartscan"
version = "0.1.0"
description = "Smart Adaptive Spectrum Scan & Signal Monitoring System"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26",
    "scipy>=1.12",
    "pandas>=2.2",
    "scikit-learn>=1.4",
    "matplotlib>=3.8",
    "plotly>=5.18",
    "pydantic>=2.6",
    "pyyaml>=6.0",
    "streamlit>=1.31",
    "joblib>=1.3",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "hypothesis>=6.98",
    "ruff>=0.2",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Create requirements.txt**

```
numpy>=1.26
scipy>=1.12
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
plotly>=5.18
pydantic>=2.6
pyyaml>=6.0
streamlit>=1.31
joblib>=1.3
pytest>=8.0
pytest-cov>=4.1
hypothesis>=6.98
```

- [ ] **Step 3: Create .gitignore**

Standard Python gitignore + data directories.

- [ ] **Step 4: Verify with `pip install -e ".[dev]"`**

---

### Task 2: Core Data Models

**Files:**
- Create: `src/smartscan/core/models.py`
- Create: `src/smartscan/core/__init__.py`
- Test: `tests/unit/test_models.py`

All shared data contracts: ScanDecision, DetectionResult, BandObservation,
BandState, EmitterConfig, ReceiverConfig, etc. Using Pydantic for validation.

- [ ] **Step 1: Write failing tests for data models**

Test that ScanDecision, DetectionResult, BandObservation, BandState
can be constructed with valid data and reject invalid data.

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement all data models in models.py**

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

---

### Task 3: Configuration System

**Files:**
- Create: `src/smartscan/core/config.py`
- Create: `config/default.yaml`
- Test: `tests/unit/test_config.py`

Pydantic settings classes that load from YAML.

- [ ] **Step 1: Write failing tests for config loading**

- [ ] **Step 2: Run tests — expect FAIL**

- [ ] **Step 3: Implement config classes and YAML loader**

- [ ] **Step 4: Create default.yaml with all baseline parameters**

- [ ] **Step 5: Run tests — expect PASS**

- [ ] **Step 6: Commit**

---

### Task 4: Abstract Interfaces

**Files:**
- Create: `src/smartscan/acquisition/base.py` — RFSource ABC
- Create: `src/smartscan/schedulers/base.py` — BaseScheduler ABC
- Create: `src/smartscan/dsp/base.py` — BaseDetector ABC
- Create: `src/smartscan/prediction/base.py` — BasePredictor ABC
- Test: `tests/unit/test_interfaces.py`

- [ ] **Step 1: Write tests verifying ABCs cannot be instantiated directly**

- [ ] **Step 2: Implement all abstract base classes**

- [ ] **Step 3: Run tests — expect PASS**

- [ ] **Step 4: Commit**

---

### Task 5: Logging & Timing Utilities

**Files:**
- Create: `src/smartscan/utils/logging.py`
- Create: `src/smartscan/utils/timing.py`
- Test: `tests/unit/test_utils.py`

- [ ] **Step 1: Implement structured logger (CSV/JSON export)**

- [ ] **Step 2: Implement timing context manager**

- [ ] **Step 3: Write tests**

- [ ] **Step 4: Commit**

---

## Phase 2 — RF Simulator

### Task 6: Noise Generator

**Files:**
- Create: `src/smartscan/simulation/noise.py`
- Test: `tests/unit/test_noise.py`

- [ ] **Step 1: Write tests — Gaussian noise has correct power, is complex**

- [ ] **Step 2: Implement AWGN generator**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 7: Waveform Generator

**Files:**
- Create: `src/smartscan/simulation/waveform.py`
- Test: `tests/unit/test_waveform.py`

Generate complex sinusoids at specified frequency offsets with configurable amplitude.

- [ ] **Step 1: Write tests — signal at correct frequency, correct power**

- [ ] **Step 2: Implement complex tone generator**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 8: Emitter Models

**Files:**
- Create: `src/smartscan/simulation/emitters.py`
- Test: `tests/unit/test_emitters.py`

Continuous, PeriodicBurst, RandomBurst, FrequencyHopping, ScanLike emitters.
Each emitter knows its ON/OFF schedule and generates IQ when active.

- [ ] **Step 1: Write tests for each emitter type's activity schedule**

- [ ] **Step 2: Implement all emitter classes**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 9: RF Environment

**Files:**
- Create: `src/smartscan/simulation/environment.py`
- Test: `tests/unit/test_environment.py`

Combines emitters + noise. Generates IQ for any requested frequency window.
Maintains ground truth log internally.

- [ ] **Step 1: Write tests — environment returns IQ, ground truth is correct**

- [ ] **Step 2: Implement RFEnvironment class**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 10: Simulator RF Source Adapter

**Files:**
- Create: `src/smartscan/acquisition/simulator_source.py`
- Test: `tests/unit/test_simulator_source.py`

Wraps RFEnvironment behind the RFSource interface. Returns IQ samples only (no ground truth).

- [ ] **Step 1: Write tests — source returns samples, no ground truth leaks**

- [ ] **Step 2: Implement SimulatedRFSource**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 3 — Receiver Model

### Task 11: Receiver

**Files:**
- Create: `src/smartscan/receiver/receiver.py`
- Test: `tests/unit/test_receiver.py`

Models limited-bandwidth observation with tuning delay.

- [ ] **Step 1: Write tests — receiver only returns data for tuned window**

- [ ] **Step 2: Implement Receiver class**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 4 — DSP Pipeline

### Task 12: FFT & PSD

**Files:**
- Create: `src/smartscan/dsp/fft.py`
- Create: `src/smartscan/dsp/psd.py`
- Test: `tests/unit/test_dsp.py`

- [ ] **Step 1: Write tests — known tone produces peak at correct bin**

- [ ] **Step 2: Implement windowed FFT and Welch PSD**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 13: Noise Floor Estimator

**Files:**
- Create: `src/smartscan/dsp/noise_floor.py`
- Test: `tests/unit/test_noise_floor.py`

- [ ] **Step 1: Write tests — median estimator on known noise**

- [ ] **Step 2: Implement median and percentile noise estimators**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 14: Activity Detector

**Files:**
- Create: `src/smartscan/dsp/detector.py`
- Test: `tests/unit/test_detector.py`

Energy detector that compares PSD to adaptive threshold.

- [ ] **Step 1: Write tests — detects strong signal, misses sub-threshold**

- [ ] **Step 2: Implement EnergyDetector**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 5 — State Tracking

### Task 15: Band State Manager

**Files:**
- Create: `src/smartscan/state/spectrum_state.py`
- Create: `src/smartscan/state/band_history.py`
- Test: `tests/unit/test_state.py`

- [ ] **Step 1: Write tests for band state updates**

- [ ] **Step 2: Implement SpectrumStateManager and BandHistory**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 16: Feature Extractor

**Files:**
- Create: `src/smartscan/features/extractor.py`
- Test: `tests/unit/test_features.py`

- [ ] **Step 1: Write tests for feature computation**

- [ ] **Step 2: Implement FeatureExtractor**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 17: SQLite Persistence

**Files:**
- Create: `src/smartscan/state/database.py`
- Test: `tests/unit/test_database.py`

- [ ] **Step 1: Write tests for save/load round-trip**

- [ ] **Step 2: Implement SQLite backend**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 6 — Baseline Schedulers

### Task 18: Round Robin Scheduler

**Files:**
- Create: `src/smartscan/schedulers/round_robin.py`
- Test: `tests/unit/test_round_robin.py`

- [ ] **Step 1: Write tests — cycles through all bands sequentially**

- [ ] **Step 2: Implement RoundRobinScheduler**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 19: Random Scheduler

**Files:**
- Create: `src/smartscan/schedulers/random_scan.py`
- Test: `tests/unit/test_random_scan.py`

- [ ] **Step 1: Write tests — selects bands, covers all eventually**

- [ ] **Step 2: Implement RandomScanScheduler**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 20: Priority Scheduler

**Files:**
- Create: `src/smartscan/schedulers/priority_scan.py`
- Test: `tests/unit/test_priority_scan.py`

- [ ] **Step 1: Write tests — high-activity bands get more visits, no starvation**

- [ ] **Step 2: Implement PriorityScanScheduler**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 7 — Evaluation

### Task 21: Ground Truth Evaluator

**Files:**
- Create: `src/smartscan/evaluation/metrics.py`
- Create: `src/smartscan/evaluation/evaluator.py`
- Test: `tests/unit/test_metrics.py`
- Test: `tests/unit/test_evaluator.py`

PD, PFA, discovery delay, coverage, scan efficiency, reward.

- [ ] **Step 1: Write tests for each metric with known inputs**

- [ ] **Step 2: Implement metric functions**

- [ ] **Step 3: Implement Evaluator class**

- [ ] **Step 4: Run tests — PASS**

- [ ] **Step 5: Commit**

---

### Task 22: Integration Test — Full Pipeline

**Files:**
- Test: `tests/integration/test_scan_loop.py`
- Test: `tests/integration/test_information_separation.py`

End-to-end: simulator → receiver → DSP → state → scheduler → evaluator.
Plus tests proving scheduler cannot access ground truth.

- [ ] **Step 1: Write integration test with deterministic scenario**

- [ ] **Step 2: Write information-separation tests**

- [ ] **Step 3: Run all tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 8 — Smart Scheduler

### Task 23: UCB1 Bandit Scheduler

**Files:**
- Create: `src/smartscan/schedulers/bandit.py`
- Test: `tests/unit/test_bandit.py`

- [ ] **Step 1: Write tests — exploration decreases over time**

- [ ] **Step 2: Implement UCB1BanditScheduler**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 24: Contextual Bandit (Adaptive) Scheduler

**Files:**
- Create: `src/smartscan/schedulers/adaptive.py`
- Test: `tests/unit/test_adaptive.py`

Uses feature vectors per band as context.

- [ ] **Step 1: Write tests — uses features, updates online**

- [ ] **Step 2: Implement AdaptiveScheduler**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 9 — Temporal Prediction

### Task 25: Periodicity Estimator

**Files:**
- Create: `src/smartscan/prediction/periodicity.py`
- Test: `tests/unit/test_periodicity.py`

Autocorrelation-based period estimation.

- [ ] **Step 1: Write tests — estimates known period within 10%**

- [ ] **Step 2: Implement periodicity estimator**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 26: Statistical Predictor

**Files:**
- Create: `src/smartscan/prediction/statistical.py`
- Test: `tests/unit/test_statistical.py`

EWMA + transition probabilities.

- [ ] **Step 1: Write tests**

- [ ] **Step 2: Implement StatisticalPredictor**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

## Phase 10 — Benchmark

### Task 27: Scenario Definitions

**Files:**
- Create: `src/smartscan/simulation/scenarios.py`
- Create: `config/experiment.yaml`
- Test: `tests/unit/test_scenarios.py`

8 benchmark scenarios (sparse, dense, periodic, burst, hopping, changing, low-SNR, mixed).

- [ ] **Step 1: Implement scenario factory**

- [ ] **Step 2: Run tests — PASS**

- [ ] **Step 3: Commit**

---

### Task 28: Benchmark Runner

**Files:**
- Create: `src/smartscan/evaluation/benchmark.py`
- Create: `scripts/benchmark_schedulers.py`

- [ ] **Step 1: Implement benchmark that runs all schedulers × all scenarios**

- [ ] **Step 2: Produce comparison tables and charts**

- [ ] **Step 3: Commit**

---

## Phase 11 — Dashboard

### Task 29: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`
- Create: `src/smartscan/visualization/spectrum.py`
- Create: `src/smartscan/visualization/timeline.py`
- Create: `src/smartscan/visualization/metrics_viz.py`

- [ ] **Step 1: Implement visualization helper functions**

- [ ] **Step 2: Build multi-page Streamlit app**

- [ ] **Step 3: Verify with `streamlit run dashboard/app.py`**

- [ ] **Step 4: Commit**

---

## Phase 12 — Recorded IQ Support

### Task 30: File-Based IQ Source

**Files:**
- Create: `src/smartscan/acquisition/file_source.py`
- Test: `tests/unit/test_file_source.py`

Read complex64 binary files with separate metadata.

- [ ] **Step 1: Write tests with a small generated test file**

- [ ] **Step 2: Implement RecordedIQSource**

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 31: Experiment & Dataset Scripts

**Files:**
- Create: `scripts/generate_dataset.py`
- Create: `scripts/run_experiment.py`

CLI entry points.

- [ ] **Step 1: Implement CLI scripts**

- [ ] **Step 2: Test with `python scripts/run_experiment.py --scheduler round_robin`**

- [ ] **Step 3: Commit**

---

### Task 32: System Test — Acceptance

**Files:**
- Test: `tests/system/test_acceptance.py`

100 bands, 5 observable, multiple source types, both schedulers, compare metrics.

- [ ] **Step 1: Implement full acceptance test per Section 35**

- [ ] **Step 2: Run and verify**

- [ ] **Step 3: Commit**

---

### Task 33: README

**Files:**
- Create: `README.md`

Full documentation per Section 32.

- [ ] **Step 1: Write comprehensive README**

- [ ] **Step 2: Commit**
