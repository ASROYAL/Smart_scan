# Smart Adaptive Spectrum Scan — Architecture

## System Overview

A locally-runnable software system for adaptive scanning of a wide RF spectrum
when there is little or no prior information about signal activity. The receiver's
instantaneous bandwidth (e.g. 20 MHz) is much smaller than the total monitored
spectrum (e.g. 1000 MHz), so the scheduler must intelligently decide where to
tune next, balancing exploration of unknown bands against exploitation of
historically active ones.

## Architecture Diagram

```mermaid
graph TD
    subgraph "Ground Truth (evaluation only)"
        ENV[RF Environment Simulator]
        GT[Ground Truth Log]
    end

    subgraph "Observation Layer"
        SRC[RF Source Abstraction]
        RX[Receiver Model]
    end

    subgraph "Processing Layer"
        DSP[DSP Pipeline<br/>FFT → PSD → Noise Est → Detector]
        FE[Feature Extractor]
    end

    subgraph "State & Memory"
        BS[Band State Manager]
        DB[(SQLite / In-Memory DB)]
        BH[Band History]
    end

    subgraph "Intelligence Layer"
        PRED[Temporal Predictor]
        SCHED[Scheduler<br/>RoundRobin | Random | Priority | Bandit | Adaptive]
    end

    subgraph "Evaluation & Output"
        EVAL[Evaluator<br/>PD, PFA, Discovery Delay, etc.]
        VIZ[Visualization]
        DASH[Streamlit Dashboard]
        LOG[Structured Logger]
    end

    ENV -->|IQ samples| SRC
    SRC -->|tune + read| RX
    RX -->|raw IQ + metadata| DSP
    DSP -->|detection result| FE
    FE -->|features| BS
    BS -->|band states| DB
    BS -->|features + history| SCHED
    BS -->|features + history| PRED
    PRED -->|activity probabilities| SCHED
    SCHED -->|ScanDecision| RX

    ENV -.->|ground truth| GT
    GT -.->|for scoring only| EVAL
    BS -->|observations| EVAL
    EVAL --> VIZ
    EVAL --> DASH
    LOG -->|CSV/JSON| DASH

    style ENV fill:#f9d,stroke:#333
    style GT fill:#f9d,stroke:#333
    style EVAL fill:#dfd,stroke:#333
```

## Information Separation Wall

```
┌─────────────────────────┐    WALL    ┌─────────────────────────┐
│   SIMULATOR SIDE        │    ║║║║    │   SCHEDULER SIDE        │
│                         │    ║║║║    │                         │
│ • Signal ON/OFF state   │    ║║║║    │ • Scan history          │
│ • True frequencies      │    ║║║║    │ • Detector output       │
│ • True amplitudes       │    ║║║║    │ • Extracted features    │
│ • True timing           │    ║║║║    │ • Elapsed time          │
│ • Activity schedule     │    ║║║║    │ • Band state estimates  │
│                         │    ║║║║    │                         │
│ Exposed ONLY to:        │    ║║║║    │ NO access to ground     │
│   • Evaluator           │    ║║║║    │   truth whatsoever      │
│   • IQ sample generator │    ║║║║    │                         │
└─────────────────────────┘    ║║║║    └─────────────────────────┘
```

The only bridge across this wall is the `RFSource.read_samples()` method,
which returns raw IQ samples — never labels.

## Data Flow Per Scan Step

```
1. Scheduler selects band → ScanDecision(band_id, center_freq, bw, dwell)
2. Receiver tunes to center_freq
3. Receiver acquires IQ samples for dwell_time seconds
4. DSP pipeline processes samples:
   a. Apply window function (Hann/Hamming/Blackman)
   b. Compute FFT
   c. Compute PSD (periodogram or Welch)
   d. Estimate noise floor (median/percentile)
   e. Set adaptive threshold = noise_floor + margin_dB
   f. Detect activity above threshold
5. Feature extractor produces BandObservation
6. Band state manager updates history
7. Evaluator (separately) compares detection vs ground truth
8. Reward calculated from detection result (not ground truth)
9. Scheduler updates its model
10. Loop to step 1
```

## Key Data Models

### ScanDecision
```
band_id, center_frequency, bandwidth, dwell_time,
priority_score, reason, timestamp
```

### DetectionResult
```
detected (bool), confidence, peak_power_dB, avg_power_dB,
noise_floor_dB, estimated_snr_dB, freq_start, freq_end, timestamp
```

### BandObservation
```
band_id, timestamp, detected, confidence, power, snr,
noise_floor, features dict
```

### BandState
```
band_id, freq_start, freq_end, last_scan_time, last_detection_time,
hit_count, miss_count, observation_count, rolling_activity_prob,
estimated_period, avg_power, avg_snr, confidence
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `acquisition/` | Abstract RF source interface; simulator and file adapters |
| `simulation/` | Generate synthetic IQ with configurable emitters; maintain ground truth |
| `receiver/` | Model limited-bandwidth observation; tuning; dwell timing |
| `dsp/` | FFT, PSD, noise estimation, energy detection |
| `features/` | Extract temporal/spectral features from observations |
| `state/` | Band state tracking, history, SQLite persistence |
| `prediction/` | Temporal activity prediction, periodicity estimation |
| `schedulers/` | All scan strategies: round-robin, random, priority, bandit, adaptive |
| `evaluation/` | Compare scheduler observations against ground truth; compute PD/PFA/delays |
| `visualization/` | Spectrum plots, waterfall, timelines, comparison charts |
| `utils/` | Config loading, logging, timing instrumentation |

## Scheduler Architecture

All schedulers implement a common interface:

```python
class BaseScheduler(ABC):
    def select_band(self, band_states: list[BandState], current_time: float) -> ScanDecision
    def update(self, decision: ScanDecision, observation: BandObservation) -> None
    def reset(self) -> None
```

Scheduler hierarchy:
- **RoundRobin**: Sequential cycling through all bands
- **RandomScan**: Uniform random band selection
- **PriorityScan**: Weighted by activity_prob × recency × uncertainty
- **BanditScheduler**: UCB1 or Thompson Sampling (non-contextual)
- **AdaptiveScheduler**: Contextual bandit with feature vector per band

## Configuration System

YAML-based with Pydantic validation. Three config files:
- `config/default.yaml` — baseline parameters
- `config/experiment.yaml` — scenario-specific overrides
- `config/receiver.yaml` — receiver hardware abstraction

## Testing Strategy

- **Unit tests**: Each module tested in isolation with synthetic inputs
- **Integration tests**: Simulator → Receiver → DSP → State pipeline
- **System tests**: Full scan loop with deterministic scenario
- **Information separation tests**: Verify scheduler cannot access ground truth
- **Deterministic tests**: Fixed seed → expected detections

## Technology Stack

- Python 3.12+
- numpy, scipy (DSP)
- pandas (tabular data)
- scikit-learn (ML predictors)
- matplotlib, plotly (visualization)
- pydantic (data validation)
- streamlit (dashboard)
- pytest, hypothesis (testing)
- SQLite (persistence)
- PyYAML (configuration)
- joblib (model serialization)
