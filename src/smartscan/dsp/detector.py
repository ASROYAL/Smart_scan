"""Energy-based activity detector.

Consumes IQ samples, computes PSD, estimates the noise floor, sets an adaptive
threshold, and declares activity where the PSD exceeds it. Uses NO ground truth.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta, DetectionResult, DetectorType
from smartscan.dsp.base import BaseDetector
from smartscan.dsp.noise_floor import compute_threshold, estimate_noise_floor
from smartscan.dsp.psd import compute_psd


def _np_window(name, n: int) -> np.ndarray:
    key = getattr(name, "value", name)
    if key == "hamming":
        return np.hamming(n)
    if key == "blackman":
        return np.blackman(n)
    return np.hanning(n)


class EnergyDetector(BaseDetector):
    """Detects signal activity by comparing PSD bins to an adaptive threshold."""

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config

    def detect(
        self,
        samples: np.ndarray,
        meta: AcquisitionMeta,
    ) -> DetectionResult:
        cfg = self._config

        freqs, psd_db = compute_psd(
            samples=samples,
            sample_rate=meta.sample_rate,
            method=cfg.psd_method,
            window=cfg.window,
            fft_size=cfg.fft_size,
            nperseg=cfg.welch_nperseg,
            overlap=cfg.welch_overlap,
            center_frequency=meta.center_frequency,
        )

        noise_floor = estimate_noise_floor(
            psd_db, method=cfg.noise_method, percentile=cfg.percentile,
        )
        threshold = compute_threshold(noise_floor, cfg.threshold_margin_db)

        # Restrict analysis to the observation bandwidth (guard against FFT edges)
        half_bw = meta.bandwidth / 2
        in_band = np.abs(freqs - meta.center_frequency) <= half_bw
        band_psd = psd_db[in_band]
        band_freqs = freqs[in_band]

        if len(band_psd) == 0:
            band_psd = psd_db
            band_freqs = freqs

        above = band_psd > threshold
        num_detections = int(np.sum(above))
        detected = num_detections > 0

        peak_power = float(np.max(band_psd))
        avg_power = float(np.mean(band_psd))
        estimated_snr = peak_power - noise_floor

        # Confidence: how far the peak is above threshold, normalized
        if detected:
            excess = peak_power - threshold
            confidence = float(np.clip(excess / (cfg.threshold_margin_db + 1e-9), 0.0, 1.0))
            confidence = max(confidence, 0.5)  # detected implies at least moderate confidence
        else:
            # How close was the peak to the threshold
            deficit = threshold - peak_power
            confidence = float(np.clip(1.0 - deficit / (cfg.threshold_margin_db + 1e-9), 0.0, 0.5))

        # Detected frequency region: extent of above-threshold bins
        if detected:
            det_freqs = band_freqs[above]
            freq_start = float(np.min(det_freqs))
            freq_end = float(np.max(det_freqs))
        else:
            freq_start = meta.center_frequency - half_bw
            freq_end = meta.center_frequency + half_bw

        return DetectionResult(
            detected=detected,
            confidence=confidence,
            peak_power_db=peak_power,
            avg_power_db=avg_power,
            noise_floor_db=noise_floor,
            estimated_snr_db=estimated_snr,
            freq_start=freq_start,
            freq_end=freq_end,
            timestamp=meta.timestamp,
            num_detections=num_detections,
        )


class MatchedFilterDetector(BaseDetector):
    """Coherent single-FFT peak-bin detector.

    For an unknown-frequency narrowband emitter (tone / CW radar), the optimal
    receiver is a filter matched to a complex exponential — i.e. the strongest
    FFT bin of a single coherent transform. This keeps the full coherent
    processing gain (10·log10(N) dB) rather than averaging it away, so it
    detects weaker narrowband signals than an averaged energy detector.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config

    def detect(self, samples: np.ndarray, meta: AcquisitionMeta) -> DetectionResult:
        cfg = self._config
        n = len(samples)
        w = _np_window(cfg.window, n)
        w = w / np.sqrt(np.mean(w ** 2))            # unit-power window
        spectrum = np.fft.fftshift(np.fft.fft(samples * w) / n)
        power = np.abs(spectrum) ** 2
        power_db = 10 * np.log10(power + 1e-30)

        noise_floor = estimate_noise_floor(power_db, method=cfg.noise_method,
                                           percentile=cfg.percentile)
        # CFAR correction: we test the MAX of N bins, whose expected value sits
        # ~10*log10(ln N) dB above the median noise bin even for pure noise, so
        # the peak threshold must include that term to keep the false-alarm rate low.
        cfar = 10.0 * np.log10(max(np.log(max(n, 2)), 1.0))
        threshold = compute_threshold(noise_floor, cfg.threshold_margin_db) + cfar

        peak = float(np.max(power_db))
        avg = float(np.mean(power_db))
        detected = peak > threshold
        snr = peak - noise_floor

        if detected:
            excess = peak - threshold
            confidence = max(float(np.clip(excess / (cfg.threshold_margin_db + 1e-9), 0, 1)), 0.5)
        else:
            deficit = threshold - peak
            confidence = float(np.clip(1.0 - deficit / (cfg.threshold_margin_db + 1e-9), 0, 0.5))

        half_bw = meta.bandwidth / 2
        return DetectionResult(
            detected=detected, confidence=confidence, peak_power_db=peak,
            avg_power_db=avg, noise_floor_db=noise_floor, estimated_snr_db=snr,
            freq_start=meta.center_frequency - half_bw,
            freq_end=meta.center_frequency + half_bw,
            timestamp=meta.timestamp, num_detections=int(detected),
        )


class CyclostationaryDetector(BaseDetector):
    """Cyclic-autocorrelation feature detector.

    Man-made signals are correlated across time (cyclostationary); white noise is
    not. This measures the normalised magnitude of the cyclic autocorrelation
    R(τ)=E[x[n+τ]·x*[n]] over a set of lags τ>0 and declares a signal when the
    peak exceeds a threshold. It catches structured signals that a pure-energy
    test can miss, and rejects white noise (whose autocorrelation → 0 for τ≠0).
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config

    def detect(self, samples: np.ndarray, meta: AcquisitionMeta) -> DetectionResult:
        cfg = self._config
        x = np.asarray(samples)
        n = len(x)
        r0 = float(np.mean(np.abs(x) ** 2)) + 1e-30

        max_lag = min(cfg.autocorr_lags, n - 1)
        feature = 0.0
        for tau in range(1, max_lag + 1):
            r_tau = np.mean(x[tau:] * np.conj(x[:-tau]))
            mag = float(np.abs(r_tau)) / r0
            feature = max(feature, mag)

        detected = feature > cfg.autocorr_threshold
        confidence = float(np.clip(feature / (cfg.autocorr_threshold * 2 + 1e-9), 0, 1))
        if detected:
            confidence = max(confidence, 0.5)

        # Report power/SNR/frequency from the PSD for consistency with other detectors
        _freqs, psd_db = compute_psd(
            samples=x, sample_rate=meta.sample_rate, method=cfg.psd_method,
            window=cfg.window, fft_size=cfg.fft_size, nperseg=cfg.welch_nperseg,
            overlap=cfg.welch_overlap, center_frequency=meta.center_frequency,
        )
        noise_floor = estimate_noise_floor(psd_db, method=cfg.noise_method,
                                           percentile=cfg.percentile)
        peak = float(np.max(psd_db))
        half_bw = meta.bandwidth / 2
        return DetectionResult(
            detected=detected, confidence=confidence, peak_power_db=peak,
            avg_power_db=float(np.mean(psd_db)), noise_floor_db=noise_floor,
            estimated_snr_db=peak - noise_floor,
            freq_start=meta.center_frequency - half_bw,
            freq_end=meta.center_frequency + half_bw,
            timestamp=meta.timestamp, num_detections=int(detected),
        )


def create_detector(config: DetectorConfig) -> BaseDetector:
    """Construct the detector selected by ``config.detector_type``."""
    dt = config.detector_type
    if isinstance(dt, str):
        dt = DetectorType(dt)
    match dt:
        case DetectorType.ENERGY:
            return EnergyDetector(config)
        case DetectorType.MATCHED_FILTER:
            return MatchedFilterDetector(config)
        case DetectorType.CYCLOSTATIONARY:
            return CyclostationaryDetector(config)
        case _:
            raise ValueError(f"Unknown detector type: {dt}")
