"""Energy-based activity detector.

Consumes IQ samples, computes PSD, estimates the noise floor, sets an adaptive
threshold, and declares activity where the PSD exceeds it. Uses NO ground truth.
"""

from __future__ import annotations

import numpy as np

from smartscan.core.config import DetectorConfig
from smartscan.core.models import AcquisitionMeta, DetectionResult
from smartscan.dsp.base import BaseDetector
from smartscan.dsp.noise_floor import compute_threshold, estimate_noise_floor
from smartscan.dsp.psd import compute_psd


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
