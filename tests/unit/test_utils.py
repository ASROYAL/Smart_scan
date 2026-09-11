"""Tests for logging and timing utilities."""

import json
import tempfile
from pathlib import Path

from smartscan.utils.logging import ScanLogEntry, ScanLogger
from smartscan.utils.timing import TimingInstrument, TimingStats


class TestScanLogger:
    def test_log_and_count(self):
        logger = ScanLogger()
        assert len(logger) == 0
        entry = ScanLogEntry(
            scan_number=0, timestamp=0.0, band_id=3, center_frequency=100e6,
            dwell_time=0.01, scheduler_score=0.5, detected=True,
            estimated_snr_db=15.0, hit=True, reward=1.0,
            processing_latency_ms=2.5, reason="test",
        )
        logger.log(entry)
        assert len(logger) == 1
        assert logger.entries[0].band_id == 3

    def test_export_csv(self):
        logger = ScanLogger()
        logger.log(ScanLogEntry(
            scan_number=0, timestamp=0.0, band_id=0, center_frequency=100e6,
            dwell_time=0.01, scheduler_score=0.0, detected=False,
            estimated_snr_db=0.0, hit=False, reward=0.0,
            processing_latency_ms=1.0,
        ))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.csv"
            logger.export_csv(path)
            text = path.read_text()
            assert "scan_number" in text
            assert "band_id" in text

    def test_export_json(self):
        logger = ScanLogger()
        logger.log(ScanLogEntry(
            scan_number=0, timestamp=1.0, band_id=2, center_frequency=200e6,
            dwell_time=0.02, scheduler_score=0.7, detected=True,
            estimated_snr_db=20.0, hit=True, reward=1.0,
            processing_latency_ms=3.0,
        ))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.json"
            logger.export_json(path)
            data = json.loads(path.read_text())
            assert len(data) == 1
            assert data[0]["band_id"] == 2

    def test_clear(self):
        logger = ScanLogger()
        logger.log(ScanLogEntry(
            scan_number=0, timestamp=0.0, band_id=0, center_frequency=100e6,
            dwell_time=0.01, scheduler_score=0.0, detected=False,
            estimated_snr_db=0.0, hit=False, reward=0.0,
            processing_latency_ms=1.0,
        ))
        logger.clear()
        assert len(logger) == 0

    def test_export_csv_empty(self):
        """Exporting empty logger should not crash."""
        logger = ScanLogger()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.csv"
            logger.export_csv(path)
            assert not path.exists()


class TestTimingInstrument:
    def test_measure_records_time(self):
        timer = TimingInstrument()
        with timer.measure("test_stage"):
            sum(range(1000))  # a small workload to time
        stats = timer.get_stats("test_stage")
        assert stats.count == 1
        assert stats.total_ms > 0
        assert stats.avg_ms > 0

    def test_multiple_measurements(self):
        timer = TimingInstrument()
        for _ in range(5):
            with timer.measure("loop"):
                pass
        stats = timer.get_stats("loop")
        assert stats.count == 5

    def test_summary(self):
        timer = TimingInstrument()
        with timer.measure("stage_a"):
            pass
        with timer.measure("stage_b"):
            pass
        summary = timer.summary()
        assert "stage_a" in summary
        assert "stage_b" in summary

    def test_reset(self):
        timer = TimingInstrument()
        with timer.measure("x"):
            pass
        timer.reset()
        assert len(timer.all_stats()) == 0


class TestTimingStats:
    def test_avg_zero_count(self):
        s = TimingStats()
        assert s.avg_ms == 0.0

    def test_record(self):
        s = TimingStats()
        s.record(10.0)
        s.record(20.0)
        assert s.count == 2
        assert s.total_ms == 30.0
        assert s.avg_ms == 15.0
        assert s.min_ms == 10.0
        assert s.max_ms == 20.0
