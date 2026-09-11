"""Tests for the receive-only hardware streaming boundary."""

import numpy as np
import pytest

from smartscan.acquisition.streaming_source import StreamingRFSource


def test_streaming_source_reads_exact_samples_and_channelizes():
    tuned = []
    source = StreamingRFSource(1000, 100e6, tune_handler=tuned.append)
    source.tune(110e6)
    source.push(np.ones(20, dtype=np.complex64))
    samples, meta = source.read_samples(110e6, 1e6, 16)
    assert len(samples) == 16
    assert meta.center_frequency == 110e6
    assert source.channel_power(samples, 4).shape == (4,)
    assert tuned == [110e6]


def test_streaming_source_reports_underflow_and_drops_old_data():
    source = StreamingRFSource(10, 100e6, max_buffer_seconds=1, read_timeout=0.001)
    source.push(np.ones(20))
    assert source.dropped_samples == 10
    source.read_samples(100e6, 1e6, 10)
    with pytest.raises(TimeoutError, match="underflow"):
        source.read_samples(100e6, 1e6, 1)
