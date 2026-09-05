"""Tests for AWGN noise generator."""

import numpy as np
import pytest

from smartscan.simulation.noise import dbm_to_watts, generate_awgn, power_for_snr, watts_to_dbm


class TestDbmConversions:
    def test_0dbm_is_1mw(self):
        assert dbm_to_watts(0.0) == pytest.approx(0.001)

    def test_30dbm_is_1w(self):
        assert dbm_to_watts(30.0) == pytest.approx(1.0)

    def test_roundtrip(self):
        for dbm in [-120, -100, -80, -60, -30, 0, 10, 30]:
            assert watts_to_dbm(dbm_to_watts(dbm)) == pytest.approx(dbm)

    def test_watts_to_dbm_zero(self):
        assert watts_to_dbm(0.0) == -np.inf


class TestGenerateAWGN:
    def test_correct_length(self):
        noise = generate_awgn(1000)
        assert len(noise) == 1000

    def test_is_complex(self):
        noise = generate_awgn(100)
        assert np.iscomplexobj(noise)

    def test_power_approximately_correct(self):
        rng = np.random.default_rng(42)
        target_dbm = -80.0
        noise = generate_awgn(100_000, power_dbm=target_dbm, rng=rng)
        measured_power = np.mean(np.abs(noise) ** 2)
        measured_dbm = watts_to_dbm(measured_power)
        assert measured_dbm == pytest.approx(target_dbm, abs=1.0)

    def test_reproducible_with_seed(self):
        n1 = generate_awgn(100, rng=np.random.default_rng(99))
        n2 = generate_awgn(100, rng=np.random.default_rng(99))
        np.testing.assert_array_equal(n1, n2)

    def test_different_seeds_differ(self):
        n1 = generate_awgn(100, rng=np.random.default_rng(1))
        n2 = generate_awgn(100, rng=np.random.default_rng(2))
        assert not np.allclose(n1, n2)

    def test_zero_mean(self):
        rng = np.random.default_rng(42)
        noise = generate_awgn(100_000, rng=rng)
        assert np.abs(np.mean(noise.real)) < 0.01
        assert np.abs(np.mean(noise.imag)) < 0.01


class TestPowerForSNR:
    def test_basic(self):
        assert power_for_snr(-100.0, 20.0) == pytest.approx(-80.0)

    def test_zero_snr(self):
        assert power_for_snr(-100.0, 0.0) == pytest.approx(-100.0)
