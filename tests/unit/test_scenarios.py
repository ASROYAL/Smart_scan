"""Tests for scenario definitions."""

import pytest

from smartscan.simulation.scenarios import ALL_SCENARIOS, Scenario, get_scenario


class TestScenarios:
    @pytest.mark.parametrize("name", ALL_SCENARIOS)
    def test_scenario_constructs(self, name):
        scenario = get_scenario(name, seed=42, num_bands=50,
                                 total_bw=1000e6, center=500e6)
        assert isinstance(scenario, Scenario)
        assert scenario.name == name
        assert len(scenario.emitters) > 0
        assert scenario.duration > 0

    @pytest.mark.parametrize("name", ALL_SCENARIOS)
    def test_emitters_within_spectrum(self, name):
        total_bw = 1000e6
        center = 500e6
        scenario = get_scenario(name, seed=42, num_bands=50,
                                 total_bw=total_bw, center=center)
        low = center - total_bw / 2
        high = center + total_bw / 2
        for em in scenario.emitters:
            assert low <= em.center_frequency <= high
            if em.hop_frequencies:
                for f in em.hop_frequencies:
                    assert low <= f <= high

    @pytest.mark.parametrize("name", ALL_SCENARIOS)
    def test_reproducible(self, name):
        s1 = get_scenario(name, seed=42, num_bands=50, total_bw=1000e6, center=500e6)
        s2 = get_scenario(name, seed=42, num_bands=50, total_bw=1000e6, center=500e6)
        freqs1 = [e.center_frequency for e in s1.emitters]
        freqs2 = [e.center_frequency for e in s2.emitters]
        assert freqs1 == freqs2

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario("nonexistent")

    def test_sparse_fewer_than_dense(self):
        sparse = get_scenario("sparse", seed=42, num_bands=50,
                               total_bw=1000e6, center=500e6)
        dense = get_scenario("dense", seed=42, num_bands=50,
                             total_bw=1000e6, center=500e6)
        assert len(sparse.emitters) < len(dense.emitters)

    def test_changing_has_time_windows(self):
        scenario = get_scenario("changing", seed=42, num_bands=50,
                                 total_bw=1000e6, center=500e6)
        # Some emitters end early, some start late
        has_end = any(e.end_time is not None for e in scenario.emitters)
        has_late_start = any(e.start_time > 0 for e in scenario.emitters)
        assert has_end
        assert has_late_start
