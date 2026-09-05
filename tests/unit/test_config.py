"""Tests for the configuration system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from smartscan.core.config import (
    SmartScanConfig,
    load_config,
    load_config_with_overrides,
    _deep_merge,
)


class TestDefaultConfig:
    def test_defaults_are_valid(self):
        cfg = SmartScanConfig()
        assert cfg.environment.total_bandwidth == 1_000e6
        assert cfg.receiver.instantaneous_bandwidth == 20e6
        assert cfg.simulation.seed == 42

    def test_receiver_bw_less_than_total(self):
        cfg = SmartScanConfig()
        assert cfg.receiver.instantaneous_bandwidth < cfg.environment.total_bandwidth


class TestLoadConfig:
    def test_load_default_yaml(self):
        cfg = load_config("config/default.yaml")
        assert cfg.environment.total_bandwidth == 1e9
        assert cfg.receiver.dwell_time == 0.01
        assert cfg.detector.window.value == "hann"
        assert cfg.simulation.seed == 42

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_partial_yaml_uses_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"simulation": {"seed": 123}}, f)
            f.flush()
            cfg = load_config(f.name)
        assert cfg.simulation.seed == 123
        assert cfg.environment.total_bandwidth == 1_000e6  # default

    def test_empty_yaml_uses_all_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            cfg = load_config(f.name)
        assert cfg.simulation.seed == 42


class TestLoadConfigWithOverrides:
    def test_override_merges(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as override:
            yaml.dump({"simulation": {"seed": 999, "duration": 120.0}}, override)
            override.flush()
            cfg = load_config_with_overrides("config/default.yaml", override.name)
        assert cfg.simulation.seed == 999
        assert cfg.simulation.duration == 120.0
        assert cfg.environment.total_bandwidth == 1e9  # unchanged

    def test_kwargs_override(self):
        cfg = load_config_with_overrides(
            "config/default.yaml",
            simulation={"seed": 777},
        )
        assert cfg.simulation.seed == 777


class TestDeepMerge:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3})
        assert base == {"a": 1, "b": 3}

    def test_nested_override(self):
        base = {"x": {"y": 1, "z": 2}}
        _deep_merge(base, {"x": {"z": 99}})
        assert base == {"x": {"y": 1, "z": 99}}

    def test_add_new_key(self):
        base = {"a": 1}
        _deep_merge(base, {"b": 2})
        assert base == {"a": 1, "b": 2}
