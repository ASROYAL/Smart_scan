"""Tests proving the scheduler CANNOT access ground truth.

This is a core requirement: the scheduler must infer activity purely from
detector output and scan history, never from the simulator's ground truth.
"""

import inspect

import pytest

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import BandObservation, BandState, ScanDecision
from smartscan.schedulers.base import BaseScheduler
from smartscan.schedulers.priority_scan import PriorityScanScheduler
from smartscan.schedulers.random_scan import RandomScanScheduler
from smartscan.schedulers.round_robin import RoundRobinScheduler


ALL_SCHEDULERS = [
    lambda: RoundRobinScheduler(ReceiverConfig()),
    lambda: RandomScanScheduler(ReceiverConfig()),
    lambda: PriorityScanScheduler(ReceiverConfig(), SchedulerConfig()),
]


class TestSchedulerInterfaceIsolation:
    def test_select_band_signature_has_no_ground_truth(self):
        """select_band must only accept band_states and current_time."""
        sig = inspect.signature(BaseScheduler.select_band)
        params = set(sig.parameters.keys())
        assert params == {"self", "band_states", "current_time"}

    def test_update_signature_has_no_ground_truth(self):
        sig = inspect.signature(BaseScheduler.update)
        params = set(sig.parameters.keys())
        assert params == {"self", "decision", "observation"}

    @pytest.mark.parametrize("make_sched", ALL_SCHEDULERS)
    def test_scheduler_has_no_environment_reference(self, make_sched):
        """No scheduler should hold a reference to the RFEnvironment."""
        sched = make_sched()
        for attr_name in vars(sched):
            attr = getattr(sched, attr_name)
            type_name = type(attr).__name__
            assert type_name != "RFEnvironment", (
                f"Scheduler {sched.name} holds an RFEnvironment reference via {attr_name}"
            )

    @pytest.mark.parametrize("make_sched", ALL_SCHEDULERS)
    def test_scheduler_has_no_ground_truth_methods(self, make_sched):
        sched = make_sched()
        methods = [m for m in dir(sched) if not m.startswith("_")]
        forbidden = {"get_ground_truth", "is_any_active_in_band",
                      "get_active_emitters", "get_all_ground_truth"}
        found = set(methods) & forbidden
        assert not found, f"{sched.name} exposes forbidden methods: {found}"


class TestBandStateHasNoGroundTruth:
    def test_band_state_fields_are_observation_derived(self):
        """BandState should only contain observation-derived fields."""
        field_names = set(BandState.model_fields.keys())
        forbidden = {"truly_active", "true_activity", "ground_truth",
                      "emitter_id", "actual_snr", "is_transmitting"}
        found = field_names & forbidden
        assert not found, f"BandState leaks ground truth: {found}"

    def test_band_observation_fields_are_detector_derived(self):
        field_names = set(BandObservation.model_fields.keys())
        forbidden = {"truly_active", "true_activity", "ground_truth", "emitter_id"}
        found = field_names & forbidden
        assert not found, f"BandObservation leaks ground truth: {found}"

    def test_scan_decision_has_no_ground_truth(self):
        field_names = set(ScanDecision.model_fields.keys())
        forbidden = {"truly_active", "ground_truth", "will_be_active", "emitter_id"}
        found = field_names & forbidden
        assert not found, f"ScanDecision leaks ground truth: {found}"


class TestSchedulerCannotSeeGroundTruthAtRuntime:
    def test_scheduler_only_sees_band_states(self):
        """Feed the scheduler band states with FALSE observation data but where
        ground truth would say active. The scheduler must act on the observation
        data, proving it uses only what it's given."""
        cfg = SchedulerConfig(
            exploration_weight=10.0, recency_weight=0.0, uncertainty_weight=0.0,
            starvation_threshold=1e9,
        )
        sched = PriorityScanScheduler(ReceiverConfig(), cfg)

        states = [
            BandState(band_id=0, freq_start=0, freq_end=20e6,
                       rolling_activity_prob=0.9, observation_count=10,
                       last_scan_time=0.0, confidence=0.5),
            BandState(band_id=1, freq_start=20e6, freq_end=40e6,
                       rolling_activity_prob=0.1, observation_count=10,
                       last_scan_time=0.0, confidence=0.5),
        ]
        # Scheduler should pick band 0 (higher observed activity prob),
        # regardless of any hidden ground truth.
        decision = sched.select_band(states, current_time=1.0)
        assert decision.band_id == 0
