"""Factory for constructing schedulers by type."""

from __future__ import annotations

from smartscan.core.config import ReceiverConfig, SchedulerConfig
from smartscan.core.models import SchedulerType
from smartscan.schedulers.adaptive import AdaptiveScheduler
from smartscan.schedulers.bandit import ThompsonSamplingScheduler, UCB1BanditScheduler
from smartscan.schedulers.base import BaseScheduler
from smartscan.schedulers.priority_scan import PriorityScanScheduler
from smartscan.schedulers.qlearning import QLearningScheduler
from smartscan.schedulers.random_scan import RandomScanScheduler
from smartscan.schedulers.round_robin import RoundRobinScheduler


def create_scheduler(
    scheduler_type: SchedulerType | str,
    receiver_config: ReceiverConfig,
    scheduler_config: SchedulerConfig,
    num_bands: int,
    seed: int = 0,
) -> BaseScheduler:
    """Construct a scheduler of the requested type."""
    if isinstance(scheduler_type, str):
        scheduler_type = SchedulerType(scheduler_type)

    match scheduler_type:
        case SchedulerType.ROUND_ROBIN:
            return RoundRobinScheduler(receiver_config)
        case SchedulerType.RANDOM:
            return RandomScanScheduler(receiver_config, seed=seed)
        case SchedulerType.PRIORITY:
            return PriorityScanScheduler(receiver_config, scheduler_config)
        case SchedulerType.BANDIT_UCB:
            return UCB1BanditScheduler(receiver_config, scheduler_config, num_bands)
        case SchedulerType.BANDIT_THOMPSON:
            return ThompsonSamplingScheduler(receiver_config, num_bands, seed=seed)
        case SchedulerType.ADAPTIVE:
            return AdaptiveScheduler(receiver_config, scheduler_config, num_bands)
        case SchedulerType.Q_LEARNING:
            return QLearningScheduler(receiver_config, scheduler_config, num_bands, seed=seed)
        case _:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


ALL_SCHEDULER_TYPES = [
    SchedulerType.ROUND_ROBIN,
    SchedulerType.RANDOM,
    SchedulerType.PRIORITY,
    SchedulerType.BANDIT_UCB,
    SchedulerType.BANDIT_THOMPSON,
    SchedulerType.ADAPTIVE,
    SchedulerType.Q_LEARNING,
]
