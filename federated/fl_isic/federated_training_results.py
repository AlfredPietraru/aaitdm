"""Mutable results collected throughout a federated training process."""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from .federated_cfg import FederatedCFG

if TYPE_CHECKING:
    import torch
    from flwr.serverapp.strategy import Result


MetricValue = int | float | list[int] | list[float]
RoundHistory = dict[int, dict[str, MetricValue]]
ClientHistory = dict[int, RoundHistory]


class FederatedTrainingStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class FederatedTrainingResults:
    """Keep configuration, global and client histories, final weights, and outcome.

    Aggregation callbacks update this object as rounds execute, preserving all
    metrics received before a failure or user interruption. Final weights are
    available only when Flower returns its final Result.
    """

    configuration: FederatedCFG
    train_metrics_clientapp: RoundHistory = field(default_factory=dict)
    evaluate_metrics_clientapp: RoundHistory = field(default_factory=dict)
    evaluate_metrics_serverapp: RoundHistory = field(default_factory=dict)
    per_client: dict[str, ClientHistory] = field(default_factory=lambda: {"train": {}, "val": {}})
    global_metrics: dict[str, RoundHistory] = field(default_factory=lambda: {"train": {}, "val": {}, "test": {}})
    class_distributions: dict = field(default_factory=dict)
    final_state_dict: "dict[str, torch.Tensor] | None" = field(default=None, repr=False)
    status: FederatedTrainingStatus = FederatedTrainingStatus.RUNNING
    failure_reason: str | None = None
    stop_reason: str | None = None
    early_stopping: dict = field(default_factory=dict)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None

    def __post_init__(self) -> None:
        # Keep the actual settings used by this run independent of later edits.
        self.configuration = deepcopy(self.configuration)

    @property
    def strategy(self) -> str:
        return "FedProx" if self.configuration.proximal_mu > 0 else "FedAvg"

    @property
    def rounds_completed(self) -> int:
        return len(self.train_metrics_clientapp)

    @property
    def elapsed_seconds(self) -> float:
        return ((self.ended_at or datetime.now(timezone.utc)) - self.started_at).total_seconds()

    def add_client_metrics(
        self, phase: str, client_id: int, server_round: int, metrics: dict[str, MetricValue],
    ) -> None:
        """Store an independent snapshot of a client's reply."""
        if phase not in self.per_client:
            raise ValueError(f"Unknown phase: {phase!r}")
        self.per_client[phase].setdefault(client_id, {})[server_round] = deepcopy(metrics)

    def add_round_metrics(
        self, phase: str, server_round: int, metrics: dict[str, MetricValue],
    ) -> None:
        """Retain aggregated metrics immediately, without waiting for run completion."""
        histories = {"train": self.train_metrics_clientapp, "val": self.evaluate_metrics_clientapp,
                     "server": self.evaluate_metrics_serverapp}
        if phase not in histories:
            raise ValueError(f"Unknown phase: {phase!r}")
        histories[phase][server_round] = deepcopy(metrics)

    def capture_result(self, result: "Result") -> None:
        """Capture Flower's complete returned histories and final CPU model weights."""
        for name in ("train_metrics_clientapp", "evaluate_metrics_clientapp", "evaluate_metrics_serverapp"):
            getattr(self, name).update({int(round_number): deepcopy(dict(metrics))
                                       for round_number, metrics in getattr(result, name).items()})
        self.final_state_dict = {
            name: tensor.detach().cpu().clone()
            for name, tensor in result.arrays.to_torch_state_dict().items()
        }

    def finish(self, status: FederatedTrainingStatus, error: BaseException | None = None) -> None:
        """Record the terminal outcome and timing."""
        if status == FederatedTrainingStatus.RUNNING:
            raise ValueError("finish() requires a terminal status.")
        self.status = status
        self.failure_reason = f"{type(error).__name__}: {error}" if error is not None else None
        self.ended_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Serialize metadata and histories; model weights are saved separately."""
        return {
            "stop_reason": self.stop_reason,
            "early_stopping": deepcopy(self.early_stopping),
            "global_metrics": deepcopy(self.global_metrics),
            "class_distributions": deepcopy(self.class_distributions),
            "strategy": self.strategy,
            "run_config": self.configuration.to_dict(),
            "status": self.status.value,
            "failure_reason": self.failure_reason,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "elapsed_seconds": self.elapsed_seconds,
            "rounds_completed": self.rounds_completed,
            "train_metrics_clientapp": deepcopy(self.train_metrics_clientapp),
            "evaluate_metrics_clientapp": deepcopy(self.evaluate_metrics_clientapp),
            "evaluate_metrics_serverapp": deepcopy(self.evaluate_metrics_serverapp),
            "per_client": deepcopy(self.per_client),
        }
