"""Typed configuration for federated coordination, partitioning, and local training."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
import math
from typing import Any


@dataclass(init=False)
class FederatedCFG:
    """Resolve Flower/YAML keys into validated federated pipeline settings.

    Runtime aggregation callbacks are supplied to strategy_kwargs; they are not
    serialized into the YAML or Flower's scalar run configuration.
    """

    resume_experiment_id: str = ""
    num_server_rounds: int = 10
    local_epochs: int = 2
    fraction_train: float = 1.0
    fraction_evaluate: float = 1.0
    min_nodes: int = 3
    min_train_nodes: int = 1
    min_evaluate_nodes: int = 1
    proximal_mu: float = 0.0
    partitioning: str = "dirichlet"
    dirichlet_alpha: float = 0.5
    train_fraction: float = 0.3
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    batch_size: int = 32
    learning_rate: float = 0.0001
    num_workers: int = 2
    seed: int = 42

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("FederatedCFG expects a mapping of federated settings.")
        supported = {item.name.replace("_", "-") for item in fields(self)}
        unknown = data.keys() - supported
        if unknown:
            raise ValueError(f"Unknown federated settings: {sorted(unknown)}")
        for item in fields(self):
            key = item.name.replace("_", "-")
            setattr(self, item.name, deepcopy(data.get(key, item.default)))
        self.validate()

    def validate(self) -> None:
        """Reject invalid execution settings before starting Flower processes."""
        if not isinstance(self.resume_experiment_id, str):
            raise ValueError("resume-experiment-id must be a string.")
        for name in ("num_server_rounds", "local_epochs", "min_nodes", "min_train_nodes",
                     "min_evaluate_nodes", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("num_workers", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32.")
        for name in ("fraction_train", "fraction_evaluate", "proximal_mu", "dirichlet_alpha",
                     "train_fraction", "val_fraction", "test_fraction", "learning_rate"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number.")
            setattr(self, name, float(value))
        if not 0 < self.fraction_train <= 1 or not 0 <= self.fraction_evaluate <= 1:
            raise ValueError("Participation fractions must be within [0, 1], with training > 0.")
        if self.min_train_nodes > self.min_nodes or self.min_evaluate_nodes > self.min_nodes:
            raise ValueError("Minimum training/evaluation nodes cannot exceed min-nodes.")
        if self.proximal_mu < 0 or self.dirichlet_alpha <= 0 or self.learning_rate <= 0:
            raise ValueError("proximal-mu must be non-negative; alpha and learning rate must be positive.")
        if self.partitioning not in ("iid", "dirichlet"):
            raise ValueError("partitioning must be 'iid' or 'dirichlet'.")
        if any(not 0 < value < 1 for value in self.split_ratios):
            raise ValueError("Dataset split fractions must be between 0 and 1.")
        total = sum(self.split_ratios)
        if total > 1 and not math.isclose(total, 1, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Dataset split fractions cannot sum to more than 1.")

    @property
    def split_ratios(self) -> tuple[float, float, float]:
        return self.train_fraction, self.val_fraction, self.test_fraction

    @property
    def min_available_nodes(self) -> int:
        return self.min_nodes

    def to_dict(self) -> dict[str, int | float | str]:
        """Serialize all resolved settings using Flower's hyphenated keys."""
        return {item.name.replace("_", "-"): getattr(self, item.name) for item in fields(self)}

    def strategy_kwargs(
        self,
        train_metrics_aggr_fn: Callable,
        evaluate_metrics_aggr_fn: Callable,
    ) -> dict[str, Any]:
        """Build the shared FedAvg/FedProx arguments with runtime callbacks."""
        self.validate()
        return {
            "fraction_train": self.fraction_train,
            "fraction_evaluate": self.fraction_evaluate,
            "min_train_nodes": self.min_train_nodes,
            "min_evaluate_nodes": self.min_evaluate_nodes,
            "min_available_nodes": self.min_available_nodes,
            "train_metrics_aggr_fn": train_metrics_aggr_fn,
            "evaluate_metrics_aggr_fn": evaluate_metrics_aggr_fn,
        }
