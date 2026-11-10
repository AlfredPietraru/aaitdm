"""Dictionary-based training configuration and transform pipelines."""

from copy import deepcopy
from typing import Any

import monai.transforms as monai_transforms
from monai.transforms.compose import Compose
from monai.transforms.transform import Transform
from monai.transforms.utility.array import Lambda
import numpy as np
import torch
from torch import nn


class TrainingCFG:
    """Resolve dictionary settings into PyTorch training components.

    Component names are case-sensitive PyTorch class names. Scheduler patience
    and early_stopping_patience are integers; scheduler factors are floats.
    Use batch_size when constructing loaders and scheduler_type=None to disable
    scheduling. Loss functions must accept multiclass logits and integer class labels and return a mean loss.
    Transform names and parameters come from monai.transforms. PIL images are
    converted to NumPy arrays before the configured steps. Each list defines a
    full pipeline; defaults convert RGB HWC images to CHW float tensors in [0, 1].
    """

    def __init__(self, data: dict[str, Any]) -> None:
        assert isinstance(data, dict), "TrainingCFG expects a dictionary."
        self.data = deepcopy(data)
        allowed = {
            "optimizer", "optimizer_kwargs", "batch_size", "loss_function",
            "loss_function_kwargs", "epochs", "scheduler_type", "scheduler_kwargs",
            "early_stopping_patience", "early_stopping_tolerance",
            "train_transforms", "val_transforms", "experiment_name", "run_name",
        }
        unknown = data.keys() - allowed
        assert not unknown, f"Unknown training settings: {sorted(unknown)}"

        self.experiment_name = data.get("experiment_name", "isic2019-baseline")
        self.run_name = data.get("run_name", "baseline")
        assert isinstance(self.experiment_name, str) and self.experiment_name
        assert isinstance(self.run_name, str) and self.run_name

        self.train_transforms = self._build_transforms(
            data.get("train_transforms", self._default_transforms())
        )
        self.val_transforms = self._build_transforms(
            data.get("val_transforms", self._default_transforms())
        )
        self.epochs = self._positive_int(data, "epochs", 10)
        self.batch_size = self._positive_int(data, "batch_size", 128)
        self.early_stopping_patience = self._positive_int(
            data, "early_stopping_patience", 7
        )
        self.early_stopping_tolerance = data.get("early_stopping_tolerance", 0.1)
        self.optimizer = self._resolve(
            torch.optim, data.get("optimizer", "Adam"), torch.optim.Optimizer
        )
        self.optimizer_kwargs = {"lr": 1e-3, **self._kwargs(data, "optimizer_kwargs")}
        loss_type = self._resolve(nn, data.get("loss_function", "CrossEntropyLoss"), nn.Module)
        self.loss_function = loss_type(**self._kwargs(data, "loss_function_kwargs"))

        scheduler_name = data.get("scheduler_type", "ReduceLROnPlateau")
        self.scheduler_type = (
            None if scheduler_name is None else self._resolve(
                torch.optim.lr_scheduler,
                scheduler_name,
                (torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau),
            )
        )
        defaults = (
            {"mode": "min", "patience": 3, "factor": 0.1}
            if self.scheduler_type is torch.optim.lr_scheduler.ReduceLROnPlateau else {}
        )
        self.scheduler_kwargs = {**defaults, **self._kwargs(data, "scheduler_kwargs")}
        if "patience" in self.scheduler_kwargs:
            patience = self.scheduler_kwargs["patience"]
            assert type(patience) is int and patience >= 0, (
                "Scheduler patience must be a non-negative integer."
            )

    @staticmethod
    def _default_transforms() -> list[dict[str, Any]]:
        return [
            {"EnsureChannelFirst": {"channel_dim": -1}},
            {"Resize": {"spatial_size": [224, 224]}},
            {"ScaleIntensityRange": {"a_min": 0, "a_max": 255, "b_min": 0, "b_max": 1}},
            {"ToTensor": {"track_meta": False}},
        ]

    @staticmethod
    def _build_transforms(pipeline: list[dict[str, Any]]) -> Compose:
        """Instantiate MONAI transforms in the exact configured order."""
        assert isinstance(pipeline, list), "Transform pipelines must be lists."
        steps: list[Transform] = [Lambda(func=np.array)]
        for entry in pipeline:
            assert isinstance(entry, dict) and len(entry) == 1, (
                "Each transform entry must contain one transform name and its parameters."
            )
            name, parameters = next(iter(entry.items()))
            assert isinstance(name, str), "Transform names must be strings."
            transform_type = getattr(monai_transforms, name, None)
            assert isinstance(transform_type, type), f"Unknown MONAI transform: {name!r}"
            parameters = TrainingCFG._kwargs({"parameters": parameters}, "parameters")
            steps.append(transform_type(**parameters))
        return Compose(steps)

    @staticmethod
    def _positive_int(data: dict[str, Any], key: str, default: int) -> int:
        value = data.get(key, default)
        assert type(value) is int and value >= 1, f"{key} must be a positive integer."
        return value

    @staticmethod
    def _kwargs(data: dict[str, Any], key: str) -> dict[str, Any]:
        value = data.get(key, {})
        assert isinstance(value, dict) and all(isinstance(k, str) for k in value), (
            f"{key} must be a dictionary with string keys."
        )
        return deepcopy(value)

    @staticmethod
    def _resolve(namespace: Any, name: str, base: type | tuple[type, ...]) -> type:
        component = getattr(namespace, name, None) if isinstance(name, str) else None
        assert isinstance(component, type) and issubclass(component, base), (
            f"Unknown or unsupported {namespace.__name__} component: {name!r}"
        )
        return component
