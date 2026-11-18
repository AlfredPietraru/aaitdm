"""Model, data partitioning, and local training shared by the ServerApp and ClientApp.

The shared net_training package lives in the project root, which must be on
PYTHONPATH; run_federated.sh sets it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch
import yaml
from monai.data.utils import worker_init_fn
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

import isic_dataloaders
from isic_dataloaders import DATA_DIR, CLASS_NAMES, ISICDataLoaders, ISICDataset
from net_training.neural_network import (
    TrainingOneEpochResults, TrainingResults, train_one_epoch, validate_one_epoch,
)
from net_training.training_configuration import TrainingCFG
if TYPE_CHECKING:
    from fl_isic.federated_cfg import FederatedCFG

PROJECT_DIR = Path(isic_dataloaders.__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "federated" / "outputs"
SPLITS = ("train", "val", "test")
# Kept finite on every client so FedAvg can weight-average them.
CLIENT_METRICS = ("loss", "accuracy", "macro_f1")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(pretrained: bool = False) -> nn.Module:
    """ResNet-18 with an eight-class head, as in the centralized baseline."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def load_training_cfg(federated_cfg: FederatedCFG) -> TrainingCFG:
    """Baseline YAML settings with the federated run configuration applied on top."""
    with (PROJECT_DIR / "app-state" / "training_configuration.yaml").open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    data.update(
        batch_size=federated_cfg.batch_size,
        num_workers=federated_cfg.num_workers,
        seed=federated_cfg.seed,
        optimizer_kwargs={
            **data.get("optimizer_kwargs", {}),
            "lr": federated_cfg.learning_rate,
        },
    )
    return TrainingCFG(data)


def load_split(cfg: TrainingCFG, split_ratios: tuple[float, float, float]) -> dict[str, ISICDataset]:
    """Return the baseline lesion-separated split, cached as JSON.

    Indexing ISIC checks every image file, which takes about a minute on a
    Windows drive mounted in WSL. The server builds this cache before assigning
    data to clients; atomic replacement keeps concurrent launches consistent.
    """
    ratios = "-".join(f"{ratio:g}" for ratio in split_ratios)
    cache = OUTPUT_DIR / "cache" / f"split-data-seed{cfg.seed}-{ratios}.json"
    if not cache.is_file():
        datasets = ISICDataLoaders(
            image_dir=DATA_DIR / "ISIC_2019_Training_Input",
            labels_csv=DATA_DIR / "ISIC_2019_Training_GroundTruth.csv",
            metadata_csv=DATA_DIR / "ISIC_2019_Training_Metadata.csv",
        ).get_datasets_baseline_experiment(train_cfg=cfg, split_ratios=split_ratios)
        payload = {
            split: {
                "samples": [[str(path), label] for path, label in dataset.samples],
                "groups": list(dataset.groups),
            }
            for split, dataset in zip(SPLITS, datasets)
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_suffix(f".{os.getpid()}.tmp")
        partial.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(partial, cache)

    payload = json.loads(cache.read_text(encoding="utf-8"))
    transforms = {"train": cfg.train_transforms, "val": cfg.val_transforms, "test": cfg.val_transforms}
    return {
        split: ISICDataset(
            [(Path(path), label) for path, label in payload[split]["samples"]],
            payload[split]["groups"],
            transform=transforms[split],
        )
        for split in SPLITS
    }


def _split_by_proportions(
    dataset: ISICDataset, proportions: np.ndarray, rng: np.random.Generator,
) -> list[list[int]]:
    """Assign whole lesion groups to clients; proportions[k, c] is client k's share of class c."""
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(dataset.groups):
        members[group].append(index)
    by_class: dict[int, list[str]] = defaultdict(list)
    for group in sorted(members):
        counts = Counter(dataset.samples[i][1] for i in members[group])
        by_class[min(counts, key=lambda label: (-counts[label], label))].append(group)

    shares: list[list[int]] = [[] for _ in range(proportions.shape[0])]
    for label in range(len(CLASS_NAMES)):
        groups = by_class[label]
        rng.shuffle(groups)
        cuts = (np.cumsum(proportions[:-1, label]) * len(groups)).astype(int)
        for client, chunk in enumerate(np.split(np.arange(len(groups)), cuts)):
            shares[client].extend(i for g in chunk for i in members[groups[g]])
    return [sorted(share) for share in shares]


def partition(
    datasets: Mapping[str, ISICDataset],
    num_partitions: int,
    partitioning: str,
    alpha: float,
    seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Deterministically split train and validation indices across clients.

    "iid" gives every client the same class mix. "dirichlet" draws each class's
    split across clients from Dir(alpha): smaller alpha means more label skew.
    A client keeps the same class mix for its training and validation data.
    The server computes this once and persists the assignments.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(CLASS_NAMES)
    for _ in range(100):
        if partitioning == "iid":
            proportions = np.full((num_partitions, num_classes), 1 / num_partitions)
        elif partitioning == "dirichlet":
            proportions = rng.dirichlet(np.full(num_partitions, alpha), size=num_classes).T
        else:
            raise ValueError(f"Unknown partitioning {partitioning!r}; use 'iid' or 'dirichlet'.")
        train = _split_by_proportions(datasets["train"], proportions, rng)
        val = _split_by_proportions(datasets["val"], proportions, rng)
        if min(map(len, train)) >= 32 and min(map(len, val)) >= 8:
            return train, val
    raise RuntimeError("Could not give every client data; increase dirichlet-alpha or the data fractions.")


def dataset_from_references(samples, transform):
    """Resolve server-assigned image IDs against this machine's dataset root."""
    root = DATA_DIR / "ISIC_2019_Training_Input"
    # Match ISICDataLoaders: archives may contain another directory of this name.
    nested = root / "ISIC_2019_Training_Input"
    if nested.is_dir():
        root = nested
    paths = [(root / sample.sample_id, sample.label) for sample in samples]
    missing = next((path for path, _ in paths if not path.is_file()), None)
    if missing is not None:
        raise FileNotFoundError(f"Assigned image is missing locally: {missing}")
    return ISICDataset(paths, [sample.group for sample in samples], transform=transform)


def class_counts(dataset: ISICDataset) -> dict[str, int]:
    counts = Counter(label for _, label in dataset.samples)
    return {name: counts.get(label, 0) for label, name in enumerate(CLASS_NAMES)}


def make_loader(dataset: ISICDataset, cfg: TrainingCFG, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(seed),
    )


class ProximalLoss(nn.Module):
    """FedProx objective: base loss + (mu / 2) * ||w - w_global||^2; mu = 0 is plain FedAvg."""

    def __init__(self, base: nn.Module, model: nn.Module, mu: float) -> None:
        super().__init__()
        self.base = base
        self.mu = mu
        # Plain lists so the model's parameters are not registered on this module.
        self.local_params = [p for p in model.parameters() if p.requires_grad]
        self.global_params = [p.detach().clone() for p in self.local_params]

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = self.base(logits, labels)
        if self.mu > 0:
            drift = sum((w - w0).pow(2).sum() for w, w0 in zip(self.local_params, self.global_params))
            loss = loss + 0.5 * self.mu * drift
        return loss


def local_train(
    model: nn.Module, loader: DataLoader, cfg: TrainingCFG, epochs: int, mu: float, label: str,
    on_epoch=None,
) -> TrainingResults:
    """Run E local epochs from the received global weights and retain their history."""
    device = get_device()
    model.to(device)
    criterion = ProximalLoss(cfg.loss_function, model, mu).to(device)
    optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_kwargs)
    if epochs < 1:
        raise ValueError("Local training requires at least one epoch.")
    metrics = TrainingResults(total_epochs=epochs)
    metrics.initialize(cfg)
    metrics.total_epochs = epochs
    for epoch in range(1, epochs + 1):
        epoch_results = train_one_epoch(
            model, loader, optimizer, criterion, device,
        )
        metrics.add_epoch_results(epoch_results)
        if on_epoch is not None:
            on_epoch(epoch, epoch_results)
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainingCFG, label: str) -> TrainingOneEpochResults:
    device = get_device()
    model.to(device)
    return validate_one_epoch(model, loader, cfg.loss_function.to(device), device)


def scalar_metrics(
    metrics: TrainingResults | TrainingOneEpochResults, keys: tuple[str, ...],
) -> dict[str, float]:
    latest = metrics.train_results[-1] if isinstance(metrics, TrainingResults) else metrics
    scalars = latest.scalar_metrics()
    return {key: float(scalars[key]) for key in keys if math.isfinite(scalars[key])}


class GlobalModelEvaluator:
    """Evaluate aggregated weights on all three fixed splits without augmentation."""

    def __init__(self, federated_cfg: FederatedCFG = None, *, training_cfg=None, datasets=None):
        self.cfg = training_cfg if training_cfg is not None else load_training_cfg(federated_cfg)
        if datasets is None:
            datasets = load_split(self.cfg, federated_cfg.split_ratios)
        self.loaders = {}
        for split, dataset in datasets.items():
            evaluation_data = ISICDataset(dataset.samples, dataset.groups,
                                          transform=self.cfg.val_transforms)
            self.loaders[split] = make_loader(evaluation_data, self.cfg, shuffle=False,
                                             seed=self.cfg.seed)
        self.model = build_model()

    def evaluate(self, arrays) -> dict[str, dict[str, float]]:
        self.model.load_state_dict(arrays.to_torch_state_dict())
        return {split: scalar_metrics(evaluate(self.model, loader, self.cfg,
                                               f"Global model - {split}"), CLIENT_METRICS)
                for split, loader in self.loaders.items()}
