"""Model, data partitioning, and local training shared by the ServerApp and ClientApp.

The baseline modules (isic_dataloaders, neural_network, ...) live in the project
root, which must be on PYTHONPATH; run_federated.sh sets it.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from monai.data.utils import worker_init_fn
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

import isic_dataloaders
from isic_dataloaders import CLASS_NAMES, ISICDataLoaders, ISICDataset
from neural_network import train_one_epoch, validate_one_epoch
from training_configuration import TrainingCFG

# The baseline imports configure the root logger; without this every Flower line prints twice.
logging.getLogger("flwr").propagate = False

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


def load_training_cfg(run_config: Mapping[str, Any]) -> TrainingCFG:
    """Baseline YAML settings with the federated run configuration applied on top."""
    with (PROJECT_DIR / "training_configuration.yaml").open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    data.update(
        batch_size=int(run_config["batch-size"]),
        num_workers=int(run_config["num-workers"]),
        seed=int(run_config["seed"]),
        split_ratios=[float(run_config[f"{split}-fraction"]) for split in SPLITS],
        optimizer_kwargs={
            **data.get("optimizer_kwargs", {}),
            "lr": float(run_config["learning-rate"]),
        },
    )
    return TrainingCFG(data)


def load_split(cfg: TrainingCFG) -> dict[str, ISICDataset]:
    """Return the baseline lesion-separated split, cached as JSON.

    Indexing ISIC checks every image file, which takes about a minute on a
    Windows drive mounted in WSL. Clients that start together may each build
    the cache once; the atomic replace keeps the file consistent.
    """
    ratios = "-".join(f"{ratio:g}" for ratio in cfg.split_ratios)
    cache = OUTPUT_DIR / "cache" / f"split-seed{cfg.seed}-{ratios}.json"
    if not cache.is_file():
        datasets = ISICDataLoaders(
            image_dir=PROJECT_DIR / "ISIC_2019_Training_Input",
            labels_csv=PROJECT_DIR / "ISIC_2019_Training_GroundTruth.csv",
            metadata_csv=PROJECT_DIR / "ISIC_2019_Training_Metadata.csv",
        ).get_datasets_baseline_experiment(train_cfg=cfg)
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
    Every client process recomputes this and gets the same result.
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


def client_datasets(
    cfg: TrainingCFG, run_config: Mapping[str, Any], partition_id: int, num_partitions: int,
) -> tuple[ISICDataset, ISICDataset]:
    """Return this client's local training and validation data."""
    datasets = load_split(cfg)
    train, val = partition(
        datasets, num_partitions, str(run_config["partitioning"]),
        float(run_config["dirichlet-alpha"]), cfg.seed,
    )
    return tuple(
        ISICDataset(
            [datasets[split].samples[i] for i in indices[partition_id]],
            [datasets[split].groups[i] for i in indices[partition_id]],
            transform=datasets[split].transform,
        )
        for split, indices in (("train", train), ("val", val))
    )


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
) -> dict[str, Any]:
    """Run E local epochs from the received global weights; return the last epoch's metrics."""
    device = get_device()
    model.to(device)
    criterion = ProximalLoss(cfg.loss_function, model, mu).to(device)
    optimizer = cfg.optimizer(model.parameters(), **cfg.optimizer_kwargs)
    metrics: dict[str, Any] = {}
    for epoch in range(1, epochs + 1):
        metrics = train_one_epoch(
            model, loader, optimizer, criterion, device,
            description=f"{label} - local epoch {epoch}/{epochs}",
        )
    return metrics


def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainingCFG, label: str) -> dict[str, Any]:
    device = get_device()
    model.to(device)
    return validate_one_epoch(model, loader, cfg.loss_function.to(device), device, description=label)


def scalar_metrics(metrics: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in keys if math.isfinite(metrics[key])}
