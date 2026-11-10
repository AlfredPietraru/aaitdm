from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from monai.data.utils import worker_init_fn
from isic_dataloaders import CLASS_NAMES, ISICDataLoaders

from neural_network import MulticlassClassifier, train
from mlflow_logger import MLflowLogger
from training_configuration import TrainingCFG
from reproducibility import set_seed


def train_baseline_pipeline() -> nn.Module:
    """Build the ISIC2019 baseline pipeline and return the trained classifier."""
    # Resolve beside this module, independently of the working directory.
    with Path(__file__).with_name("training_configuration.yaml").open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    assert isinstance(data, dict), "Training configuration YAML must contain a mapping."
    train_cfg = TrainingCFG(data)
    print("Initialization of training parameters complete")

    seed: int = 42
    set_seed(seed, pipelines=(train_cfg.train_transforms, train_cfg.val_transforms))
    print("Dataloading helper started...")
    dataset = ISICDataLoaders()
    print("Dataloading helper complete...")

    loader_kwargs = {
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
    }
    print("Preparing dataloaders started...")
    train_loader, val_loader, test_loader = dataset.get_dataloaders_baseline_experiment(
        split_ratios=(0.75, 0.15, 0.15),
        batch_size=train_cfg.batch_size,
        seed=seed,
        train_transform=train_cfg.train_transforms,
        eval_transform=train_cfg.val_transforms,
        **loader_kwargs,
    )
    print("Preparing dataloaders completed...")
    model = MulticlassClassifier(num_classes=len(CLASS_NAMES))
    logger = MLflowLogger()
    print("Starting training pipeline...")
    return train(model, train_cfg, train_loader, val_loader, test_loader, logger)


if __name__ == "__main__":
    try:
        trained_model = train_baseline_pipeline()
    except KeyboardInterrupt:
        print("Pipeline stopped by user (user_ended_pipeline).")
