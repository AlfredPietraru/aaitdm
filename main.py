import logging
from app_logging import configure_logging

configure_logging()

from pathlib import Path
from typing import Any

import yaml
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from isic_dataloaders import CLASS_NAMES, ISICDataLoaders

from neural_network import train
from mlflow_logger import MLflowLogger
from training_configuration import TrainingCFG
from reproducibility import set_seed


app_logger = logging.getLogger(__name__)


def train_baseline_pipeline() -> nn.Module:
    """Build the ISIC2019 baseline pipeline and return the trained classifier."""
    configure_logging()
    # Resolve beside this module, independently of the working directory.
    with Path(__file__).with_name("training_configuration.yaml").open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    assert isinstance(data, dict), "Training configuration YAML must contain a mapping."
    train_cfg = TrainingCFG(data)
    app_logger.info("Initialization of training parameters complete")

    seed: int = train_cfg.seed
    set_seed(seed, pipelines=(train_cfg.train_transforms, train_cfg.val_transforms))
    app_logger.info("Dataloading helper started...")
    dataset = ISICDataLoaders(oversample_classes=train_cfg.oversample_classes)
    app_logger.info("Dataloading helper complete...")

    app_logger.info("Preparing dataloaders started...")
    train_loader, val_loader, test_loader = dataset.get_dataloaders_baseline_experiment(
        train_cfg=train_cfg,
    )
    app_logger.info("Preparing dataloaders completed...")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    logger = MLflowLogger()
    app_logger.info("Starting training pipeline...")
    return train(model, train_cfg, train_loader, val_loader, test_loader, logger)


if __name__ == "__main__":
    try:
        trained_model = train_baseline_pipeline()
    except KeyboardInterrupt:
        app_logger.info("Pipeline stopped by user (user_ended_pipeline).")
    except Exception:
        app_logger.exception("Training pipeline failed")
        raise SystemExit(1)
