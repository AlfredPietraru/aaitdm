"""Launch federated training, resume an experiment, or run a centralized baseline."""

from pathlib import Path
import os

import mlflow
from mlflow.tracking import MlflowClient
from typing import Any

import yaml
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from isic_dataloaders import CLASS_NAMES, ISICDataLoaders, ensure_isic_dataset

from net_training.neural_network import TrainingResults, train
from mlflow_logger import MLflowLogger
from net_training.training_configuration import TrainingCFG, set_seed


def train_baseline_pipeline() -> tuple[nn.Module, TrainingResults]:
    """Build the ISIC2019 baseline pipeline and return the trained classifier and results."""
    # Resolve relative to the repository, independently of the working directory.
    config_path = Path(__file__).resolve().parent / "app-state" / "training_configuration.yaml"
    with config_path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = yaml.safe_load(handle)
    assert isinstance(data, dict), "Training configuration YAML must contain a mapping."
    train_cfg = TrainingCFG(data)

    seed: int = train_cfg.seed
    set_seed(seed, pipelines=(train_cfg.train_transforms, train_cfg.val_transforms))
    dataset = ISICDataLoaders(oversample_classes=False)

    train_loader, val_loader, test_loader = dataset.get_dataloaders_baseline_experiment(
        train_cfg=train_cfg,
        split_ratios=(0.8, 0.1, 0.1),
    )
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(train_cfg.experiment_name)
    experiment_id = (
        experiment.experiment_id if experiment is not None
        else client.create_experiment(train_cfg.experiment_name)
    )
    run_id = client.create_run(experiment_id, run_name=train_cfg.run_name).info.run_id
    return train(
        model, train_cfg, train_loader, val_loader, test_loader, logger=MLflowLogger(),
        experiment_id=experiment_id, run_id=run_id,
    )



def centralized_main() -> None:
    """Skip downloading complete datasets, then run centralized training."""
    try:
        ensure_isic_dataset()
        train_baseline_pipeline()
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def main() -> None:
    """Start a new federation, or explicitly resume an MLflow experiment."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id", nargs="?", help="MLflow experiment to resume")
    parser.add_argument("--centralized", action="store_true", help="Run the original baseline")
    args = parser.parse_args()
    if args.centralized and args.experiment_id:
        parser.error("experiment_id is only supported for federated training")
    from mlflow_server import tracking_server
    with tracking_server(Path(__file__).resolve().parent / "app-state"):
        launch_training(args, parser)


def launch_training(args, parser) -> None:
    import subprocess
    import sys
    if args.centralized:
        centralized_main()
        return
    environment = os.environ.copy()
    environment["RESUME_EXPERIMENT_ID"] = args.experiment_id or ""
    environment["VENV_BIN"] = str(Path(sys.executable).parent)
    script = Path(__file__).resolve().parent / "federated" / "run_federated.sh"
    raise SystemExit(subprocess.call(["bash", str(script)], env=environment))


if __name__ == "__main__":
    main()
