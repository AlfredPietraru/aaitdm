"""MLflow tracking for baseline training runs."""

from contextlib import contextmanager
from tempfile import TemporaryDirectory
from pathlib import Path
import os
import json
import math
import yaml
from typing import Any
from collections.abc import Iterator

from mlflow.tracking import MlflowClient

import mlflow.pytorch
from torch import nn

from training_configuration import TrainingCFG


class MLflowLogger:
    """Stateless logging operations against a configurable MLflow tracking server.

    Defaults to MLFLOW_TRACKING_URI, or http://localhost:5000 when unset.
    Authentication uses MLflow's standard environment variables.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )

    @contextmanager
    def run(self, cfg: TrainingCFG) -> Iterator[str]:
        """Create a run from this call's configuration and yield its explicit ID."""
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name(cfg.experiment_name)
        experiment_id = (
            experiment.experiment_id if experiment is not None
            else client.create_experiment(cfg.experiment_name)
        )
        run_id = client.create_run(experiment_id, run_name=cfg.run_name).info.run_id
        status = "FAILED"
        try:
            self.log_configuration(run_id, cfg)
            self.log_tags(run_id, {"stop_reason": "running"})
            yield run_id
            status = "FINISHED"
            if client.get_run(run_id).data.tags.get("stop_reason") == "running":
                self.log_tags(run_id, {"stop_reason": "completed"})
        except KeyboardInterrupt:
            status = "KILLED"
            self.log_tags(run_id, {"stop_reason": "user_ended_pipeline"})
            raise
        except Exception as error:
            self.log_tags(run_id, {
                "stop_reason": "pipeline_error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:1000],
            })
            raise
        finally:
            client.set_terminated(run_id, status=status)

    def log_tags(self, run_id: str, tags: dict[str, str]) -> None:
        """Attach descriptive metadata to an explicitly selected run."""
        client = MlflowClient(tracking_uri=self.tracking_uri)
        for key, value in tags.items():
            client.set_tag(run_id, key, value)

    def log_configuration(self, run_id: str, cfg: TrainingCFG) -> None:
        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.log_dict(run_id, cfg.data, "configuration.json")
        params = {
            "experiment_name": cfg.experiment_name,
            "run_name": cfg.run_name,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "seed": cfg.seed,
            "split_ratios": cfg.split_ratios,
            "oversample_classes": cfg.oversample_classes,
            "optimizer": cfg.optimizer.__name__,
            "loss_function": type(cfg.loss_function).__name__,
            "scheduler": cfg.scheduler_type.__name__ if cfg.scheduler_type else "None",
            "early_stopping_patience": cfg.early_stopping_patience,
            "early_stopping_tolerance": cfg.early_stopping_tolerance,
            **{f"optimizer.{k}": v for k, v in cfg.optimizer_kwargs.items()},
            **{f"scheduler.{k}": v for k, v in cfg.scheduler_kwargs.items()},
        }
        for key, value in params.items():
            client.log_param(run_id, key, value)
        client.log_text(
            run_id,
            yaml.safe_dump({
                "train_transforms": cfg.data.get(
                    "train_transforms", cfg._default_transforms()
                ),
                "val_transforms": cfg.data.get(
                    "val_transforms", cfg._default_transforms()
                ),
            }, sort_keys=False),
            "transforms.txt",
        )

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int) -> None:
        from mlflow.entities import Metric
        import time

        client = MlflowClient(tracking_uri=self.tracking_uri)
        timestamp = int(time.time() * 1000)
        client.log_batch(
            run_id,
            metrics=[Metric(key, float(value), timestamp, step) for key, value in metrics.items()],
            synchronous=True,
        )

    def log_training_history(
        self, run_id: str, epochs: list[dict[str, Any]],
        summary: dict[str, Any], test_metrics: dict[str, Any] | None = None,
        *, cfg: TrainingCFG,
    ) -> None:
        """Save versioned JSON in training_history.txt for cross-run analysis.

        Epochs are one-based. Use pandas.json_normalize(report["epochs"]) for
        columns such as epoch, train.loss, and validation.macro_f1. Undefined
        metrics are JSON null, which pandas can interpret as missing values.
        """
        def json_safe(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: json_safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_safe(item) for item in value]
            return value

        report = {
            "schema_version": 1,
            "run_id": run_id,
            "experiment_name": cfg.experiment_name,
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "configuration": cfg.data,
            "epochs": epochs,
            "summary": summary,
            "test": test_metrics,
        }
        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.log_text(
            run_id, json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
            "training_history.txt",
        )

    def log_epoch_metrics(
        self,
        run_id: str,
        metrics: dict[str, float | list[list[int]]],
        split: str,
        step: int,
        class_names: tuple[str, ...],
    ) -> None:
        """Log scalars plus labeled confusion-matrix JSON and PNG artifacts.

        Matrix rows are true classes; columns are predicted classes.
        """
        from matplotlib.figure import Figure
        import numpy as np

        self.log_metrics(run_id, {
            f"{split}.{key}": value for key, value in metrics.items()
            if key != "confusion_matrix"
        }, step)
        matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
        client = MlflowClient(tracking_uri=self.tracking_uri)
        artifact = f"confusion_matrices/{split}/epoch_{step:04d}"
        client.log_dict(run_id, {
            "class_names": list(class_names),
            "rows": "true", "columns": "predicted", "counts": matrix.tolist(),
        }, f"{artifact}.json")
        figure = Figure(figsize=(9, 8), layout="constrained")
        axis = figure.subplots()
        plot = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(plot, ax=axis)
        axis.set(
            xticks=range(len(class_names)), yticks=range(len(class_names)),
            xticklabels=class_names, yticklabels=class_names,
            xlabel="Predicted class", ylabel="True class",
            title=f"{split.capitalize()} confusion matrix — epoch {step}",
        )
        for (row, column), count in np.ndenumerate(matrix):
            axis.text(column, row, str(count), ha="center", va="center",
                      color="white" if count > matrix.max() / 2 else "black")
        client.log_figure(run_id, figure, f"{artifact}.png")
        figure.clear()

        # Row sums count true-class examples; diagonal entries are correct predictions.
        encountered = matrix.sum(axis=1)
        correct = matrix.diagonal()
        figure = Figure(figsize=(11, 6), layout="constrained")
        axis = figure.subplots()
        positions = np.arange(len(class_names))
        axis.bar(positions, encountered, width=0.7, color="#4477AA",
                 label="Images encountered")
        axis.bar(positions, correct, width=0.4, color="#228833",
                 label="Correctly classified")
        offset = max(int(encountered.max()), 1) * 0.025
        for index, (total, matched) in enumerate(zip(encountered, correct)):
            percentage = f"{100 * matched / total:.1f}%" if total else "N/A"
            axis.text(index, total + offset, f"{matched}/{total}\n{percentage}",
                      ha="center", va="bottom", fontsize=10)
        axis.set(
            xticks=positions, xticklabels=class_names,
            xlabel="True class", ylabel="Number of images",
            title=f"{split.capitalize()} correct predictions by class — epoch {step}\n"
                  "Labels: correct / encountered and per-class recall",
            ylim=(0, max(int(encountered.max()), 1) * 1.25),
        )
        axis.legend()
        client.log_figure(
            run_id, figure, f"class_recall/{split}/epoch_{step:04d}.png"
        )
        figure.clear()

    def log_model(self, run_id: str, model: nn.Module) -> None:
        """Upload a PyTorch model under runs:/<run_id>/model without an active run."""
        client = MlflowClient(tracking_uri=self.tracking_uri)
        client.log_text(run_id, str(model), "model_architecture.txt")
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            mlflow.pytorch.save_model(model, path=str(model_path), serialization_format="pickle")
            client.log_artifacts(run_id, str(model_path), artifact_path="model")

    def get_metric_history(
        self,
        run_id: str,
        metric_names: list[str] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch all stored epoch points, including after this process restarts.

        Omit metric_names to fetch every metric stored for the selected run.
        """
        client = MlflowClient(tracking_uri=self.tracking_uri)
        names = (
            metric_names if metric_names is not None
            else list(client.get_run(run_id).data.metrics)
        )
        return {
            name: [
                {"epoch": point.step, "value": point.value, "timestamp": point.timestamp}
                for point in sorted(
                    client.get_metric_history(run_id, name),
                    key=lambda point: (point.step, point.timestamp),
                )
            ]
            for name in names
        }
