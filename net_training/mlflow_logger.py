"""Stateless publishing of complete training results to an existing MLflow run."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
import json
import math
import time

import mlflow.pytorch
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import yaml

if TYPE_CHECKING:
    from .neural_network import TrainingResults


class MLflowLogger:
    """Publish supplied results without retaining a run, client, or metric history.

    The caller creates the experiment and run. MLflow's standard environment
    variables configure the tracking server and authentication.
    """

    def log(self, experiment_id: str, run_id: str, results: "TrainingResults") -> None:
        """Publish configuration, all epochs, artifacts, model, and final status."""
        client = MlflowClient()
        if str(client.get_run(run_id).info.experiment_id) != str(experiment_id):
            raise ValueError("The run does not belong to the supplied experiment.")
        self.log_configuration(experiment_id, run_id, results)
        self.log_epoch_metrics(experiment_id, run_id, results)
        self.log_training_history(experiment_id, run_id, results)
        self.log_model(experiment_id, run_id, results)
        self.log_status(experiment_id, run_id, results)

    def log_configuration(self, run_id: str, results: "TrainingResults") -> None:
        """Publish the configuration carried by the results."""
        cfg = results.configuration
        if cfg is None:
            raise ValueError("TrainingResults must include its training configuration.")
        client = MlflowClient()
        client.log_dict(run_id, cfg.data, "configuration.json")
        params = {
            "experiment_name": cfg.experiment_name,
            "run_name": cfg.run_name,
            "epochs": results.total_epochs,
            "batch_size": cfg.batch_size,
            "num_workers": cfg.num_workers,
            "seed": cfg.seed,
            "optimizer": cfg.optimizer.__name__,
            "loss_function": type(cfg.loss_function).__name__,
            "scheduler": cfg.scheduler_type.__name__ if cfg.scheduler_type else "None",
            "early_stopping_patience": results.early_stopping_patience,
            "early_stopping_tolerance": results.early_stopping_tolerance,
            **{f"optimizer.{key}": value for key, value in cfg.optimizer_kwargs.items()},
            **{f"scheduler.{key}": value for key, value in cfg.scheduler_kwargs.items()},
        }
        for key, value in params.items():
            client.log_param(run_id, key, value)
        client.log_text(run_id, yaml.safe_dump({
            "train_transforms": cfg.data.get("train_transforms", cfg._default_transforms()),
            "val_transforms": cfg.data.get("val_transforms", cfg._default_transforms()),
        }, sort_keys=False), "transforms.txt")

    def log_epoch_metrics(self, run_id: str, results: "TrainingResults") -> None:
        """Publish every epoch, deriving step numbers from its position in history.

        Existing server metric history prevents duplicate points on repeated calls.
        Confusion-matrix and recall artifacts are overwritten at stable paths.
        """
        from matplotlib.figure import Figure
        import numpy as np

        client = MlflowClient()
        histories = (
            ("train", results.train_results),
            ("val", results.val_results),
            ("test", [results.test_results] if results.test_results is not None else []),
        )
        timestamp = int(time.time() * 1000)
        existing: dict[str, dict[int, float]] = {}
        pending = []
        for split, epochs in histories:
            for step, epoch in enumerate(epochs, start=1):
                scalars = {f"{split}.{key}": value for key, value in epoch.scalar_metrics().items()}
                if split == "train":
                    scalars["epochs_completed"] = float(step)
                    if step <= len(results.learning_rates):
                        scalars.update(results.learning_rates[step - 1])
                for key, value in scalars.items():
                    if not math.isfinite(value):
                        continue
                    if key not in existing:
                        existing[key] = {point.step: point.value
                                         for point in client.get_metric_history(run_id, key)}
                    if existing[key].get(step) != value:
                        pending.append(Metric(key, float(value), timestamp, step))
                matrix = np.asarray(epoch.confusion_matrix, dtype=int)
                class_names = epoch.class_names
                if matrix.shape != (len(class_names), len(class_names)):
                    raise ValueError("Confusion matrix must match the epoch's class names.")
                artifact = f"confusion_matrices/{split}/epoch_{step:04d}"
                client.log_dict(run_id, {
                    "class_names": list(class_names), "rows": "true",
                    "columns": "predicted", "counts": matrix.tolist(),
                }, f"{artifact}.json")
                figure = Figure(figsize=(9, 8), layout="constrained")
                axis = figure.subplots()
                plot = axis.imshow(matrix, cmap="Blues")
                figure.colorbar(plot, ax=axis)
                axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)),
                         xticklabels=class_names, yticklabels=class_names,
                         xlabel="Predicted class", ylabel="True class",
                         title=f"{split.capitalize()} confusion matrix — epoch {step}")
                for (row, column), count in np.ndenumerate(matrix):
                    axis.text(column, row, str(count), ha="center", va="center",
                              color="white" if count > matrix.max() / 2 else "black")
                client.log_figure(run_id, figure, f"{artifact}.png")
                figure.clear()

                encountered = matrix.sum(axis=1)
                correct = matrix.diagonal()
                figure = Figure(figsize=(11, 6), layout="constrained")
                axis = figure.subplots()
                positions = np.arange(len(class_names))
                axis.bar(positions, encountered, width=0.7, color="#4477AA", label="Images encountered")
                axis.bar(positions, correct, width=0.4, color="#228833", label="Correctly classified")
                offset = max(int(encountered.max()), 1) * 0.025
                for index, (total, matched) in enumerate(zip(encountered, correct)):
                    percentage = f"{100 * matched / total:.1f}%" if total else "N/A"
                    axis.text(index, total + offset, f"{matched}/{total}\n{percentage}",
                              ha="center", va="bottom", fontsize=10)
                axis.set(xticks=positions, xticklabels=class_names, xlabel="True class",
                         ylabel="Number of images",
                         title=f"{split.capitalize()} correct predictions by class — epoch {step}",
                         ylim=(0, max(int(encountered.max()), 1) * 1.25))
                axis.legend()
                client.log_figure(run_id, figure, f"class_recall/{split}/epoch_{step:04d}.png")
                figure.clear()
        # Keep requests within MLflow's per-batch metric limit.
        for start in range(0, len(pending), 1000):
            client.log_batch(run_id, metrics=pending[start:start + 1000], synchronous=True)

    def log_training_history(self, experiment_id: str, run_id: str, results: "TrainingResults") -> None:
        """Publish full structured history, including interrupted validation epochs."""
        epochs = []
        for index in range(max(len(results.train_results), len(results.val_results))):
            entry = {"epoch": index + 1}
            if index < len(results.train_results):
                entry["train"] = results.train_results[index].to_dict()
            if index < len(results.val_results):
                entry["validation"] = results.val_results[index].to_dict()
            if index < len(results.learning_rates):
                entry["learning_rates"] = results.learning_rates[index]
            if index < len(results.early_stopping):
                entry.update(results.early_stopping[index])
            epochs.append(entry)
        cfg = results.configuration
        report = {
            "schema_version": 2, "experiment_id": experiment_id, "run_id": run_id,
            "experiment_name": cfg.experiment_name if cfg else None,
            "run_name": cfg.run_name if cfg else None,
            "seed": cfg.seed if cfg else None,
            "configuration": cfg.data if cfg else None,
            "status": results.status.value, "failure_reason": results.failure_reason,
            "epochs": epochs, "summary": results.summary,
            "test": results.test_results.to_dict() if results.test_results else None,
        }
        # JSON null represents undefined metrics rather than non-standard NaN/Infinity.
        serialized = json.dumps(report, allow_nan=True)
        safe = json.loads(serialized, parse_constant=lambda _: None)
        MlflowClient().log_text(run_id, json.dumps(safe, indent=2, allow_nan=False) + "\n",
                               "training_history.txt")

    def log_model(self, run_id: str, results: "TrainingResults") -> None:
        """Upload the successful run's final model when supplied."""
        if results.status.value != "succeeded" or results.model is None:
            return
        client = MlflowClient()
        client.log_text(run_id, str(results.model), "model_architecture.txt")
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model"
            mlflow.pytorch.save_model(results.model, path=str(model_path), serialization_format="pickle")
            client.log_artifacts(run_id, str(model_path), artifact_path="model")

    def log_status(self, run_id: str, results: "TrainingResults") -> None:
        """Map the training outcome to MLflow status and descriptive tags."""
        client = MlflowClient()
        state = results.status.value
        reason = {
            "running": "running", "succeeded": "completed",
            "failed": "pipeline_error", "interrupted": "user_ended_pipeline",
        }[state]
        if state == "succeeded" and results.summary["early_stopped"]:
            reason = "early_stopper_enabled"
        client.set_tag(run_id, "training_status", state)
        client.set_tag(run_id, "stop_reason", reason)
        if results.failure_reason is not None:
            client.set_tag(run_id, "error_message", results.failure_reason[:1000])
        status = {"succeeded": "FINISHED", "failed": "FAILED", "interrupted": "KILLED"}.get(state)
        if status is not None:
            client.set_terminated(run_id, status=status)
