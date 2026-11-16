"""Multiclass image classification and training utilities."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import math
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from .training_configuration import TrainingCFG
from .mlflow_logger import MLflowLogger


@dataclass
class TrainingOneEpochResults:
    """Mutable metrics produced by one training or evaluation epoch."""

    loss: float
    accuracy: float
    macro_f1: float
    macro_auroc: float
    balanced_accuracy: float
    confusion_matrix: list[list[int]]
    entries: int
    split: str = "train"
    class_names: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    recalls: dict[str, float] = field(default_factory=dict)

    def scalar_metrics(self) -> dict[str, float]:
        """Return this epoch's scalar metrics using the tracking names."""
        return {
            **{name: float(getattr(self, name)) for name in
               ("loss", "accuracy", "macro_f1", "macro_auroc", "balanced_accuracy",
                "entries", "elapsed_seconds")},
            **{f"recall.{name}": value for name, value in self.recalls.items()},
        }

    def to_dict(self) -> dict[str, float | list[list[int]]]:
        """Serialize this epoch's metrics."""
        return {**self.scalar_metrics(),
                "confusion_matrix": deepcopy(self.confusion_matrix)}


class TrainingStatus(str, Enum):
    """Lifecycle state and terminal reason for a training process."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class TrainingResults:
    """Store results for the entire training process, including final testing.

    Each list stores independent epoch snapshots in execution order. The
    configured total and the number of completed epochs remain distinct when
    early stopping ends training before the configured limit.
    """

    total_epochs: int = 0
    status: TrainingStatus = TrainingStatus.RUNNING
    failure_reason: str | None = None
    configuration: TrainingCFG | None = field(default=None, repr=False)
    model: nn.Module | None = field(default=None, repr=False)
    train_results: list[TrainingOneEpochResults] = field(default_factory=list)
    val_results: list[TrainingOneEpochResults] = field(default_factory=list)
    test_results: TrainingOneEpochResults | None = None
    learning_rates: list[dict[str, float]] = field(default_factory=list)
    early_stopping_patience: int = 7
    early_stopping_tolerance: float = 0.0
    early_stopping: list[dict[str, int | float | None]] = field(default_factory=list)

    def initialize(self, train_cfg: TrainingCFG) -> None:
        """Initialize training metadata and lifecycle state from configuration."""
        self.configuration = train_cfg
        self.model = None
        self.total_epochs = train_cfg.epochs
        self.early_stopping_patience = train_cfg.early_stopping_patience
        self.early_stopping_tolerance = train_cfg.early_stopping_tolerance
        self.status = TrainingStatus.RUNNING
        self.failure_reason = None

    @property
    def epoch(self) -> int:
        """Infer the number of completed training epochs from stored results."""
        return len(self.train_results)

    @property
    def summary(self) -> dict[str, int | float | bool]:
        """Derive the summary from current validation history and stop settings."""
        best_epoch, best_val_loss = min(
            ((index, result.loss) for index, result in enumerate(self.val_results, start=1)
             if math.isfinite(result.loss)),
            key=lambda item: item[1],
            default=(0, float("inf")),
        )
        stopper = EarlyStopper(
            patience=self.early_stopping_patience,
            tolerance=self.early_stopping_tolerance,
        )
        early_stopped = False
        for result in self.val_results:
            if math.isfinite(result.loss):
                early_stopped = stopper.step(result.loss) or early_stopped
        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "epochs_completed": self.epoch,
            "early_stopped": early_stopped,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_tolerance": self.early_stopping_tolerance,
        }

    def add_epoch_results(self, results: TrainingOneEpochResults) -> None:
        """Store an independent snapshot of a training, validation, or test result."""
        if results.split == "train":
            self.train_results.append(deepcopy(results))
        elif results.split == "val":
            self.val_results.append(deepcopy(results))
        elif results.split == "test":
            self.test_results = deepcopy(results)
        else:
            raise ValueError(f"Unknown dataset split: {results.split!r}")


@dataclass
class EarlyStopper:
    """Stop after patience epochs without improving the best accepted loss.

    The first validation establishes the baseline. Smaller improvements, ties,
    and regressions count toward patience. With tolerance=0, every new minimum
    resets patience. Positive tolerance requires that much cumulative improvement
    over the last accepted best loss before resetting.
    """

    patience: int = 7
    tolerance: float = 0.0
    best_loss: float | None = field(default=None, init=False)
    bad_epochs: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        assert type(self.patience) is int and self.patience > 0
        assert type(self.tolerance) in (int, float)
        assert math.isfinite(self.tolerance) and self.tolerance >= 0

    def step(self, val_loss: float) -> bool:
        assert math.isfinite(val_loss), "Validation loss must be finite."
        if self.best_loss is None or (
            val_loss < self.best_loss
            and (
                self.best_loss - val_loss >= self.tolerance
                or math.isclose(self.best_loss - val_loss, self.tolerance, rel_tol=1e-12)
            )
        ):
            self.bad_epochs = 0
            self.best_loss = val_loss
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    split: str = "val",
) -> TrainingOneEpochResults:
    """Accept (images, labels) batches, optionally followed by metadata."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = perf_counter()
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_labels = []
    epoch_scores = []

    with torch.set_grad_enabled(training):
        for images, labels, *_ in dataloader:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.long).reshape(-1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.numel()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.numel()
            epoch_labels.append(labels.detach().cpu())
            epoch_scores.append(logits.detach().softmax(dim=1).cpu())

    assert total > 0, "The dataloader must contain at least one sample."
    labels = torch.cat(epoch_labels).numpy()
    scores = torch.cat(epoch_scores).numpy()
    num_classes = scores.shape[1]
    predictions = scores.argmax(axis=1)
    # All one-vs-rest problems need positives and negatives for macro AUC.
    if len(set(labels)) == num_classes:
        auc = float(roc_auc_score(labels, scores[:, 1])) if num_classes == 2 else float(
            roc_auc_score(labels, scores, multi_class="ovr", average="macro",
                          labels=list(range(num_classes)))
        )
    else:
        auc = float("nan")
    matrix = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    support = matrix.sum(axis=1)
    recalls = [float(matrix[i, i] / support[i]) if support[i] else float("nan")
               for i in range(num_classes)]
    class_names = getattr(dataloader.dataset, "class_names", tuple(map(str, range(num_classes))))
    if len(class_names) != num_classes:
        raise ValueError("Dataset class names must match the model output dimension.")
    metrics = TrainingOneEpochResults(
        split=split,
        class_names=tuple(class_names),
        loss=total_loss / total,
        accuracy=correct / total,
        macro_f1=float(f1_score(labels, predictions, average="macro",
                                labels=list(range(num_classes)), zero_division=0)),
        macro_auroc=auc,
        # Undefined when any class has no examples: this is an all-class average.
        balanced_accuracy=sum(recalls) / num_classes,
        recalls=dict(zip(class_names, recalls)),
        confusion_matrix=matrix.tolist(),
        entries=total,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics.elapsed_seconds = perf_counter() - started_at
    return metrics


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> TrainingOneEpochResults:
    """Update weights and return metrics, sample count, and elapsed seconds."""
    return _run_one_epoch(
        model, train_dataloader, criterion, device, optimizer, split="train"
    )


def validate_one_epoch(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
) -> TrainingOneEpochResults:
    """Evaluate without gradients or updates to model weights/statistics."""
    return _run_one_epoch(model, val_dataloader, criterion, device, split=split)


def evaluate_test_set(
    model: nn.Module,
    best_state: dict[str, torch.Tensor] | None,
    test_dataloader: DataLoader | None,
    criterion: nn.Module,
    device: torch.device,
    results: TrainingResults,
) -> nn.Module:
    """Restore selected weights if available and optionally evaluate test data."""
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if test_dataloader is not None:
        test_metrics = validate_one_epoch(
            model, test_dataloader, criterion, device, split="test"
        )
        results.add_epoch_results(test_metrics)
    results.model = model
    return model


def train(
    model: nn.Module,
    train_cfg: TrainingCFG,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader | None = None,
    test_dataloader: DataLoader | None = None,
    *,
    logger: MLflowLogger,
    experiment_id: str,
    run_id: str,
) -> tuple[nn.Module, TrainingResults]:
    """Train in place, restore the best validation weights, then evaluate test data.

    The supplied model must output one raw logit per class per image. Test data is used
    only after model selection. The returned model is in evaluation mode.
    Without validation, retain the final weights and skip early stopping and
    ReduceLROnPlateau updates. Without a test loader, skip test evaluation.
    Return the model and its complete TrainingResults. Failures propagate;
    Ctrl+C returns the current model and interrupted results without running
    test evaluation or uploading a final model.
    """
    results = TrainingResults()
    results.initialize(train_cfg)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = train_cfg.loss_function.to(device)
        optimizer = train_cfg.optimizer(model.parameters(), **train_cfg.optimizer_kwargs)
        scheduler = (
            train_cfg.scheduler_type(optimizer, **train_cfg.scheduler_kwargs)
            if train_cfg.scheduler_type is not None
            else None
        )
        early_stopper = (
            EarlyStopper(
                patience=train_cfg.early_stopping_patience,
                tolerance=train_cfg.early_stopping_tolerance,
            ) if val_dataloader is not None else None
        )
        best_val_loss = float("inf")
        best_state = None
        early_stopped = False
        for epoch in range(1, train_cfg.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_dataloader, optimizer, criterion, device,
            )
            results.add_epoch_results(train_metrics)
            val_metrics = None
            if val_dataloader is not None:
                val_metrics = validate_one_epoch(
                    model, val_dataloader, criterion, device,
                )
                assert math.isfinite(val_metrics.loss), "Validation loss must be finite."
                results.add_epoch_results(val_metrics)
            learning_rates = {
                f"learning_rate.group_{i}": group["lr"]
                for i, group in enumerate(optimizer.param_groups)
            }
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if val_metrics is not None:
                    scheduler.step(val_metrics.loss)
            elif scheduler is not None:
                scheduler.step()
            if val_metrics is not None:
                if val_metrics.loss < best_val_loss:
                    best_val_loss = val_metrics.loss
                    best_state = deepcopy(model.state_dict())
                early_stopped = early_stopper.step(val_metrics.loss)
                results.early_stopping.append({
                    "early_stopping_counter": early_stopper.bad_epochs,
                    "early_stopping_reference_loss": early_stopper.best_loss,
                })
            results.learning_rates.append(learning_rates)
            if early_stopped:
                break

        model = evaluate_test_set(
            model, best_state, test_dataloader, criterion, device, results,
        )
        results.status = TrainingStatus.SUCCEEDED
    except KeyboardInterrupt:
        results.status = TrainingStatus.INTERRUPTED
        model.eval()
        logger.log(experiment_id, run_id, results)
        return model, results
    except Exception as error:
        results.status = TrainingStatus.FAILED
        results.failure_reason = f"{type(error).__name__}: {error}"
        try:
            logger.log(experiment_id, run_id, results)
        except Exception as logging_error:
            error.add_note(f"MLflow publishing also failed: {logging_error}")
        raise
    logger.log(experiment_id, run_id, results)
    return model, results
