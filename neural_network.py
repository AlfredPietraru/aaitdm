"""Multiclass image classification and training utilities."""

from copy import deepcopy
from dataclasses import dataclass, field
import math
import logging
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from training_configuration import TrainingCFG
from mlflow_logger import MLflowLogger


app_logger = logging.getLogger(__name__)

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


class MulticlassClassifier(nn.Module):
    """Three convolution blocks followed by a multiclass classification head."""

    def __init__(
        self, in_channels: int = 3, dropout: float = 0.3, num_classes: int = 8,
    ) -> None:
        super().__init__()
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer greater than one.")
        self.num_classes = num_classes
        blocks = []
        for out_channels in (32, 64, 128):
            blocks.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits with shape (batch_size, num_classes)."""
        return self.classifier(self.features(images))

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Return integer class indices; call model.eval() for inference."""
        return self(images).argmax(dim=1)


def _run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str = "Evaluation",
) -> dict[str, float | list[list[int]]]:
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
    running_confusion = None

    app_logger.info("%s started: %s batches", description, len(dataloader))
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels, *_) in enumerate(dataloader, start=1):
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
            num_classes = logits.shape[1]
            batch_confusion = torch.bincount(
                epoch_labels[-1] * num_classes + epoch_scores[-1].argmax(dim=1),
                minlength=num_classes * num_classes,
            ).reshape(num_classes, num_classes)
            if running_confusion is None:
                running_confusion = batch_confusion
            else:
                running_confusion += batch_confusion
            # Aggregate all batches seen so far; averaging batch F1 is incorrect.
            denominator = running_confusion.sum(dim=0) + running_confusion.sum(dim=1)
            running_macro_f1 = (
                2 * running_confusion.diag().double() / denominator.clamp_min(1)
            ).mean().item()
            app_logger.info(
                "%s | batch %d/%d | loss=%.4f | macro_f1=%.4f",
                description, batch_index, len(dataloader), total_loss / total, running_macro_f1,
            )

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
    metrics = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "macro_f1": float(f1_score(labels, predictions, average="macro",
                             labels=list(range(num_classes)), zero_division=0)),
        "macro_auroc": auc,
        # Undefined when any class has no examples: this is an all-class average.
        "balanced_accuracy": sum(recalls) / num_classes,
        **{f"recall.{name}": recall for name, recall in zip(class_names, recalls)},
        "confusion_matrix": matrix.tolist(),
        "entries": float(total),
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics["elapsed_seconds"] = perf_counter() - started_at
    return metrics


def train_one_epoch(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    description: str = "Training",
) -> dict[str, float | list[list[int]]]:
    """Update weights and return metrics, sample count, and elapsed seconds."""
    return _run_one_epoch(
        model, train_dataloader, criterion, device, optimizer, description=description
    )


def validate_one_epoch(
    model: nn.Module,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    description: str = "Validation",
) -> dict[str, float | list[list[int]]]:
    """Evaluate without gradients or updates to model weights/statistics."""
    return _run_one_epoch(model, val_dataloader, criterion, device, description=description)


def train(
    model: nn.Module,
    train_cfg: TrainingCFG,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
    logger: MLflowLogger,
) -> nn.Module:
    """Train in place, restore the best validation weights, then evaluate test data.

    The supplied model must output one raw logit per class per image. Test data is used
    only after model selection. The returned model is in evaluation mode.
    """
    with logger.run(train_cfg) as run_id:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = train_cfg.loss_function.to(device)
        optimizer = train_cfg.optimizer(model.parameters(), **train_cfg.optimizer_kwargs)
        scheduler = (
            train_cfg.scheduler_type(optimizer, **train_cfg.scheduler_kwargs)
            if train_cfg.scheduler_type is not None
            else None
        )
        early_stopper = EarlyStopper(
            patience=train_cfg.early_stopping_patience,
            tolerance=train_cfg.early_stopping_tolerance,
        )
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        early_stopped = False
        epoch_history = []
        for epoch in range(1, train_cfg.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_dataloader, optimizer, criterion, device,
                description=f"Epoch {epoch}/{train_cfg.epochs} - Training",
            )
            val_metrics = validate_one_epoch(
                model, val_dataloader, criterion, device,
                description=f"Epoch {epoch}/{train_cfg.epochs} - Validation",
            )
            logger.log_epoch_metrics(run_id, train_metrics, "train", epoch,
                                     getattr(train_dataloader.dataset, "class_names",
                                             tuple(map(str, range(len(train_metrics["confusion_matrix"]))))))
            logger.log_epoch_metrics(run_id, val_metrics, "val", epoch,
                                     getattr(val_dataloader.dataset, "class_names",
                                             tuple(map(str, range(len(val_metrics["confusion_matrix"]))))))
            learning_rates = {
                f"learning_rate.group_{i}": group["lr"]
                for i, group in enumerate(optimizer.param_groups)
            }
            logger.log_metrics(run_id, learning_rates, step=epoch)
            assert math.isfinite(val_metrics["loss"]), "Validation loss must be finite."
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            elif scheduler is not None:
                scheduler.step()
            app_logger.info(
                f"Epoch {epoch}/{train_cfg.epochs} | "
                f"train loss: {train_metrics['loss']:.4f}, "
                f"balanced accuracy: {train_metrics['balanced_accuracy']:.4f}, "
                f"Macro F1: {train_metrics['macro_f1']:.4f}, Macro OvR AUC: {train_metrics['macro_auroc']:.4f}, "
                f"time: {train_metrics['elapsed_seconds']:.2f}s | "
                f"val loss: {val_metrics['loss']:.4f}, "
                f"balanced accuracy: {val_metrics['balanced_accuracy']:.4f}, "
                f"Macro F1: {val_metrics['macro_f1']:.4f}, Macro OvR AUC: {val_metrics['macro_auroc']:.4f}, "
                f"time: {val_metrics['elapsed_seconds']:.2f}s"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
            early_stopped = early_stopper.step(val_metrics["loss"])
            epoch_history.append({
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "learning_rates": learning_rates,
                "early_stopping_counter": early_stopper.bad_epochs,
                "early_stopping_reference_loss": early_stopper.best_loss,
            })
            summary = {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "epochs_completed": epoch,
                "early_stopped": early_stopped,
                "early_stopping_patience": early_stopper.patience,
                "early_stopping_tolerance": early_stopper.tolerance,
            }
            logger.log_training_history(run_id, epoch_history, summary, cfg=train_cfg)
            if early_stopped:
                logger.log_tags(run_id, {"stop_reason": "early_stopper_enabled"})
                app_logger.info(f"Early stopping after {epoch} epochs.")
                break

        assert best_state is not None, "Training did not produce a finite validation loss."
        model.load_state_dict(best_state)
        test_metrics = validate_one_epoch(
            model, test_dataloader, criterion, device, description="Test"
        )
        app_logger.info(
            f"Test loss: {test_metrics['loss']:.4f}, "
            f"balanced accuracy: {test_metrics['balanced_accuracy']:.4f}, "
            f"Macro F1: {test_metrics['macro_f1']:.4f}, Macro OvR AUC: {test_metrics['macro_auroc']:.4f}, "
            f"time: {test_metrics['elapsed_seconds']:.2f}s"
        )
        logger.log_epoch_metrics(run_id, test_metrics, "test", epoch,
                                 getattr(test_dataloader.dataset, "class_names",
                                             tuple(map(str, range(len(test_metrics["confusion_matrix"]))))))
        logger.log_metrics(run_id, {
            "epochs_completed": epoch,
        }, step=epoch)
        logger.log_training_history(run_id, epoch_history, summary, test_metrics, cfg=train_cfg)
        logger.log_model(run_id, model)
        return model
