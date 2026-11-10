"""Binary image classification and training utilities for WILDS data loaders."""

from copy import deepcopy
from dataclasses import dataclass, field
import math
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from tqdm.auto import tqdm
from training_configuration import TrainingCFG
from mlflow_logger import MLflowLogger


@dataclass
class EarlyStopper:
    """Require an absolute loss drop of at least tolerance from the previous epoch.

    The first validation establishes the baseline. Smaller improvements, ties,
    and regressions count toward patience. A zero tolerance still requires a drop.
    """

    patience: int = 7
    tolerance: float = 0.1
    previous_loss: float | None = field(default=None, init=False)
    bad_epochs: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        assert type(self.patience) is int and self.patience > 0
        assert type(self.tolerance) in (int, float)
        assert math.isfinite(self.tolerance) and self.tolerance >= 0

    def step(self, val_loss: float) -> bool:
        assert math.isfinite(val_loss), "Validation loss must be finite."
        if self.previous_loss is None or (
            val_loss < self.previous_loss
            and (
                self.previous_loss - val_loss >= self.tolerance
                or math.isclose(self.previous_loss - val_loss, self.tolerance, rel_tol=1e-12)
            )
        ):
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.previous_loss = val_loss
        return self.bad_epochs >= self.patience


class BinaryClassifier(nn.Module):
    """Three convolution blocks followed by a binary classification head."""

    def __init__(self, in_channels: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
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
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one logit per image, with shape (batch_size,)."""
        return self.classifier(self.features(images)).squeeze(-1)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Return integer 0/1 predictions; call model.eval() for inference."""
        return (self(images) >= 0).long()


def _run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    description: str = "Evaluation",
) -> dict[str, float]:
    """Accept (images, labels) batches, optionally followed by WILDS metadata."""
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

    with torch.set_grad_enabled(training), tqdm(
        dataloader, desc=description, unit="batch", dynamic_ncols=True
    ) as progress:
        for images, labels, *_ in progress:
            images = images.to(device)
            labels = labels.to(device=device, dtype=torch.float32).reshape(-1)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images).reshape(-1)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.numel()
            correct += ((logits >= 0) == (labels >= 0.5)).sum().item()
            total += labels.numel()
            epoch_labels.append(labels.detach().cpu())
            epoch_scores.append(logits.detach().cpu())
            progress.set_postfix(
                loss=f"{total_loss / total:.4f}",
                accuracy=f"{correct / total:.4f}",
                refresh=False,
            )

    assert total > 0, "The dataloader must contain at least one sample."
    labels = torch.cat(epoch_labels).numpy()
    scores = torch.cat(epoch_scores).numpy()
    # ROC AUC needs both classes. Preserve an undefined value rather than inventing a score.
    auc = float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else float("nan")
    metrics = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "f1": float(f1_score(labels, scores >= 0, zero_division=0)),
        "auc": auc,
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
) -> dict[str, float]:
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
) -> dict[str, float]:
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

    The supplied model must output one binary logit per image. Test data is used
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

        for epoch in range(1, train_cfg.epochs + 1):
            train_metrics = train_one_epoch(
                model, train_dataloader, optimizer, criterion, device,
                description=f"Epoch {epoch}/{train_cfg.epochs} - Training",
            )
            val_metrics = validate_one_epoch(
                model, val_dataloader, criterion, device,
                description=f"Epoch {epoch}/{train_cfg.epochs} - Validation",
            )
            logger.log_metrics(run_id, {
                **{f"train.{key}": value for key, value in train_metrics.items()},
                **{f"val.{key}": value for key, value in val_metrics.items()},
                **{f"learning_rate.group_{i}": group["lr"]
                   for i, group in enumerate(optimizer.param_groups)},
            }, step=epoch)
            assert math.isfinite(val_metrics["loss"]), "Validation loss must be finite."
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            elif scheduler is not None:
                scheduler.step()
            print(
                f"Epoch {epoch}/{train_cfg.epochs} | "
                f"train loss: {train_metrics['loss']:.4f}, "
                f"accuracy: {train_metrics['accuracy']:.4f}, "
                f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}, "
                f"time: {train_metrics['elapsed_seconds']:.2f}s | "
                f"val loss: {val_metrics['loss']:.4f}, "
                f"accuracy: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}, "
                f"time: {val_metrics['elapsed_seconds']:.2f}s"
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(model.state_dict())
                best_epoch = epoch
            if early_stopper.step(val_metrics["loss"]):
                print(f"Early stopping after {epoch} epochs.")
                break

        assert best_state is not None, "Training did not produce a finite validation loss."
        model.load_state_dict(best_state)
        test_metrics = validate_one_epoch(
            model, test_dataloader, criterion, device, description="Test"
        )
        print(
            f"Test loss: {test_metrics['loss']:.4f}, "
            f"accuracy: {test_metrics['accuracy']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}, "
            f"time: {test_metrics['elapsed_seconds']:.2f}s"
        )
        logger.log_metrics(run_id, {
            **{f"test.{key}": value for key, value in test_metrics.items()},
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_completed": epoch,
            "early_stopped": float(epoch < train_cfg.epochs),
        }, step=epoch)
        logger.log_model(run_id, model)
        return model
