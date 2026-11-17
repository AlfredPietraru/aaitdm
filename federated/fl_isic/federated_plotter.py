"""Create figures from federated training results without storing or uploading them."""

from typing import TYPE_CHECKING

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from .federated_training_results import FederatedTrainingResults


class FederatedPlotter:
    """Render the current results snapshot, including partially completed rounds."""

    def __init__(self, results: "FederatedTrainingResults") -> None:
        self.results = results

    def plot_all(self) -> dict[str, Figure]:
        """Return artifact filenames and their figures; the caller owns cleanup."""
        figures = {}
        if self.results.train_metrics_clientapp or self.results.evaluate_metrics_clientapp:
            figures["metrics.png"] = self.plot_metrics()
        if any(self.results.per_client.values()):
            figures["per_client.png"] = self.plot_per_client()
        if self.results.evaluate_metrics_serverapp:
            figures["server_metrics.png"] = self.plot_server_metrics()
        return figures

    def plot_metrics(self) -> Figure:
        """Plot the client-weighted global training and validation histories."""
        figure = Figure(figsize=(11, 4), layout="constrained")
        scores, losses = figure.subplots(1, 2)
        for history, phase, style in (
            (self.results.evaluate_metrics_clientapp, "validation", "-"),
            (self.results.train_metrics_clientapp, "training", "--"),
        ):
            rounds = sorted(history)
            if not rounds:
                continue
            for key in ("accuracy", "macro_f1"):
                scores.plot(rounds, [history[r].get(key, float("nan")) for r in rounds],
                            marker="o", linestyle=style, label=f"client {phase} {key}")
            losses.plot(rounds, [history[r].get("loss", float("nan")) for r in rounds],
                        marker="o", linestyle=style, label=f"client {phase} loss")
        scores.set(title="Global model, scored by the clients", xlabel="round", ylim=(0, 1))
        losses.set(title="Loss", xlabel="round")
        for axis in (scores, losses):
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.3)
            if axis.get_lines():
                axis.legend()
        figure.suptitle(f"{self.title} (client metrics weighted by sample count)")
        return figure

    def plot_per_client(self) -> Figure:
        """Plot the available training and validation metrics for each hospital."""
        figure = Figure(figsize=(11, 10), layout="constrained")
        axes = figure.subplots(3, 2, sharex=True)
        panels = (
            ("train", "loss", "Training loss (last local epoch)"),
            ("val", "loss", "Validation loss (received global model)"),
            ("train", "accuracy", "Training accuracy (last local epoch)"),
            ("val", "accuracy", "Validation accuracy (received global model)"),
            ("train", "macro_f1", "Training macro-F1 (last local epoch)"),
            ("val", "macro_f1", "Validation macro-F1 (received global model)"),
        )
        for axis, (phase, key, title) in zip(axes.flat, panels):
            for client, history in sorted(self.results.per_client[phase].items()):
                rounds = sorted(history)
                if not rounds:
                    continue
                count = history[rounds[-1]].get("num-examples", "?")
                axis.plot(rounds, [history[r].get(key, float("nan")) for r in rounds],
                          marker="o", label=f"hospital {client} (n={count})")
            axis.set(title=title, xlabel="round")
            if key in ("accuracy", "macro_f1"):
                axis.set_ylim(0, 1)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.3)
            if axis.get_lines():
                axis.legend(fontsize=8)
        figure.suptitle(f"{self.title}: per hospital")
        return figure

    def plot_server_metrics(self) -> Figure:
        """Plot any metrics from centralized evaluation of the global model."""
        figure = Figure(figsize=(11, 4), layout="constrained")
        axis = figure.subplots()
        history = self.results.evaluate_metrics_serverapp
        keys = sorted({key for metrics in history.values() for key, value in metrics.items()
                       if isinstance(value, (int, float))})
        rounds = sorted(history)
        for key in keys:
            axis.plot(rounds, [history[r].get(key, float("nan")) for r in rounds],
                      marker="o", label=key)
        axis.set(title="Server evaluation of the global model", xlabel="round")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.3)
        if axis.get_lines():
            axis.legend()
        figure.suptitle(self.title)
        return figure

    @property
    def title(self) -> str:
        return (f"{self.results.strategy} on ISIC 2019, "
                f"{self.results.configuration.partitioning} partitioning")
