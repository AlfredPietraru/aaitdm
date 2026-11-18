"""Create figures from federated training results without storing or uploading them."""

from typing import TYPE_CHECKING

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from .federated_training_results import FederatedTrainingResults


class FederatedPlotter:
    """Stateless figure generation from caller-provided metrics and distributions."""

    @staticmethod
    def plot_all(results: "FederatedTrainingResults") -> dict[str, Figure]:
        """Return artifact filenames and figures from explicit data; caller owns cleanup."""
        figures = {}
        if any(results.global_metrics.values()):
            title = (f"{results.strategy} on ISIC 2019, "
                     f"{results.configuration.partitioning} partitioning")
            figures["metrics.png"] = FederatedPlotter.plot_metrics(results.global_metrics, title)
        for split in ("train", "val"):
            distributions = {client: counts[split] for client, counts in
                             results.class_distributions.items() if split in counts}
            if distributions:
                figures[f"data_distribution/{split}.png"] = FederatedPlotter.plot_distribution(distributions, split)
        return figures

    @staticmethod
    def plot_metrics(global_metrics, title: str = "Global model") -> Figure:
        """Loss and macro F1 of the aggregated model evaluated by the server."""
        figure = Figure(figsize=(12, 4.5), layout="constrained")
        losses, scores = figure.subplots(1, 2)
        for split, label in (("train", "train"), ("val", "validation"), ("test", "test")):
            history = global_metrics.get(split, {})
            rounds = sorted(history)
            if not rounds:
                continue
            for axis, key in ((losses, "loss"), (scores, "macro_f1")):
                axis.plot(rounds, [history[r].get(key, float("nan")) for r in rounds],
                          marker="o", label=label)
        losses.set(title="Global model loss", xlabel="federated round", ylabel="loss")
        scores.set(title="Global model macro F1", xlabel="federated round", ylabel="macro F1", ylim=(0, 1))
        for axis in (losses, scores):
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=.3)
            if axis.get_lines():
                axis.legend()
        figure.suptitle(f"{title}: server evaluation after aggregation")
        return figure

    @staticmethod
    def plot_client_training(client_id: int, histories: dict[str, dict[int, float]]) -> Figure:
        figure = Figure(figsize=(11, 4), layout="constrained")
        for axis, key, title in zip(figure.subplots(1, 2), ("accuracy", "loss"),
                                    ("Training accuracy", "Training loss")):
            history = histories[key]
            steps = sorted(history)
            axis.plot(steps, [history[step] for step in steps], marker="o")
            axis.set(title=title, xlabel="local epoch (cumulative)", ylabel=key)
            if key == "accuracy":
                axis.set_ylim(0, 1)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=.3)
        figure.suptitle(f"Client {client_id}")
        return figure

    @staticmethod
    def plot_distribution(distributions: dict[str, dict[str, int]], split: str) -> Figure:
        """Percentages use each client's split total; counts are image entries."""
        clients = sorted(distributions, key=int)
        classes = list(dict.fromkeys(name for client in clients for name in distributions[client]))
        figure = Figure(figsize=(max(9, len(classes) * 1.1), max(3.8, 3.3 * len(clients))),
                        layout="constrained")
        axes = figure.subplots(len(clients), 1, squeeze=False).flat
        maximum = max((count for counts in distributions.values() for count in counts.values()), default=0)
        for axis, client in zip(axes, clients):
            counts = [distributions[client].get(name, 0) for name in classes]
            total = sum(counts)
            bars = axis.bar(classes, counts, color="steelblue")
            axis.bar_label(bars, labels=[f"{count}\n{100 * count / total if total else 0:.1f}%"
                                        for count in counts], padding=3, fontsize=9)
            axis.set(title=f"Client {client} (n={total})", ylabel="image entries",
                     ylim=(0, max(1, maximum * 1.3)))
            axis.yaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(axis="y", alpha=.2)
        figure.suptitle(f"{split.capitalize()} class distribution (percentage within each client)")
        return figure
