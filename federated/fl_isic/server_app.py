"""Flower ServerApp: coordinates FedAvg/FedProx rounds without touching any data.

Each round the server broadcasts the global weights, the clients train locally,
and the server replaces the global model with the sample-weighted average
w_{t+1} = sum_k (n_k / n) * w_k. The global model is evaluated only by the
clients on their own validation data; the server just averages those metrics.
"""

from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx, Result

from fl_isic.task import OUTPUT_DIR, build_model
from reproducibility import set_seed

app = ServerApp()

# client id -> round -> metrics
ClientHistory = dict[int, dict[int, dict[str, float]]]


def recording_aggregator(
    history: ClientHistory,
) -> Callable[[list[RecordDict], str], MetricRecord]:
    """Weighted average of client metrics that also stores each client's own values."""

    def aggregate(records: list[RecordDict], weighting_key: str) -> MetricRecord:
        total = 0.0
        sums: dict[str, float] = defaultdict(float)
        for record in records:
            metrics = dict(next(iter(record.metric_records.values())))
            client = int(metrics.pop("partition-id"))
            history.setdefault(client, {})[int(metrics.pop("server-round"))] = metrics
            weight = float(metrics[weighting_key])
            total += weight
            for key, value in metrics.items():
                if key != weighting_key:
                    sums[key] += weight * float(value)
        return MetricRecord({key: value / total for key, value in sums.items()})

    return aggregate


@app.main()
def main(grid: Grid, context: Context) -> None:
    run_config = context.run_config
    set_seed(int(run_config["seed"]))
    num_nodes = int(run_config["min-nodes"])
    mu = float(run_config["proximal-mu"])
    per_client: dict[str, ClientHistory] = {"train": {}, "val": {}}
    common = {
        "fraction_train": float(run_config["fraction-train"]),
        "fraction_evaluate": 1.0,
        "min_train_nodes": 1,
        "min_evaluate_nodes": 1,
        "min_available_nodes": num_nodes,
        "train_metrics_aggr_fn": recording_aggregator(per_client["train"]),
        "evaluate_metrics_aggr_fn": recording_aggregator(per_client["val"]),
    }
    strategy = FedProx(proximal_mu=mu, **common) if mu > 0 else FedAvg(**common)
    print(
        f"Server | {type(strategy).__name__} for {run_config['num-server-rounds']} rounds, "
        f"waiting for {num_nodes} clients"
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(build_model(pretrained=True).state_dict()),
        num_rounds=int(run_config["num-server-rounds"]),
    )
    save_results(result, per_client, dict(run_config), type(strategy).__name__)


def save_results(
    result: Result, per_client: dict[str, ClientHistory], run_config: dict, strategy: str,
) -> None:
    """Write per-round metrics, plots, and the final global weights."""
    out = OUTPUT_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    history = {
        name: {int(r): dict(record) for r, record in getattr(result, name).items()}
        for name in ("train_metrics_clientapp", "evaluate_metrics_clientapp")
    }
    (out / "results.json").write_text(
        json.dumps(
            {"strategy": strategy, "run_config": run_config, **history, "per_client": per_client},
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(result.arrays.to_torch_state_dict(), out / "global_model.pt")
    title = f"{strategy} on ISIC 2019, {run_config['partitioning']} partitioning"

    figure, (scores, losses) = plt.subplots(1, 2, figsize=(11, 4))
    for name, phase in (("evaluate_metrics_clientapp", "validation"), ("train_metrics_clientapp", "training")):
        rounds = sorted(history[name])
        for key in ("accuracy", "macro_f1"):
            scores.plot(rounds, [history[name][r][key] for r in rounds], marker="o",
                        linestyle="-" if phase == "validation" else "--", label=f"client {phase} {key}")
        losses.plot(rounds, [history[name][r]["loss"] for r in rounds], marker="o", label=f"client {phase} loss")
    scores.set(title="Global model, scored by the clients", xlabel="round", ylim=(0, 1))
    losses.set(title="Loss", xlabel="round")
    for axis in (scores, losses):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.3)
        axis.legend()
    figure.suptitle(f"{title} (client metrics weighted by n_k)")
    figure.tight_layout()
    figure.savefig(out / "metrics.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = (
        ("train", "loss", "Training loss (last local epoch)"),
        ("val", "loss", "Validation loss (received global model)"),
        ("train", "accuracy", "Training accuracy (last local epoch)"),
        ("val", "accuracy", "Validation accuracy (received global model)"),
    )
    for axis, (phase, key, panel_title) in zip(axes.flat, panels):
        for client, rounds in sorted(per_client[phase].items()):
            ordered = sorted(rounds)
            axis.plot(ordered, [rounds[r][key] for r in ordered], marker="o",
                      label=f"hospital {client} (n={int(rounds[ordered[-1]]['num-examples'])})")
        axis.set(title=panel_title, xlabel="round")
        if key == "accuracy":
            axis.set_ylim(0, 1)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"{title}: per hospital")
    figure.tight_layout()
    figure.savefig(out / "per_client.png", dpi=120)
    plt.close(figure)
    print(f"Server | results saved to {out}")
