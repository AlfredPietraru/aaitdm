"""Flower ServerApp: coordinates FedAvg/FedProx rounds without touching any data.

Each round the server broadcasts the global weights, the clients train locally,
and the server replaces the global model with the sample-weighted average
w_{t+1} = sum_k (n_k / n) * w_k. The global model is evaluated only by the
clients on their own validation data; the server just averages those metrics.
"""

from collections import defaultdict
from collections.abc import Callable
import json
import os

import mlflow
from mlflow_logger import MLflowLogger
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from fl_isic.federated_cfg import FederatedCFG
from fl_isic.federated_training_results import (
    FederatedTrainingResults, FederatedTrainingStatus,
)
from fl_isic.task import build_model
from net_training.training_configuration import set_seed

app = ServerApp()


def recording_aggregator(
    results: FederatedTrainingResults,
    phase: str,
    on_update: Callable[[FederatedTrainingResults], None] | None = None,
) -> Callable[[list[RecordDict], str], MetricRecord]:
    """Weighted average of client metrics that also stores each client's own values."""

    def aggregate(records: list[RecordDict], weighting_key: str) -> MetricRecord:
        total = 0.0
        sums: dict[str, float] = defaultdict(float)
        server_round = None
        for record in records:
            metrics = dict(next(iter(record.metric_records.values())))
            client = int(metrics.pop("partition-id"))
            round_number = int(metrics.pop("server-round"))
            if server_round is not None and round_number != server_round:
                raise ValueError("Cannot aggregate replies from different rounds.")
            server_round = round_number
            results.add_client_metrics(phase, client, server_round, metrics)
            weight = float(metrics[weighting_key])
            total += weight
            for key, value in metrics.items():
                if key != weighting_key:
                    sums[key] += weight * float(value)
        if total <= 0 or server_round is None:
            raise ValueError("Cannot aggregate replies without positive sample counts.")
        aggregated = MetricRecord({key: value / total for key, value in sums.items()})
        results.add_round_metrics(phase, server_round, dict(aggregated))
        if on_update is not None:
            on_update(results)
        print(
            f"Server | {phase} | loss={aggregated['loss']:.4f} | "
            f"accuracy={aggregated['accuracy']:.4f} | "
            f"macro_f1={aggregated['macro_f1']:.4f} (sample-weighted client average)",
            flush=True,
        )
        return aggregated

    return aggregate


@app.main()
def main(grid: Grid, context: Context) -> None:
    cfg = FederatedCFG(context.run_config)
    results = FederatedTrainingResults(configuration=cfg)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    from fl_isic.tracking import FederatedTracking
    tracking_config = cfg.to_dict()
    tracking_config.pop("resume-experiment-id")
    tracking = FederatedTracking(tracking_config, cfg.resume_experiment_id)
    experiment_id, run_id = tracking.experiment_id, tracking.run_id
    results.configuration.resume_experiment_id = ""
    checkpoint = tracking.restore()
    completed_round = checkpoint["round"] if checkpoint else 0
    print(f"MLflow experiment: {experiment_id}; completed round: {completed_round}", flush=True)
    if checkpoint and checkpoint.get("results"):
        saved = checkpoint["results"]
        for name in ("train_metrics_clientapp", "evaluate_metrics_clientapp", "evaluate_metrics_serverapp"):
            setattr(results, name, {int(k): v for k, v in saved[name].items()})
        results.per_client = {phase: {int(client): {int(r): values for r, values in history.items()}
            for client, history in clients.items()} for phase, clients in saved["per_client"].items()}
    logger = MLflowLogger()

    def publish(updated_results: FederatedTrainingResults) -> None:
        logger.log_federated(experiment_id, run_id, updated_results)

    try:
        publish(results)
        set_seed(cfg.seed)
        common = cfg.strategy_kwargs(
            train_metrics_aggr_fn=recording_aggregator(results, "train", publish),
            evaluate_metrics_aggr_fn=recording_aggregator(results, "val", publish),
        )
        strategy = (
            FedProx(proximal_mu=cfg.proximal_mu, **common)
            if cfg.proximal_mu > 0 else FedAvg(**common)
        )
        print(
            f"Server | {results.strategy} for {cfg.num_server_rounds} rounds, "
            f"waiting for {cfg.min_nodes} clients"
        )
        from flwr.app import ConfigRecord
        arrays = ArrayRecord(checkpoint["state_dict"]) if checkpoint else ArrayRecord(build_model(pretrained=True).state_dict())
        if checkpoint is None:
            tracking.checkpoint(0, arrays, results)
        for round_number in range(completed_round + 1, completed_round + cfg.num_server_rounds + 1):
            # One strategy round per call makes the returned aggregate a commit boundary.
            offset = round_number - 1
            strategy.train_metrics_aggr_fn = recording_aggregator(results, "train", publish)
            strategy.evaluate_metrics_aggr_fn = recording_aggregator(results, "val", publish)
            config = ConfigRecord({"round-offset": offset, "mlflow-experiment-id": experiment_id,
                "mlflow-client-runs": json.dumps(tracking.manifest["clients"]),
                "mlflow-tracking-uri": mlflow.get_tracking_uri()})
            result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=1,
                                    train_config=config, evaluate_config=config)
            if round_number not in results.train_metrics_clientapp:
                raise RuntimeError("Round did not produce aggregated training metrics")
            arrays = result.arrays
            tracking.checkpoint(round_number, arrays, results)
            results.final_state_dict = arrays.to_torch_state_dict()
        results.finish(FederatedTrainingStatus.SUCCEEDED)
    except KeyboardInterrupt:
        results.finish(FederatedTrainingStatus.INTERRUPTED)
        publish(results)
        tracking.finish("KILLED")
        return
    except Exception as error:
        results.finish(FederatedTrainingStatus.FAILED, error)
        try:
            publish(results)
        except Exception as saving_error:
            error.add_note(f"Publishing partial federated results also failed: {saving_error}")
        tracking.finish("FAILED")
        raise
    publish(results)
    tracking.finish("FINISHED")
