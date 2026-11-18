"""Coordinate FedAvg/FedProx and evaluate aggregated models on fixed data splits."""

from collections import defaultdict
from collections.abc import Callable
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
from fl_isic.task import build_model, GlobalModelEvaluator
from fl_isic.protocol import ClientResponse
from fl_isic.coordination import ServerTrainingPlan, NamedRequestStrategy


class TrainingFedAvg(NamedRequestStrategy, FedAvg):
    """FedAvg using the shared request protocol."""


class TrainingFedProx(NamedRequestStrategy, FedProx):
    """FedProx using the shared request protocol."""

from net_training.training_configuration import set_seed
from net_training.neural_network import EarlyStopper

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
        validation_sums: dict[str, float] = defaultdict(float)
        validation_total = 0
        server_round = None
        for record in records:
            response = ClientResponse.from_content(record)
            client = response.client_id
            round_number = response.server_round
            metrics = {**response.metrics, weighting_key: response.num_examples}
            if server_round is not None and round_number != server_round:
                raise ValueError("Cannot aggregate replies from different rounds.")
            server_round = round_number
            if response.distributions is not None:
                results.class_distributions[str(client)] = response.distributions
            results.add_client_metrics(phase, client, server_round, metrics)
            if phase == "train" and response.validation_metrics is not None:
                validation = {**response.validation_metrics,
                              "num-examples": response.num_validation_examples}
                results.add_client_metrics("val", client, server_round, validation)
                validation_total += response.num_validation_examples
                for key, value in response.validation_metrics.items():
                    validation_sums[key] += response.num_validation_examples * value
            weight = float(metrics[weighting_key])
            total += weight
            for key, value in metrics.items():
                if key != weighting_key:
                    sums[key] += weight * float(value)
        if total <= 0 or server_round is None:
            raise ValueError("Cannot aggregate replies without positive sample counts.")
        aggregated = MetricRecord({key: value / total for key, value in sums.items()})
        results.add_round_metrics(phase, server_round, dict(aggregated))
        if validation_total:
            results.add_round_metrics("val", server_round,
                {key: value / validation_total for key, value in validation_sums.items()})
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


def update_early_stopping(results, stopper, round_number, validation_loss):
    """Only server validation loss controls patience; tests never influence it."""
    previous_best = stopper.best_loss
    stopped = stopper.step(validation_loss)
    best_round = (round_number if stopper.best_loss != previous_best
                  else results.early_stopping.get("best_round"))
    results.early_stopping = {"metric": "global.val.loss", "patience": stopper.patience,
        "best_loss": stopper.best_loss, "best_round": best_round,
        "bad_rounds": stopper.bad_epochs, "stopped": stopped, "round": round_number}
    return stopped


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
        results.global_metrics = {split: {int(r): values for r, values in history.items()}
                                  for split, history in saved.get("global_metrics", {
                                      "train": {}, "val": {}, "test": {}}).items()}
        results.class_distributions = saved.get("class_distributions", {})
        for name in ("train_metrics_clientapp", "evaluate_metrics_clientapp", "evaluate_metrics_serverapp"):
            setattr(results, name, {int(k): v for k, v in saved[name].items()})
        results.per_client = {phase: {int(client): {int(r): values for r, values in history.items()}
            for client, history in clients.items()} for phase, clients in saved["per_client"].items()}
    stopper = EarlyStopper(patience=cfg.early_stopping_patience, tolerance=0.0)
    # Rebuild patience from committed server validation history on resume.
    stopped = False
    for round_number, metrics in sorted(results.global_metrics["val"].items()):
        stopped = update_early_stopping(results, stopper, round_number, metrics["loss"])
    logger = MLflowLogger()

    def publish(updated_results: FederatedTrainingResults) -> None:
        logger.log_federated(experiment_id, run_id, updated_results)

    try:
        if stopped:
            results.stop_reason = "early_stopping"
            results.finish(FederatedTrainingStatus.SUCCEEDED)
            publish(results)
            tracking.finish("FINISHED")
            print("Validation-loss patience already exhausted; no further rounds scheduled.", flush=True)
            return
        publish(results)
        set_seed(cfg.seed)
        common = cfg.strategy_kwargs(
            train_metrics_aggr_fn=recording_aggregator(results, "train", publish),
            evaluate_metrics_aggr_fn=recording_aggregator(results, "val", publish),
        )
        plan = ServerTrainingPlan(cfg, tracking)
        node_clients = plan.initialize_clients(grid, mlflow.get_tracking_uri())
        results.class_distributions = plan.class_distributions()
        common.update(node_clients=node_clients, training_plan=plan)
        strategy = (
            TrainingFedProx(proximal_mu=cfg.proximal_mu, **common)
            if cfg.proximal_mu > 0 else TrainingFedAvg(**common)
        )
        print(
            f"Server | {results.strategy} for {cfg.num_server_rounds} rounds, "
            f"waiting for {cfg.min_nodes} clients"
        )
        arrays = ArrayRecord(checkpoint["state_dict"]) if checkpoint else ArrayRecord(build_model(pretrained=True).state_dict())
        if checkpoint is None:
            tracking.checkpoint(0, arrays, results)
        evaluator = GlobalModelEvaluator(training_cfg=plan.training_cfg, datasets=plan.evaluation_datasets())
        for round_number in range(completed_round + 1, completed_round + cfg.num_server_rounds + 1):
            # One strategy round per call makes the returned aggregate a commit boundary.
            strategy.round_number = round_number
            strategy.train_metrics_aggr_fn = recording_aggregator(results, "train", publish)
            strategy.evaluate_metrics_aggr_fn = recording_aggregator(results, "val", publish)
            result = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=1)
            if round_number not in results.train_metrics_clientapp:
                raise RuntimeError("Round did not produce aggregated training metrics")
            arrays = result.arrays
            global_metrics = evaluator.evaluate(arrays)
            for split, metrics in global_metrics.items():
                results.global_metrics[split][round_number] = metrics
                print(f"Global | round {round_number} | {split} | "
                      f"loss={metrics['loss']:.4f} | accuracy={metrics['accuracy']:.4f} | "
                      f"macro_f1={metrics['macro_f1']:.4f}", flush=True)
            stopped = update_early_stopping(results, stopper, round_number, global_metrics["val"]["loss"])
            if stopped:
                results.stop_reason = "early_stopping"
            elif round_number == completed_round + cfg.num_server_rounds:
                results.stop_reason = "round_limit"
            results.final_state_dict = arrays.to_torch_state_dict()
            tracking.checkpoint(round_number, arrays, results)
            publish(results)
            if stopped:
                print(f"Early stopping at round {round_number}: no validation-loss improvement for "
                      f"{cfg.early_stopping_patience} rounds (best={stopper.best_loss:.6f}).", flush=True)
                break
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
