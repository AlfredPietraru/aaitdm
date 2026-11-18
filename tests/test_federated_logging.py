"""Regression checks for federated plotting and incremental MLflow publishing."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy import Result

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "federated"))
from fl_isic.federated_cfg import FederatedCFG
from fl_isic.federated_training_results import FederatedTrainingResults, FederatedTrainingStatus
from fl_isic.federated_plotter import FederatedPlotter
from fl_isic import server_app
from mlflow_logger import MLflowLogger


class FederatedLoggingTests(unittest.TestCase):
    def setUp(self):
        self.results = FederatedTrainingResults(FederatedCFG({}))

    def add_round(self, number):
        values = {"loss": 1 / number, "accuracy": .6, "macro_f1": .5}
        for split in ("train", "val", "test"):
            self.results.global_metrics[split][number] = dict(values)
        for phase in ("train", "val"):
            self.results.add_round_metrics(phase, number, values)
            self.results.add_client_metrics(phase, 0, number, {**values, "num-examples": 8})

    def test_plotter_handles_partial_histories_and_updates_rounds(self):
        self.assertEqual(FederatedPlotter.plot_all(self.results), {})
        self.results.global_metrics["train"][1] = {"loss": .8}
        figures = FederatedPlotter.plot_all(self.results)
        self.assertEqual(set(figures), {"metrics.png"})
        for figure in figures.values():
            figure.clear()
        self.add_round(2)
        figures = FederatedPlotter.plot_all(self.results)
        try:
            self.assertEqual(set(figures), {"metrics.png"})
            self.assertEqual(list(figures["metrics.png"].axes[0].lines[0].get_xdata()), [1, 2])
        finally:
            for figure in figures.values():
                figure.clear()

    def test_distribution_counts_percentages_and_shared_scales(self):
        figure = FederatedPlotter.plot_distribution({"0": {"A": 3, "B": 1, "C": 0},
                                       "1": {"A": 1, "B": 1, "C": 0}}, "train")
        try:
            self.assertEqual([text.get_text() for text in figure.axes[0].texts],
                             ["3\n75.0%", "1\n25.0%", "0\n0.0%"])
            self.assertEqual([text.get_text() for text in figure.axes[1].texts],
                             ["1\n50.0%", "1\n50.0%", "0\n0.0%"])
            self.assertEqual(figure.axes[0].get_ylim(), figure.axes[1].get_ylim())
            self.assertEqual([bar.get_height() for bar in figure.axes[0].patches], [3, 1, 0])
        finally:
            figure.clear()
        curve = FederatedPlotter.plot_client_training(0, {"accuracy": {1: .5, 3: .8}, "loss": {1: 1., 3: .4}})
        try:
            self.assertEqual(len(curve.axes), 2)
            self.assertEqual(list(curve.axes[0].lines[0].get_xdata()), [1, 3])
            self.assertEqual(list(curve.axes[1].lines[0].get_ydata()), [1., .4])
        finally:
            curve.clear()

    def test_global_distribution_figures_use_reported_client_counts(self):
        self.results.class_distributions = {"0": {"train": {"A": 3}, "val": {"A": 1}},
                                            "1": {"train": {"A": 7}, "val": {"A": 2}}}
        figures = FederatedPlotter.plot_all(self.results)
        try:
            self.assertEqual(set(figures), {"data_distribution/train.png", "data_distribution/val.png"})
            self.assertEqual(len(figures["data_distribution/train.png"].axes), 2)
        finally:
            for figure in figures.values():
                figure.clear()

    @patch("mlflow_logger.MlflowClient")
    def test_publishing_retains_history_without_duplicate_metric_points(self, factory):
        client = factory.return_value
        client.get_run.return_value.info.experiment_id = "experiment"
        stored = {}
        client.get_metric_history.side_effect = lambda run_id, key: [
            SimpleNamespace(step=step, value=value) for step, value in stored.get(key, {}).items()
        ]
        def store(run_id, *, metrics, synchronous):
            for metric in metrics:
                stored.setdefault(metric.key, {})[metric.step] = metric.value
        client.log_batch.side_effect = store
        logger = MLflowLogger()
        self.add_round(1)
        logger.log_federated("experiment", "run", self.results)
        calls = client.log_batch.call_count
        logger.log_federated("experiment", "run", self.results)
        self.assertEqual(client.log_batch.call_count, calls)
        self.add_round(2)
        logger.log_federated("experiment", "run", self.results)
        self.assertEqual(set(stored["train.loss"]), {1, 2})
        self.assertEqual(set(stored["global.test.macro_f1"]), {1, 2})
        self.assertEqual(set(stored["clients.0.val.accuracy"]), {1, 2})
        report = json.loads(client.log_text.call_args.args[1])
        self.assertEqual(report["rounds_completed"], 2)
        self.assertEqual(set(report["per_client"]["train"]["0"]), {"1", "2"})
        artifact_names = {call.args[2] for call in client.log_figure.call_args_list}
        self.assertEqual(artifact_names, {"federated/metrics.png"})
        client.set_terminated.assert_not_called()
        self.results.finish(FederatedTrainingStatus.SUCCEEDED)
        logger.log_federated("experiment", "run", self.results)
        client.set_terminated.assert_called_once_with("run", status="FINISHED")
        self.assertEqual(vars(logger), {})

    def test_server_publishes_after_each_aggregation_and_at_completion(self):
        records = [RecordDict({"metrics": MetricRecord({
            "partition-id": 0, "server-round": 1, "num-examples": 8,
            "loss": .5, "accuracy": .6, "macro_f1": .5,
        })})]
        states = []
        with patch("fl_isic.tracking.FederatedTracking") as tracking_type, \
             patch.object(server_app, "GlobalModelEvaluator", return_value=SimpleNamespace(
                 evaluate=lambda _: {split: {"loss": .4, "accuracy": .7, "macro_f1": .6}
                                     for split in ("train", "val", "test")})), \
             patch.object(server_app, "ServerTrainingPlan", return_value=SimpleNamespace(
                 initialize_clients=lambda *args: {}, training_cfg=None,
                 class_distributions=lambda: {}, evaluation_datasets=lambda: {})), \
             patch.object(server_app, "MLflowLogger") as logger_type, \
             patch.object(server_app, "TrainingFedAvg") as strategy_type, \
             patch.object(server_app, "build_model", return_value=nn.Linear(2, 2)), \
             patch.object(server_app, "set_seed"), \
             patch.object(server_app.mlflow, "set_tracking_uri"):
            tracking_type.return_value.experiment_id = "experiment"
            tracking_type.return_value.run_id = "run"
            tracking_type.return_value.restore.return_value = None
            tracking_type.return_value.manifest = {"clients": {"0": "client-run"}}
            logger_type.return_value.log_federated.side_effect = lambda e, r, value: states.append(
                deepcopy(value.to_dict()))
            def start(**kwargs):
                callbacks = strategy_type.call_args.kwargs
                train = callbacks["train_metrics_aggr_fn"](records, "num-examples")
                validation = callbacks["evaluate_metrics_aggr_fn"](records, "num-examples")
                return Result(arrays=ArrayRecord({"weight": torch.ones(1)}),
                              train_metrics_clientapp={1: train}, evaluate_metrics_clientapp={1: validation})
            strategy_type.return_value.start.side_effect = start
            server_app.main(object(), SimpleNamespace(run_config={"num-server-rounds": 1}))
        self.assertEqual(len(states), 5)  # initial, client train/val, global evaluation, finished
        self.assertEqual(states[0]["rounds_completed"], 0)
        self.assertEqual(states[1]["train_metrics_clientapp"][1]["loss"], .5)
        self.assertEqual(states[2]["evaluate_metrics_clientapp"][1]["loss"], .5)
        self.assertEqual(states[-1]["status"], "succeeded")
        self.assertEqual(states[-1]["global_metrics"]["test"][1]["macro_f1"], .6)

    def test_resumed_server_starts_at_next_round_with_restored_weights(self):
        weights = {"weight": torch.tensor([7.])}
        with patch("fl_isic.tracking.FederatedTracking") as tracking_type, \
             patch.object(server_app, "GlobalModelEvaluator", return_value=SimpleNamespace(
                 evaluate=lambda _: {split: {"loss": .4, "accuracy": .7, "macro_f1": .6}
                                     for split in ("train", "val", "test")})), \
             patch.object(server_app, "ServerTrainingPlan", return_value=SimpleNamespace(
                 initialize_clients=lambda *args: {}, training_cfg=None,
                 class_distributions=lambda: {}, evaluation_datasets=lambda: {})), \
             patch.object(server_app, "MLflowLogger"), \
             patch.object(server_app, "TrainingFedAvg") as strategy_type, \
             patch.object(server_app, "build_model") as build, \
             patch.object(server_app, "set_seed"):
            tracking = tracking_type.return_value
            tracking.experiment_id, tracking.run_id = "experiment", "global-run"
            tracking.manifest = {"clients": {"0": "client-run"}}
            tracking.restore.return_value = {"round": 5, "state_dict": weights}
            def start(**kwargs):
                self.assertEqual(strategy_type.return_value.round_number, 6)
                torch.testing.assert_close(kwargs["initial_arrays"].to_torch_state_dict()["weight"], weights["weight"])
                record = RecordDict({"metrics": MetricRecord({"partition-id": 0,
                    "server-round": 6, "num-examples": 8, "loss": .5,
                    "accuracy": .6, "macro_f1": .5})})
                strategy_type.return_value.train_metrics_aggr_fn([record], "num-examples")
                return Result(arrays=ArrayRecord(weights))
            strategy_type.return_value.start.side_effect = start
            server_app.main(object(), SimpleNamespace(run_config={
                "num-server-rounds": 1, "resume-experiment-id": "experiment"}))
            build.assert_not_called()
            self.assertEqual(tracking.checkpoint.call_args.args[0], 6)
            tracking.finish.assert_called_once_with("FINISHED")

    def test_client_logs_each_local_epoch_into_existing_run(self):
        from fl_isic import client_app
        from fl_isic.protocol import RoundRequest
        message = object()
        prepared = client_app.PreparedClientTask(
            SimpleNamespace(tracking=SimpleNamespace(uri="http://tracking", run_id="client-run")),
            RoundRequest(0, 6, 42, 3, 0.), None, nn.Linear(2, 2), [1, 2], [])
        with patch.object(client_app, "Message"), \
             patch.object(client_app, "_prepare", return_value=prepared), \
             patch.object(client_app, "make_loader"), \
             patch.object(client_app, "class_counts", return_value={}), \
             patch.object(client_app, "scalar_metrics", return_value={"loss": .5}), \
             patch.object(client_app, "local_train") as local, \
             patch.object(client_app, "evaluate_model") as validation, \
             patch("mlflow.set_tracking_uri"), \
             patch.object(client_app, "MlflowClient") as client_type:
            client = client_type.return_value
            client.get_metric_history.side_effect = lambda run, key: [SimpleNamespace(step=15)] if key == "local_epoch" else []
            def train(*args, **kwargs):
                for epoch in range(1, 4):
                    kwargs["on_epoch"](epoch, SimpleNamespace(scalar_metrics=lambda: {"loss": .5}))
            local.side_effect = train
            client_app.train(message, object())
            calls = [call for call in client.log_metric.call_args_list if call.args[1] == "train/loss"]
            self.assertEqual([call.args[0] for call in calls], ["client-run"] * 3)
            self.assertEqual([call.kwargs["step"] for call in calls], [16, 17, 18])
            artifacts = [call.args[2] for call in client.log_figure.call_args_list]
            self.assertEqual(artifacts.count("training_metrics.png"), 3)
            validation.assert_called_once()
            self.assertTrue(any(call.args[1] == "val/loss" for call in client.log_metric.call_args_list))


if __name__ == "__main__":
    unittest.main()
