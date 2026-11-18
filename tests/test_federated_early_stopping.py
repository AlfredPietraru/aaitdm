"""Server early stopping follows validation loss and configurable round patience."""
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import Result

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "federated"))
from fl_isic import server_app
from fl_isic.federated_cfg import FederatedCFG
from fl_isic.federated_training_results import FederatedTrainingResults


class FederatedEarlyStoppingTests(unittest.TestCase):
    def run_server(self, patience, validation_losses, saved=None):
        losses = iter(validation_losses)
        snapshots = []
        with patch("fl_isic.tracking.FederatedTracking") as tracking_type, \
             patch.object(server_app, "GlobalModelEvaluator") as evaluator_type, \
             patch.object(server_app, "ServerTrainingPlan") as plan_type, \
             patch.object(server_app, "MLflowLogger") as logger_type, \
             patch.object(server_app, "TrainingFedAvg") as strategy_type, \
             patch.object(server_app, "build_model", return_value=nn.Linear(1, 1)), \
             patch.object(server_app, "set_seed"):
            tracking = tracking_type.return_value
            tracking.experiment_id, tracking.run_id = "experiment", "global"
            tracking.restore.return_value = saved
            tracking.manifest = {"clients": {"0": "client"}}
            plan_type.return_value.class_distributions.return_value = {}
            strategy = strategy_type.return_value
            def start(**kwargs):
                # The loop must not schedule another round after patience expires.
                round_number = strategy.round_number
                callback = strategy.train_metrics_aggr_fn
                from fl_isic.protocol import ClientResponse
                callback([ClientResponse(0, round_number, 10,
                    {"loss": .1, "accuracy": .9, "macro_f1": .9}).to_content()], "num-examples")
                return Result(arrays=ArrayRecord({"weight": torch.tensor([float(round_number)])}))
            strategy.start.side_effect = start
            def evaluate(arrays):
                round_number = int(arrays.to_torch_state_dict()["weight"].item())
                # Test metrics keep improving even when validation stagnates.
                return {"train": {"loss": 1 / round_number, "accuracy": .8, "macro_f1": .7},
                        "val": {"loss": next(losses), "accuracy": .8, "macro_f1": .7},
                        "test": {"loss": 1 / round_number, "accuracy": .8, "macro_f1": .7}}
            evaluator_type.return_value.evaluate.side_effect = evaluate
            logger_type.return_value.log_federated.side_effect = lambda e, r, value: snapshots.append(deepcopy(value.to_dict()))
            server_app.main(object(), SimpleNamespace(run_config={"num-server-rounds": 10,
                "early-stopping-patience": patience}))
            return strategy.start.call_count, tracking, plan_type, snapshots

    def test_patience_is_configurable_and_improvement_resets_counter(self):
        for patience, expected in ((2, 5), (3, 6)):
            with self.subTest(patience=patience):
                calls, tracking, _, snapshots = self.run_server(patience, [.5, .6, .4, .4, .4, .4])
                self.assertEqual(calls, expected)
                self.assertEqual(tracking.checkpoint.call_args.args[0], expected)
                latest = snapshots[-1]
                self.assertEqual(latest["status"], "succeeded")
                self.assertEqual(latest["stop_reason"], "early_stopping")
                self.assertEqual(latest["early_stopping"]["bad_rounds"], patience)
                self.assertEqual(latest["early_stopping"]["best_round"], 3)
                tracking.finish.assert_called_once_with("FINISHED")

    def test_resume_preserves_validation_patience_and_does_not_schedule_past_stop(self):
        results = FederatedTrainingResults(FederatedCFG({}))
        results.global_metrics["val"] = {1: {"loss": .4}, 2: {"loss": .5}, 3: {"loss": .6}}
        saved = {"round": 3, "state_dict": {"weight": torch.tensor([3.])}, "results": results.to_dict()}
        calls, _, _, snapshots = self.run_server(3, [.7], saved)
        self.assertEqual(calls, 1)
        self.assertEqual(snapshots[-1]["early_stopping"]["bad_rounds"], 3)
        results.global_metrics["val"][4] = {"loss": .7}
        saved.update(round=4, results=results.to_dict())
        calls, tracking, plan, snapshots = self.run_server(3, [], saved)
        self.assertEqual(calls, 0)
        plan.assert_not_called()
        tracking.checkpoint.assert_not_called()
        self.assertEqual(snapshots[-1]["stop_reason"], "early_stopping")

    def test_patience_must_be_a_positive_integer(self):
        for value in (0, -1, 2.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                FederatedCFG({"early-stopping-patience": value})
