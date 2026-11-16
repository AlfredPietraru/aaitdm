"""Regression checks for accumulated results and MLflow epoch inference."""

import json
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from net_training.mlflow_logger import MLflowLogger
from net_training.neural_network import TrainingResults, train, validate_one_epoch
from net_training.training_configuration import TrainingCFG


class TrainingResultsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = nn.Linear(4, 2)
        self.loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.tensor([0, 1] * 4)), batch_size=4,
        )
        self.cfg = TrainingCFG({"epochs": 3, "scheduler_type": None})

    def epoch_results(self):
        return validate_one_epoch(
            self.model, self.loader, self.cfg.loss_function, torch.device("cpu"),
            split="train",
        )

    def test_history_preserves_independent_mutable_epochs(self):
        epoch = self.epoch_results()
        results = TrainingResults(split="train")
        results.extend(epoch)
        first_loss = results.loss[0]
        epoch.loss[0] = 99.0
        epoch.confusion_matrix[0][0][0] = 99
        results.extend(self.epoch_results())
        self.assertEqual(results.epoch, 2)
        self.assertEqual(results.loss[0], first_loss)
        self.assertNotEqual(results.confusion_matrix[0][0][0], 99)
        results.accuracy.pop()
        with self.assertRaises(ValueError):
            _ = results.epoch

    @patch("net_training.mlflow_logger.MlflowClient")
    def test_logger_infers_epochs_without_duplicate_metric_points(self, factory):
        client = factory.return_value
        client.create_run.return_value.info.run_id = "test"
        results = self.epoch_results()
        results.extend(self.epoch_results())
        logger = MLflowLogger()
        with logger.run(self.cfg):
            logger.log_epoch_metrics(results)
            count = client.log_batch.call_count
            logger.log_epoch_metrics(results)
            self.assertEqual(client.log_batch.call_count, count)
            results.extend(self.epoch_results())
            logger.log_epoch_metrics(results)
        points = [metric for call in client.log_batch.call_args_list
                  for metric in call.kwargs["metrics"]]
        for key in ("train.loss", "train.accuracy", "epochs_completed"):
            self.assertEqual([point.step for point in points if point.key == key], [1, 2, 3])
        self.assertIsNone(logger._run_id)

    @patch.object(MLflowLogger, "log_model")
    @patch("net_training.mlflow_logger.MlflowClient")
    def test_training_exports_every_epoch_and_test_results(self, factory, log_model):
        client = factory.return_value
        client.create_run.return_value.info.run_id = "test"
        logger = MLflowLogger()
        train(self.model, self.cfg, self.loader, self.loader, self.loader, logger)
        report = json.loads(client.log_text.call_args.args[1])
        self.assertEqual(len(report["epochs"]), 3)
        self.assertEqual(report["test"]["entries"], 8)
        for split, report_key in (("train", "train"), ("val", "validation")):
            results = logger._results[split]
            self.assertEqual(results.epoch, 3)
            for index, epoch in enumerate(report["epochs"]):
                self.assertEqual(epoch[report_key]["loss"], results.loss[index])
                self.assertEqual(epoch[report_key]["accuracy"], results.accuracy[index])
                self.assertEqual(epoch[report_key]["confusion_matrix"],
                                 results.confusion_matrix[index])
        log_model.assert_called_once_with(self.model)


if __name__ == "__main__":
    unittest.main()
