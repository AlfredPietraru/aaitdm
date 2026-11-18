"""Verify global evaluation uses aggregated weights and every fixed split."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from flwr.app import ArrayRecord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "federated"))
from fl_isic import task
from fl_isic.federated_cfg import FederatedCFG
from fl_isic.federated_plotter import FederatedPlotter
from fl_isic.federated_training_results import FederatedTrainingResults


class GlobalEvaluationTests(unittest.TestCase):
    def test_all_splits_use_aggregated_weights_and_evaluation_transforms(self):
        cfg = SimpleNamespace(val_transforms=object(), seed=42)
        datasets = {split: SimpleNamespace(samples=[split], groups=[split])
                    for split in ("train", "val", "test")}
        model = nn.Linear(1, 1, bias=False)
        arrays = ArrayRecord({"weight": torch.tensor([[3.]])})
        def evaluate(received_model, loader, received_cfg, label):
            torch.testing.assert_close(received_model.weight, torch.tensor([[3.]]))
            return SimpleNamespace(scalar_metrics=lambda: {"loss": .4, "accuracy": .8, "macro_f1": .7})
        with patch.object(task, "load_training_cfg", return_value=cfg), \
             patch.object(task, "load_split", return_value=datasets), \
             patch.object(task, "ISICDataset") as dataset_type, \
             patch.object(task, "make_loader") as loader, \
             patch.object(task, "build_model", return_value=model), \
             patch.object(task, "evaluate", side_effect=evaluate) as scoring:
            evaluator = task.GlobalModelEvaluator(FederatedCFG({}))
            metrics = evaluator.evaluate(arrays)
        self.assertEqual(set(metrics), {"train", "val", "test"})
        self.assertEqual(scoring.call_count, 3)
        self.assertTrue(all(call.kwargs["transform"] is cfg.val_transforms
                            for call in dataset_type.call_args_list))
        self.assertTrue(all(call.kwargs["shuffle"] is False for call in loader.call_args_list))

    def test_metrics_figure_has_only_three_loss_and_three_f1_curves(self):
        results = FederatedTrainingResults(FederatedCFG({}))
        for split in ("train", "val", "test"):
            results.global_metrics[split] = {2: {"loss": .6, "accuracy": .8, "macro_f1": .7},
                                            3: {"loss": .4, "accuracy": .9, "macro_f1": .85}}
        figure = FederatedPlotter.plot_metrics(results.global_metrics)
        try:
            self.assertEqual(len(figure.axes), 2)
            for axis in figure.axes:
                self.assertEqual([line.get_label() for line in axis.lines], ["train", "validation", "test"])
                self.assertTrue(all(list(line.get_xdata()) == [2, 3] for line in axis.lines))
            self.assertEqual(list(figure.axes[0].lines[0].get_ydata()), [.6, .4])
            self.assertEqual(list(figure.axes[1].lines[0].get_ydata()), [.7, .85])
        finally:
            figure.clear()
