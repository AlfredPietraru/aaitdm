"""Exercise initialization, persistent plans, and personalized Flower requests."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from flwr.app import ArrayRecord, Context, Message, RecordDict
from flwr.supercore.task_identity import TaskIdentity

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "federated"))
from fl_isic import client_app, coordination, task
from fl_isic.protocol import (ClientIdentity, ClientInitialization, ClientDataAssignment,
    SampleReference, TrackingSettings, RoundRequest, ClientResponse)
from fl_isic.federated_cfg import FederatedCFG
from fl_isic.server_app import TrainingFedAvg, TrainingFedProx, recording_aggregator
from fl_isic.federated_training_results import FederatedTrainingResults
from net_training.training_configuration import TrainingCFG


class LocalGrid:
    """Deliver real Flower messages to the handlers without starting processes."""
    def __init__(self):
        self.contexts = {node: Context(run_id=1, node_id=node,
            node_config={"partition-id": client, "num-partitions": 2},
            state=RecordDict(), run_config={}) for client, node in enumerate((101, 202))}
        self.sent = []
        self.pending = []
        self.counter = 0

    def get_node_ids(self):
        return list(self.contexts)

    def send_and_receive(self, messages, *, timeout=None):
        for message in messages:
            self.sent.append(message)
            yield client_app.app(message, self.contexts[message.metadata.dst_node_id])

    def push_messages(self, messages):
        messages = list(messages)
        for message in messages:
            self.counter += 1
            message.metadata.__dict__["_message_id"] = str(self.counter)
        self.pending.extend(self.send_and_receive(messages))
        return [message.metadata.message_id for message in messages]

    def pull_messages(self, message_ids):
        replies = [reply for reply in self.pending if reply.metadata.reply_to_message_id in message_ids]
        self.pending = [reply for reply in self.pending if reply not in replies]
        return replies


class ProtocolTests(unittest.TestCase):
    def test_full_participation_is_mandatory(self):
        for key in ("fraction-train", "fraction-evaluate", "min-train-nodes", "min-evaluate-nodes"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unknown federated settings"):
                FederatedCFG({key: 1})
        cfg = FederatedCFG({"min-nodes": 2})
        self.assertFalse(any(key in cfg.to_dict() for key in
            ("fraction-train", "fraction-evaluate", "min-train-nodes", "min-evaluate-nodes")))
        plan = SimpleNamespace(cfg=cfg, training_cfg=SimpleNamespace(seed=42))
        strategy = TrainingFedAvg(node_clients={101: 0, 202: 1}, training_plan=plan)
        self.assertEqual(strategy.min_train_nodes, 2)
        self.assertEqual(strategy.fraction_train, 1.)
        self.assertEqual(strategy.fraction_evaluate, 0.)
        with self.assertRaisesRegex(RuntimeError, "every client"):
            strategy.aggregate_train(1, [])
        with patch.object(TaskIdentity, "_run_id", 1), \
             patch.object(TaskIdentity, "_task_id", 1), \
             patch.object(TaskIdentity, "_node_id", 0):
            strategy.round_number = 1
            arrays = ArrayRecord({"weight": torch.tensor([[1.]])})
            replies = []
            for node, client_id in strategy.node_clients.items():
                request = Message(content=RecordDict(), dst_node_id=node, message_type="train")
                response = ClientResponse(client_id, 1, 8,
                    {"loss": .5, "accuracy": .8, "macro_f1": .7}, arrays=arrays)
                replies.append(Message(content=response.to_content(), reply_to=request))
            with self.assertRaisesRegex(RuntimeError, "local validation"):
                strategy.aggregate_train(1, replies)


    def test_assignment_loader_supports_flat_and_nested_archive_layouts(self):
        reference = SampleReference("ISIC_0000022_downsampled.jpg", 0, "lesion-1")
        for nested in (False, True):
            with self.subTest(nested=nested), TemporaryDirectory() as directory:
                root = Path(directory) / "ISIC_2019_Training_Input"
                if nested:
                    root /= "ISIC_2019_Training_Input"
                root.mkdir(parents=True)
                image = root / reference.sample_id
                image.touch()
                transform = object()
                with patch.object(task, "DATA_DIR", Path(directory)):
                    dataset = task.dataset_from_references([reference], transform)
                    self.assertEqual(dataset.samples, ((image, 0),))
                    self.assertEqual(dataset.groups, ("lesion-1",))
                    self.assertIs(dataset.transform, transform)
                    with self.assertRaisesRegex(FileNotFoundError, "missing locally"):
                        task.dataset_from_references([SampleReference("missing.jpg", 0, "lesion-2")], transform)

    def test_schema_roundtrip_and_reject_uninitialized_or_wrong_version(self):
        cfg = TrainingCFG({"optimizer": "SGD", "optimizer_kwargs": {"lr": .02}})
        settings = cfg.to_dict()
        self.assertEqual(TrainingCFG(settings).to_dict(), settings)
        initialization = ClientInitialization(ClientIdentity(0, 2), settings,
            ClientDataAssignment("v1", [SampleReference("image.jpg", 0, "lesion")], []),
            TrackingSettings("experiment", "run", "http://tracking"))
        self.assertEqual(ClientInitialization.from_record(initialization.to_record()), initialization)
        ctx = SimpleNamespace(state=RecordDict())
        with self.assertRaisesRegex(ValueError, "initialized"):
            ClientInitialization.load(ctx)
        initialization.save(ctx)
        self.assertEqual(ClientInitialization.load(ctx), initialization)
        record = RoundRequest(0, 4, 4042, 2, 0.).to_record()
        record["schema-version"] = 9
        with self.assertRaises(ValueError):
            RoundRequest.from_record(record)
        with self.assertRaises(ValueError):
            SampleReference("../image.jpg", 0, "lesion")
        response = ClientResponse(0, 4, 8, {"loss": .5}, {"train": {"A": 8}})
        self.assertEqual(ClientResponse.from_content(response.to_content()), response)

    def test_duplicate_client_id_is_rejected_before_initialization(self):
        with patch.object(TaskIdentity, "_run_id", 1), \
             patch.object(TaskIdentity, "_task_id", 1), \
             patch.object(TaskIdentity, "_node_id", 0):
            grid = LocalGrid()
            grid.contexts[202].node_config["partition-id"] = 0
            plan = coordination.ServerTrainingPlan.__new__(coordination.ServerTrainingPlan)
            plan.cfg = FederatedCFG({"min-nodes": 2})
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                plan.initialize_clients(grid, "http://tracking")
            self.assertTrue(all(message.metadata.message_type == "query.identity" for message in grid.sent))

    def test_server_initializes_once_and_sends_personalized_rounds(self):
        with TemporaryDirectory() as directory, \
             patch.object(TaskIdentity, "_run_id", 1), \
             patch.object(TaskIdentity, "_task_id", 1), \
             patch.object(TaskIdentity, "_node_id", 0):
            image_root = Path(directory) / "ISIC_2019_Training_Input"
            image_root.mkdir()
            datasets = {}
            for split, count in (("train", 64), ("val", 16), ("test", 8)):
                samples = []
                for i in range(count):
                    path = image_root / f"{split}-{i}.jpg"
                    path.touch()
                    samples.append((path, 0))
                datasets[split] = SimpleNamespace(samples=samples,
                    groups=[f"{split}-{i}" for i in range(count)])
            cfg = FederatedCFG({"min-nodes": 2, "partitioning": "iid", "num-workers": 0})
            training = TrainingCFG({"num_workers": 0})
            artifacts = {}
            tracking = SimpleNamespace(client=Mock(), run_id="global", experiment_id="experiment",
                manifest={"clients": {"0": "client-run-0", "1": "client-run-1"}},
                read_json=lambda path: artifacts[path])
            tracking.client.list_artifacts.side_effect = lambda *args: [SimpleNamespace(path=p) for p in artifacts]
            tracking.client.log_dict.side_effect = lambda run, data, path: artifacts.__setitem__(path, data)
            mlflow_client = Mock()
            mlflow_client.get_metric_history.return_value = []
            with patch.object(coordination, "load_training_cfg", return_value=training) as load_cfg, \
                 patch.object(coordination, "load_split", return_value=datasets) as load_split, \
                 patch.object(task, "DATA_DIR", Path(directory)), \
                 patch.object(client_app, "MlflowClient", return_value=mlflow_client), \
                 patch.object(client_app.mlflow, "set_tracking_uri"), \
                 patch.object(client_app, "build_model", side_effect=lambda: nn.Linear(1, 1, bias=False)), \
                 patch.object(client_app, "local_train", return_value=object()), \
                 patch.object(client_app, "evaluate_model", return_value=object()), \
                 patch.object(client_app, "scalar_metrics", return_value={"loss": .5, "accuracy": .8, "macro_f1": .7}), \
                 patch.object(client_app, "set_seed") as seed:
                plan = coordination.ServerTrainingPlan(cfg, tracking)
                grid = LocalGrid()
                mapping = plan.initialize_clients(grid, "http://tracking")
                self.assertEqual(mapping, {101: 0, 202: 1})
                self.assertEqual([msg.metadata.message_type for msg in grid.sent],
                                 ["query.identity"] * 2 + ["query.initialize"] * 2)
                # No later handler depends on node_config or run_config.
                for ctx in grid.contexts.values():
                    ctx.node_config.clear()
                arrays = ArrayRecord({"weight": torch.tensor([[2.]])})
                results = FederatedTrainingResults(cfg)
                for strategy_type in (TrainingFedAvg, TrainingFedProx):
                    strategy = strategy_type(node_clients=mapping, training_plan=plan,
                        min_available_nodes=2, min_train_nodes=2,
                        train_metrics_aggr_fn=recording_aggregator(results, "train"))
                    strategy.round_number = 6
                    output = strategy.start(grid=grid, initial_arrays=arrays, num_rounds=1,
                                            timeout=5)
                    torch.testing.assert_close(output.arrays.to_torch_state_dict()["weight"], torch.tensor([[2.]]))
                train_messages = [message for message in grid.sent if message.metadata.message_type == "train"]
                self.assertEqual(len(train_messages), 4)
                self.assertFalse(any(message.metadata.message_type == "evaluate" for message in grid.sent))
                self.assertEqual(set(results.per_client["val"]), {0, 1})
                self.assertEqual(results.per_client["val"][0][6]["num-examples"], 8)
                for message in train_messages:
                    self.assertEqual(set(message.content), {"arrays", "request"})
                    request = RoundRequest.from_message(message)
                    self.assertEqual(request.server_round, 6)
                    self.assertEqual(request.seed, training.seed + 6000 + mapping[message.metadata.dst_node_id])
                self.assertEqual(seed.call_count, 4)  # validation runs inside each client training task
                self.assertEqual(load_cfg.call_count, 1)
                self.assertEqual(load_split.call_count, 1)
                # Resume reads the saved resolved settings/assignments, even if YAML changed.
                load_cfg.side_effect = AssertionError("Must not reload training YAML")
                load_split.side_effect = AssertionError("Must not recompute dataset splits")
                resumed = coordination.ServerTrainingPlan(cfg, tracking)
                self.assertEqual(resumed.data, plan.data)
                self.assertEqual(len(mlflow_client.log_dict.call_args_list), 2)
                self.assertEqual(len([message for message in grid.sent
                    if message.metadata.message_type == "query.initialize"]), 2)
