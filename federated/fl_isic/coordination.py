"""Server-owned training plan, discovery, initialization, and round addressing."""
from dataclasses import asdict
from hashlib import sha256
import json
import time

from flwr.app import Message, RecordDict

from fl_isic.protocol import (
    ClientIdentity, ClientInitialization, ClientDataAssignment,
    InitializationAcknowledgement, RoundRequest, SampleReference, TrackingSettings, ClientResponse,
)
from fl_isic.task import load_training_cfg, load_split, partition
from net_training.training_configuration import TrainingCFG


class ServerTrainingPlan:
    """Persist exact settings/assignments independently of future YAML changes."""
    artifact = "federated/training_plan.json"

    def __init__(self, cfg, tracking):
        self.cfg = cfg
        self.tracking = tracking
        paths = {item.path for item in tracking.client.list_artifacts(tracking.run_id, "federated")}
        if self.artifact in paths:
            self.data = tracking.read_json(self.artifact)
        else:
            training = load_training_cfg(cfg)
            datasets = load_split(training, cfg.split_ratios)
            train, val = partition(datasets, cfg.min_nodes, cfg.partitioning,
                                   cfg.dirichlet_alpha, training.seed)
            splits = {split: [asdict(SampleReference(path.name, int(label), str(group)))
                      for (path, label), group in zip(dataset.samples, dataset.groups)]
                      for split, dataset in datasets.items()}
            version = sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
            assignments = {str(client): asdict(ClientDataAssignment(version,
                [SampleReference(**splits["train"][i]) for i in train[client]],
                [SampleReference(**splits["val"][i]) for i in val[client]]))
                for client in range(cfg.min_nodes)}
            self.data = {"training_config": training.to_dict(), "splits": splits,
                         "assignments": assignments}
            tracking.client.log_dict(tracking.run_id, self.data, self.artifact)
        if set(self.data["assignments"]) != set(tracking.manifest["clients"]):
            raise ValueError("Saved data assignments do not match client identities")
        self.training_cfg = TrainingCFG(self.data["training_config"])

    def initialization(self, client_id):
        return ClientInitialization(ClientIdentity(client_id, self.cfg.min_nodes),
            self.data["training_config"],
            ClientDataAssignment.from_dict(self.data["assignments"][str(client_id)]),
            TrackingSettings(str(self.tracking.experiment_id),
                self.tracking.manifest["clients"][str(client_id)], self.tracking_uri))

    def evaluation_datasets(self):
        from fl_isic.task import dataset_from_references
        return {split: dataset_from_references([SampleReference(**item) for item in samples],
                                               self.training_cfg.val_transforms)
                for split, samples in self.data["splits"].items()}

    def class_distributions(self):
        from collections import Counter
        from isic_dataloaders import CLASS_NAMES
        distributions = {}
        for client, assignment in self.data["assignments"].items():
            distributions[client] = {}
            for split in ("train", "val"):
                counts = Counter(sample["label"] for sample in assignment[split])
                distributions[client][split] = {name: counts.get(label, 0)
                    for label, name in enumerate(CLASS_NAMES)}
        return distributions

    def initialize_clients(self, grid, tracking_uri, timeout=120):
        self.tracking_uri = tracking_uri
        deadline = time.monotonic() + timeout
        while len(nodes := list(grid.get_node_ids())) < self.cfg.min_nodes:
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for clients to initialize")
            time.sleep(.25)
        if len(nodes) != self.cfg.min_nodes:
            raise ValueError("Connected client count differs from the configured client count")
        replies = list(grid.send_and_receive([
            Message(content=RecordDict(), dst_node_id=node, message_type="query.identity")
            for node in nodes], timeout=timeout))
        identities = {}
        for reply in replies:
            if reply.has_error():
                raise RuntimeError(f"Client discovery failed: {reply.error}")
            identity = ClientIdentity.from_record(reply.content["identity"])
            if identity.num_clients != self.cfg.min_nodes or identity.client_id in identities.values():
                raise ValueError("Duplicate or inconsistent client identity")
            identities[reply.metadata.src_node_id] = identity.client_id
        if set(identities) != set(nodes):
            raise RuntimeError("Did not discover every client")
        replies = list(grid.send_and_receive([
            Message(content=RecordDict({"initialization": self.initialization(client).to_record()}),
                    dst_node_id=node, message_type="query.initialize")
            for node, client in identities.items()], timeout=timeout))
        acknowledged = set()
        for reply in replies:
            if reply.has_error():
                raise RuntimeError(f"Client initialization failed: {reply.error}")
            ack = InitializationAcknowledgement.from_record(reply.content["acknowledgement"])
            node = reply.metadata.src_node_id
            if (node not in identities or ack.client_id != identities[node]
                    or ack.dataset_version != self.data["assignments"][str(ack.client_id)]["dataset_version"]):
                raise ValueError("Invalid initialization acknowledgement")
            acknowledged.add(node)
        if acknowledged != set(nodes):
            raise RuntimeError("Not every client acknowledged initialization")
        return identities


class NamedRequestStrategy:
    """Mixin personalizing built-in Flower strategy messages after node sampling."""
    def __init__(self, *args, node_clients, training_plan, **kwargs):
        self.node_clients = node_clients
        self.training_plan = training_plan
        self.round_number = 0
        kwargs.update(fraction_train=1.0, fraction_evaluate=0.0,
                      min_train_nodes=len(node_clients), min_available_nodes=len(node_clients))
        super().__init__(*args, **kwargs)

    def _requests(self, messages):
        for message in messages:
            node = message.metadata.dst_node_id
            if node not in self.node_clients:
                raise ValueError("An uninitialized client was selected")
            client = self.node_clients[node]
            cfg = self.training_plan.cfg
            seed = (self.training_plan.training_cfg.seed + 1000 * self.round_number + client) % 2**32
            request = RoundRequest(client, self.round_number, seed, cfg.local_epochs, cfg.proximal_mu)
            # New messages avoid mutating the RecordDict shared by built-in strategies.
            yield Message(content=request.to_content(message.content[self.arrayrecord_key]),
                          dst_node_id=node, message_type=message.metadata.message_type,
                          group_id=message.metadata.group_id, ttl=message.metadata.ttl)

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        expected = set(self.node_clients)
        sources = [reply.metadata.src_node_id for reply in replies]
        if len(sources) != len(expected) or set(sources) != expected or any(reply.has_error() for reply in replies):
            raise RuntimeError("Round requires successful training and validation replies from every client")
        valid, errors = self._check_and_log_replies(replies, is_train=True)
        if errors or len(valid) != len(expected):
            raise RuntimeError("Cannot aggregate a round with invalid client models")
        for reply in valid:
            response = ClientResponse.from_content(reply.content)
            if (response.client_id != self.node_clients[reply.metadata.src_node_id]
                    or response.server_round != self.round_number
                    or response.num_examples <= 0 or response.arrays is None
                    or response.validation_metrics is None or response.num_validation_examples <= 0):
                raise RuntimeError("Client reply must include this round's trained model and local validation")
        return super().aggregate_train(server_round, valid)

    def configure_train(self, server_round, arrays, config, grid):
        return self._requests(super().configure_train(server_round, arrays, config, grid))

    def configure_evaluate(self, server_round, arrays, config, grid):
        return self._requests(super().configure_evaluate(server_round, arrays, config, grid))
