"""Versioned named payloads at the Flower Message/Context boundary.

Only JSON settings and stable image IDs cross the boundary; model weights stay
in Flower ArrayRecords. No runtime optimizer, transform, or dataset is pickled.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, MetricRecord, RecordDict


class Payload:
    def to_record(self) -> ConfigRecord:
        return ConfigRecord({"schema-version": 1, "type": type(self).__name__,
                             "payload": json.dumps(asdict(self), allow_nan=False)})

    @classmethod
    def from_record(cls, record):
        if record.get("schema-version") != 1 or record.get("type") != cls.__name__:
            raise ValueError(f"Unsupported {cls.__name__} payload")
        return cls.from_dict(json.loads(record["payload"]))

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass(frozen=True)
class ClientIdentity(Payload):
    client_id: int
    num_clients: int

    def __post_init__(self):
        if type(self.client_id) is not int or type(self.num_clients) is not int or not 0 <= self.client_id < self.num_clients:
            raise ValueError("Invalid client identity")

    @classmethod
    def from_context(cls, context):
        return cls(int(context.node_config["partition-id"]),
                   int(context.node_config["num-partitions"]))


@dataclass(frozen=True)
class SampleReference:
    sample_id: str
    label: int
    group: str

    def __post_init__(self):
        if not self.sample_id or Path(self.sample_id).name != self.sample_id or "\\" in self.sample_id:
            raise ValueError("Sample IDs must be image filenames, not filesystem paths")
        if type(self.label) is not int or self.label < 0:
            raise ValueError("Invalid sample label")


@dataclass(frozen=True)
class ClientDataAssignment:
    dataset_version: str
    train: list[SampleReference]
    val: list[SampleReference]

    @classmethod
    def from_dict(cls, data):
        return cls(data["dataset_version"],
                   [SampleReference(**sample) for sample in data["train"]],
                   [SampleReference(**sample) for sample in data["val"]])


@dataclass(frozen=True)
class TrackingSettings:
    experiment_id: str
    run_id: str
    uri: str


@dataclass(frozen=True)
class ClientInitialization(Payload):
    identity: ClientIdentity
    training_config: dict[str, Any]
    data_assignment: ClientDataAssignment
    tracking: TrackingSettings

    @classmethod
    def from_dict(cls, data):
        return cls(ClientIdentity(**data["identity"]), data["training_config"],
                   ClientDataAssignment.from_dict(data["data_assignment"]),
                   TrackingSettings(**data["tracking"]))

    def save(self, context):
        context.state["client-initialization"] = self.to_record()

    @classmethod
    def load(cls, context):
        if "client-initialization" not in context.state:
            raise ValueError("Client must be initialized before training or evaluation")
        return cls.from_record(context.state["client-initialization"])


@dataclass(frozen=True)
class InitializationAcknowledgement(Payload):
    client_id: int
    dataset_version: str


@dataclass(frozen=True)
class RoundRequest(Payload):
    client_id: int
    server_round: int
    seed: int
    local_epochs: int
    proximal_mu: float

    def __post_init__(self):
        if (type(self.client_id) is not int or self.client_id < 0
                or type(self.server_round) is not int or self.server_round < 1
                or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.local_epochs) is not int or self.local_epochs < 1
                or self.proximal_mu < 0):
            raise ValueError("Invalid round request")

    def to_content(self, arrays):
        return RecordDict({"arrays": arrays, "request": self.to_record()})

    @classmethod
    def from_message(cls, message):
        return cls.from_record(message.content["request"])


@dataclass
class ClientResponse:
    client_id: int
    server_round: int
    num_examples: int
    metrics: dict[str, float]
    distributions: dict[str, dict[str, int]] | None = None
    arrays: ArrayRecord | None = None
    validation_metrics: dict[str, float] | None = None
    num_validation_examples: int = 0

    def to_content(self):
        # Keep Flower's weighting key in its MetricRecord for built-in strategies.
        content = RecordDict({"metrics": MetricRecord({**self.metrics,
            "num-examples": self.num_examples, "partition-id": self.client_id,
            "server-round": self.server_round})})
        if self.distributions is not None:
            content["data-distribution"] = ConfigRecord({split: json.dumps(counts)
                for split, counts in self.distributions.items()})
        if self.validation_metrics is not None:
            content["local-validation"] = ConfigRecord({
                "metrics": json.dumps(self.validation_metrics, allow_nan=False),
                "num-examples": self.num_validation_examples})
        if self.arrays is not None:
            content["arrays"] = self.arrays
        return content

    @classmethod
    def from_content(cls, content):
        metrics = dict(content["metrics"])
        client_id = int(metrics.pop("partition-id"))
        server_round = int(metrics.pop("server-round"))
        num_examples = int(metrics.pop("num-examples"))
        distributions = ({split: json.loads(counts) for split, counts in
                          content["data-distribution"].items()}
                         if "data-distribution" in content else None)
        return cls(client_id, server_round, num_examples, metrics, distributions,
                   content.get("arrays"),
                   json.loads(content["local-validation"]["metrics"]) if "local-validation" in content else None,
                   int(content["local-validation"]["num-examples"]) if "local-validation" in content else 0)
