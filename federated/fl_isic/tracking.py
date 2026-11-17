"""Durable MLflow identities and committed global-round checkpoints."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from mlflow.tracking import MlflowClient
import torch


class FederatedTracking:
    def __init__(self, configuration, experiment_id=""):
        self.client = MlflowClient()
        if experiment_id:
            experiment = self.client.get_experiment(str(experiment_id))
            if experiment.lifecycle_stage != "active":
                raise ValueError("Cannot resume a deleted experiment")
            parents = self.client.search_runs([str(experiment_id)],
                filter_string="tags.federated_role = 'parent'")
            if len(parents) != 1:
                raise ValueError("Expected exactly one federated parent run")
            self.experiment_id = str(experiment_id)
            self.run_id = parents[0].info.run_id
            self.manifest = self.read_json("federated/identity.json")
            if self.manifest["configuration"] != configuration:
                raise ValueError("Resume configuration differs from the saved experiment")
        else:
            self.experiment_id = self.client.create_experiment(f"federated-{uuid4().hex}")
            self.run_id = self.client.create_run(self.experiment_id,
                tags={"federated_role": "parent"}, run_name="global").info.run_id
            children = {}
            for partition in range(configuration["min-nodes"]):
                children[str(partition)] = self.client.create_run(self.experiment_id,
                    tags={"mlflow.parentRunId": self.run_id, "partition-id": str(partition)},
                    run_name=f"client-{partition}").info.run_id
            self.manifest = {"configuration": configuration, "clients": children}
            self.client.log_dict(self.run_id, self.manifest, "federated/identity.json")
        expected = {str(i) for i in range(configuration["min-nodes"])}
        if set(self.manifest["clients"]) != expected:
            raise ValueError("Saved client identities do not match the federation")
        for child_id in self.manifest["clients"].values():
            child = self.client.get_run(child_id)
            if (str(child.info.experiment_id) != str(self.experiment_id)
                    or child.data.tags.get("mlflow.parentRunId") != self.run_id):
                raise ValueError("Saved client run does not belong to this parent")
        for run_id in [self.run_id, *self.manifest["clients"].values()]:
            self.client.update_run(run_id, status="RUNNING")

    def read_json(self, path):
        with TemporaryDirectory() as directory:
            local = self.client.download_artifacts(self.run_id, path, directory)
            return json.loads(Path(local).read_text())

    def restore(self):
        # The pointer is published only after its immutable checkpoint upload succeeds.
        artifacts = self.client.list_artifacts(self.run_id, "federated")
        if not any(item.path == "federated/latest.json" for item in artifacts):
            return None
        pointer = self.read_json("federated/latest.json")
        with TemporaryDirectory() as directory:
            path = self.client.download_artifacts(self.run_id, pointer["checkpoint"], directory)
            return torch.load(path, map_location="cpu", weights_only=True)

    def checkpoint(self, round_number, arrays, results=None):
        checkpoint = {"round": round_number, "state_dict": arrays.to_torch_state_dict(),
                      "configuration": self.manifest["configuration"],
                      "results": results.to_dict() if results else {}}
        name = f"round-{round_number:06d}-{uuid4().hex}.pt"
        with TemporaryDirectory() as directory:
            path = Path(directory) / name
            torch.save(checkpoint, path)
            self.client.log_artifact(self.run_id, str(path), "federated/checkpoints")
        self.client.log_dict(self.run_id, {"round": round_number,
            "checkpoint": f"federated/checkpoints/{name}"}, "federated/latest.json")

    def finish(self, status):
        for run_id in [self.run_id, *self.manifest["clients"].values()]:
            self.client.set_terminated(run_id, status=status)
