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
            global_runs = self.client.search_runs([str(experiment_id)],
                filter_string="tags.`mlflow.runName` = 'global'")
            if len(global_runs) != 1:
                raise ValueError("Expected exactly one global run in the experiment")
            self.experiment_id = str(experiment_id)
            self.run_id = global_runs[0].info.run_id
            self.manifest = self.read_json("federated/identity.json")
            # Historical experiments may contain retired sampling controls.
            retired = {"fraction-train", "fraction-evaluate", "min-train-nodes", "min-evaluate-nodes"}
            self.manifest["configuration"] = {key: value for key, value in
                self.manifest["configuration"].items() if key not in retired}
            if "early-stopping-patience" in configuration:
                self.manifest["configuration"].setdefault("early-stopping-patience", 3)
            if self.manifest["configuration"] != configuration:
                raise ValueError("Resume configuration differs from the saved experiment")
        else:
            self.experiment_id = self.client.create_experiment(f"federated-{uuid4().hex}")
            self.run_id = self.client.create_run(self.experiment_id,
                tags={"federated_role": "global"}, run_name="global").info.run_id
            client_runs = {}
            for partition in range(configuration["min-nodes"]):
                client_runs[str(partition)] = self.client.create_run(self.experiment_id,
                    tags={"federated_role": "client", "partition-id": str(partition)},
                    run_name=f"client-{partition}").info.run_id
            self.manifest = {"configuration": configuration, "clients": client_runs}
            self.client.log_dict(self.run_id, self.manifest, "federated/identity.json")
        expected = {str(i) for i in range(configuration["min-nodes"])}
        if set(self.manifest["clients"]) != expected:
            raise ValueError("Saved client identities do not match the federation")
        for partition, client_run_id in self.manifest["clients"].items():
            client_run = self.client.get_run(client_run_id)
            if (str(client_run.info.experiment_id) != str(self.experiment_id)
                    or client_run.data.tags.get("partition-id") != partition):
                raise ValueError("Saved client run does not match its experiment and partition")
        # Explicitly resumed experiments from the previous layout keep their IDs
        # and history, but lose the MLflow nesting tag.
        for client_run_id in self.manifest["clients"].values():
            client_run = self.client.get_run(client_run_id)
            if "mlflow.parentRunId" in client_run.data.tags:
                self.client.delete_tag(client_run_id, "mlflow.parentRunId")
            self.client.set_tag(client_run_id, "federated_role", "client")
        self.client.set_tag(self.run_id, "federated_role", "global")
        for run_id in [self.run_id, *self.manifest["clients"].values()]:
            self.client.update_run(run_id, status="RUNNING")

    def read_json(self, path):
        with TemporaryDirectory() as directory:
            local = self.client.download_artifacts(self.run_id, path, directory)
            return json.loads(Path(local).read_text())

    def restore(self):
        artifacts = self.client.list_artifacts(self.run_id, "federated")
        paths = {item.path for item in artifacts}
        if "federated/global_model.pt" in paths:
            checkpoint_path = "federated/global_model.pt"
        elif "federated/latest.json" in paths:
            # Read checkpoints written before the latest-only storage policy.
            checkpoint_path = self.read_json("federated/latest.json")["checkpoint"]
        else:
            return None
        with TemporaryDirectory() as directory:
            path = self.client.download_artifacts(self.run_id, checkpoint_path, directory)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if "round" in checkpoint and "state_dict" in checkpoint:
                return checkpoint
            # Older runs also stored a weights-only file at global_model.pt.
            if "federated/latest.json" not in paths:
                raise ValueError("Global model has no round metadata for resuming")
            legacy_path = self.read_json("federated/latest.json")["checkpoint"]
            path = self.client.download_artifacts(self.run_id, legacy_path, directory)
            return torch.load(path, map_location="cpu", weights_only=True)

    def checkpoint(self, round_number, arrays, results=None):
        checkpoint = {"round": round_number, "state_dict": arrays.to_torch_state_dict(),
                      "configuration": self.manifest["configuration"],
                      "results": results.to_dict() if results else {}}
        name = "global_model.pt"
        with TemporaryDirectory() as directory:
            path = Path(directory) / name
            torch.save(checkpoint, path)
            self.client.log_artifact(self.run_id, str(path), "federated")

    def finish(self, status):
        for run_id in [self.run_id, *self.manifest["clients"].values()]:
            self.client.set_terminated(run_id, status=status)
