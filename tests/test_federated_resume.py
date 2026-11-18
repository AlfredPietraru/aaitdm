"""Exercise durable identities/checkpoints with a real temporary MLflow store."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mlflow
import torch
from flwr.app import ArrayRecord

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "federated"))
from fl_isic.tracking import FederatedTracking


class ResumeTests(unittest.TestCase):
    def test_new_experiments_and_explicit_resume(self):
        old_uri = mlflow.get_tracking_uri()
        old_cwd = os.getcwd()
        with TemporaryDirectory() as directory:
            os.chdir(directory)
            mlflow.set_tracking_uri(f"sqlite:///{directory}/tracking.db")
            try:
                configuration = {"min-nodes": 4, "local-epochs": 3}
                tracking = FederatedTracking(configuration)
                second = FederatedTracking(configuration)
                self.assertNotEqual(tracking.experiment_id, second.experiment_id)
                self.assertEqual(len(tracking.manifest["clients"]), 4)
                runs = tracking.client.search_runs([tracking.experiment_id])
                self.assertEqual(len(runs), 5)
                self.assertEqual({run.info.run_name for run in runs},
                                 {"global", "client-0", "client-1", "client-2", "client-3"})
                self.assertTrue(all("mlflow.parentRunId" not in run.data.tags for run in runs))
                self.assertIsNone(tracking.restore())
                arrays = ArrayRecord({"weight": torch.tensor([1., 2.])})
                tracking.checkpoint(5, arrays)
                tracking.finish("FINISHED")
                # Also support resuming experiments created with the former layout.
                tracking.client.set_tag(tracking.run_id, "federated_role", "parent")
                for client_run_id in tracking.manifest["clients"].values():
                    tracking.client.set_tag(client_run_id, "mlflow.parentRunId", tracking.run_id)
                legacy_manifest = tracking.read_json("federated/identity.json")
                legacy_manifest["configuration"].update({"fraction-train": 1.0, "fraction-evaluate": 1.0,
                    "min-train-nodes": 1, "min-evaluate-nodes": 1})
                tracking.client.log_dict(tracking.run_id, legacy_manifest, "federated/identity.json")
                resumed = FederatedTracking(configuration, tracking.experiment_id)
                self.assertEqual(resumed.manifest["configuration"], configuration)
                self.assertEqual(resumed.run_id, tracking.run_id)
                self.assertEqual(resumed.manifest["clients"], tracking.manifest["clients"])
                runs = resumed.client.search_runs([tracking.experiment_id])
                self.assertEqual(len(runs), 5)
                self.assertTrue(all("mlflow.parentRunId" not in run.data.tags for run in runs))
                self.assertEqual(resumed.client.get_run(resumed.run_id).data.tags["federated_role"], "global")
                checkpoint = resumed.restore()
                self.assertEqual(checkpoint["round"], 5)
                torch.testing.assert_close(checkpoint["state_dict"]["weight"], torch.tensor([1., 2.]))
                # An upload failure must leave the previous committed round recoverable.
                with patch.object(resumed.client, "log_artifact", side_effect=RuntimeError("upload")):
                    with self.assertRaises(RuntimeError):
                        resumed.checkpoint(6, arrays)
                self.assertEqual(resumed.restore()["round"], 5)
                next_arrays = ArrayRecord({"weight": torch.tensor([3., 4.])})
                resumed.checkpoint(6, next_arrays)
                latest = resumed.restore()
                self.assertEqual(latest["round"], 6)
                torch.testing.assert_close(latest["state_dict"]["weight"], torch.tensor([3., 4.]))
                artifacts = resumed.client.list_artifacts(resumed.run_id, "federated")
                self.assertEqual({item.path for item in artifacts},
                                 {"federated/identity.json", "federated/global_model.pt"})
                with self.assertRaises(ValueError):
                    FederatedTracking({**configuration, "min-nodes": 3}, tracking.experiment_id)
            finally:
                mlflow.set_tracking_uri(old_uri)
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
