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
                self.assertIsNone(tracking.restore())
                arrays = ArrayRecord({"weight": torch.tensor([1., 2.])})
                tracking.checkpoint(5, arrays)
                tracking.finish("FINISHED")
                resumed = FederatedTracking(configuration, tracking.experiment_id)
                self.assertEqual(resumed.run_id, tracking.run_id)
                self.assertEqual(resumed.manifest["clients"], tracking.manifest["clients"])
                checkpoint = resumed.restore()
                self.assertEqual(checkpoint["round"], 5)
                torch.testing.assert_close(checkpoint["state_dict"]["weight"], torch.tensor([1., 2.]))
                # An upload failure must leave the previous committed round recoverable.
                with patch.object(resumed.client, "log_artifact", side_effect=RuntimeError("upload")):
                    with self.assertRaises(RuntimeError):
                        resumed.checkpoint(6, arrays)
                self.assertEqual(resumed.restore()["round"], 5)
                with self.assertRaises(ValueError):
                    FederatedTracking({**configuration, "min-nodes": 3}, tracking.experiment_id)
            finally:
                os.chdir(directory)
            mlflow.set_tracking_uri(old_uri)


if __name__ == "__main__":
    unittest.main()
