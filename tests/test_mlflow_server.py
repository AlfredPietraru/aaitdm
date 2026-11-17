"""Verify startup ownership, readiness, and failure cleanup."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from mlflow_server import tracking_server


class TrackingServerTests(unittest.TestCase):
    @patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://localhost:5000"})
    def test_existing_server_is_reused_without_shutdown(self):
        with patch("mlflow_server.ready", return_value=True), \
             patch("mlflow_server.subprocess.Popen") as spawn, \
             patch("mlflow_server.os.killpg") as kill:
            with tracking_server(Path("unused")) as uri:
                self.assertEqual(uri, "http://localhost:5000")
            spawn.assert_not_called()
            kill.assert_not_called()

    @patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://localhost:5001"})
    def test_launch_waits_and_stops_owned_server_even_on_training_error(self):
        process = Mock(pid=123)
        process.poll.return_value = None
        with TemporaryDirectory() as directory, \
             patch("mlflow_server.ready", side_effect=[False, False, True]), \
             patch("mlflow_server.time.sleep"), \
             patch("mlflow_server.subprocess.Popen", return_value=process) as spawn, \
             patch("mlflow_server.os.killpg") as kill:
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with tracking_server(Path(directory)):
                    raise RuntimeError("training failed")
            command = spawn.call_args.args[0]
            self.assertIn(f"sqlite:///{directory}/mlflow.db", command)
            self.assertEqual(command[-1], "5001")
            kill.assert_called_once()
            process.wait.assert_called_once()

    @patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://localhost:5000"})
    def test_startup_failure_prevents_training_and_cleans_up(self):
        process = Mock(pid=123)
        process.poll.return_value = 1
        with TemporaryDirectory() as directory, \
             patch("mlflow_server.ready", return_value=False), \
             patch("mlflow_server.subprocess.Popen", return_value=process), \
             patch("mlflow_server.os.killpg"):
            with self.assertRaisesRegex(RuntimeError, "exited during startup"):
                with tracking_server(Path(directory)):
                    self.fail("Training must not start")
            process.wait.assert_called_once()
