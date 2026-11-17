"""Manage the local tracking server for the lifetime of a training launch."""

from contextlib import contextmanager
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def ready(uri: str) -> bool:
    try:
        with urlopen(uri.rstrip("/") + "/health", timeout=1) as response:
            return response.status == 200 and response.read().strip() == b"OK"
    except (URLError, OSError):
        return False


@contextmanager
def tracking_server(state_dir: Path):
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("MLFLOW_TRACKING_URI must be an HTTP(S) server URL")
    if ready(uri):
        print(f"Using MLflow server at {uri}", flush=True)
        yield uri
        return
    if parsed.hostname not in ("localhost", "127.0.0.1") or parsed.scheme != "http" or parsed.path not in ("", "/"):
        raise RuntimeError(f"MLflow server at {uri} is unavailable; only local HTTP servers can be started automatically")
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    log_path = state_dir / "mlflow-server.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([
            sys.executable, "-m", "mlflow", "server",
            "--backend-store-uri", f"sqlite:///{state_dir / 'mlflow.db'}",
            "--artifacts-destination", str(state_dir / "mlartifacts"),
            "--host", "127.0.0.1", "--port", str(parsed.port or 80),
        ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            while not ready(uri):
                if process.poll() is not None:
                    raise RuntimeError(f"MLflow exited during startup; see {log_path}")
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"MLflow startup timed out; see {log_path}")
                time.sleep(0.25)
            print(f"MLflow ready at {uri}; logs: {log_path}", flush=True)
            yield uri
        finally:
            # MLflow may spawn server workers; shut down the process group we own.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
