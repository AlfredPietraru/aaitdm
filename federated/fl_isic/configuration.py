"""Read federated YAML and generate the runtime Flower app configuration."""

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

if __package__:
    from .federated_cfg import FederatedCFG
else:
    from federated_cfg import FederatedCFG


def load_federated_configuration(path: Path) -> dict[str, bool | int | float | str]:
    """Load the flat scalar settings supported by Flower's run configuration."""
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not config:
        raise ValueError("Federated configuration must be a nonempty YAML mapping.")
    for key, value in config.items():
        if not isinstance(key, str) or not key or any(char.isspace() for char in key):
            raise ValueError("Federated configuration keys must be nonempty words.")
        if type(value) not in (bool, int, float, str):
            raise ValueError(f"Unsupported federated configuration value: {key}")
        # Reject NaN and Infinity, which are not portable Flower CLI values.
        json.dumps(value, allow_nan=False)
    if type(config.get("min-nodes")) is not int or config["min-nodes"] < 1:
        raise ValueError("min-nodes must be a positive integer.")
    return FederatedCFG(config).to_dict()


def flower_overrides(config: dict[str, bool | int | float | str]) -> list[str]:
    """Quote scalar values for Flower's TOML-based --run-config parser."""
    return [f"{key}={json.dumps(value, allow_nan=False)}" for key, value in config.items()]


def prepare_federated_app(
    app_dir: Path, output_dir: Path, config: dict[str, bool | int | float | str],
) -> None:
    """Stage a Flower app with YAML-derived settings declared in its manifest.

    Flower rejects overrides for undeclared keys. Generating the manifest in a
    temporary bundle keeps YAML authoritative without editing the source TOML.
    """
    resolved = FederatedCFG(config).to_dict()
    manifest = (app_dir / "pyproject.toml").read_text(encoding="utf-8")
    header = "[tool.flwr.app.config]"
    before, separator, after = manifest.partition(header)
    if separator and any(line.lstrip().startswith("[") for line in after.splitlines()):
        raise ValueError("The run-config section must be the last section in pyproject.toml.")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(app_dir / "fl_isic", output_dir / "fl_isic",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output_dir / "pyproject.toml").write_text(
        before.rstrip() + "\n\n" + header + "\n" + "\n".join(flower_overrides(resolved)) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prepare-app", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-clients", type=int)
    args = parser.parse_args()
    config = load_federated_configuration(args.config)
    config["resume-experiment-id"] = os.environ.get("RESUME_EXPERIMENT_ID", "")
    if args.num_clients is not None:
        config["min-nodes"] = args.num_clients
    if args.prepare_app is not None:
        if args.output_dir is None:
            parser.error("--prepare-app requires --output-dir")
        prepare_federated_app(args.prepare_app, args.output_dir, config)
    else:
        print("\n".join(flower_overrides(FederatedCFG(config).to_dict())))
