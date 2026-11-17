# Federated training

Edit `app-state/federated_configuration.yaml` in the repository root to configure
rounds, local epochs, hospital count, participation, FedAvg/FedProx, partitioning,
split fractions, batch size, learning rate, workers, and seed. These settings are
owned by the federated pipeline; they are not stored in `TrainingCFG` or duplicated
in `pyproject.toml`.

From the repository root:

```bash
./federated/run_federated.sh
NUM_CLIENTS=4 ./federated/run_federated.sh
```

The launcher reads the YAML and stages a temporary Flower app with those settings
declared in its generated `pyproject.toml`. Flower requires configuration keys
to be declared in the app manifest; the source TOML and YAML are not duplicated.
The bundle's `pyproject.toml` is regenerated on every launch and the temporary
bundle is deleted when the launcher exits. Edit only the YAML for training settings.
`min-nodes` sets the default hospital count; `NUM_CLIENTS` overrides that count and
the server's minimum nodes. The script accepts no command-line arguments; change
all other federated training settings in `app-state/federated_configuration.yaml`.
Set `VENV_BIN` to select the Python/Flower environment (the launcher otherwise
uses `.venv/bin`).

`pyproject.toml` retains package metadata, dependencies, and Flower entry points.
Launch through `run_federated.sh` so the generated manifest includes the YAML settings. The shared
`app-state/training_configuration.yaml` still supplies network settings such as
optimizer type, loss, and transforms; federated batch size, learning rate, workers,
and seed override those values. Federated split fractions are passed directly to
the data loader and are separate from `TrainingCFG`.

`fl_isic/federated_cfg.py` defines `FederatedCFG`. It validates and exposes all
federated execution settings as typed attributes, derives `split_ratios`, and
builds common FedAvg/FedProx strategy arguments. Aggregation callbacks are passed
to `cfg.strategy_kwargs(...)` at runtime. The YAML also controls evaluation
participation and minimum training/evaluation nodes. Server and client both
construct this class from Flower's resolved run configuration.

`fl_isic/federated_training_results.py` defines `FederatedTrainingResults`.
It retains the run configuration, aggregated train/validation/server metrics,
per-client metrics by round, final CPU model weights, timestamps, and terminal
status (`succeeded`, `failed`, or `interrupted`). Aggregation callbacks fill the
object while training runs, so partial histories are saved on failure or Ctrl+C.
Only metrics actually returned by clients are available; individual local epoch
histories are not currently sent to the server. Final weights are available when
Flower returns its `Result`. `FederatedPlotter(results).plot_all()` creates figures without writing files.
The server uses one stateless `MLflowLogger` instance to publish configuration,
aggregate/per-client metrics, complete JSON history, live plots, status, and final
weights to the `federated-train` MLflow experiment. `MLFLOW_TRACKING_URI` defaults
to `http://localhost:5000`.

Uploads happen after each training and validation aggregation and again on
completion, failure, or Ctrl+C. MLflow steps correspond to server rounds; clients
currently report their last local epoch, not every local epoch. The latest
`federated/metrics.png`, `federated/per_client.png`, and `federated/results.json`
artifacts are replaced as new results arrive. Final weights are stored as
`federated/global_model.pt`. Metrics already present on the server are not
re-logged. Completed run histories remain visible even if a later round fails.


`python3 main.py` automatically starts MLflow at `http://localhost:5000` and
waits until it is ready. The database and artifacts persist in `app-state/mlflow.db`
and `app-state/mlartifacts/`; startup logs go to `app-state/mlflow-server.log`.
The launcher stops the server it started when training exits. An existing healthy
server is reused and left running. `MLFLOW_TRACKING_URI` can select another server;
an unavailable remote server produces an error instead of starting one locally.

### Persistent experiments and resuming

From the repository root:

```bash
python3 main.py                    # new experiment, parent, and one child per hospital
python3 main.py <experiment_id>    # restore that experiment and continue training
python3 main.py --centralized      # original centralized baseline
```

The server prints the new MLflow experiment ID. Each client logs every training
epoch to its persistent child run. Steps append across rounds and restarts;
`server_round` and `local_epoch` identify the round and epoch within that round.
A retried incomplete round retains its earlier epoch logs and appends new points.

The parent stores the client/run mapping in `federated/identity.json`, global
checkpoints in `federated/checkpoints/`, and the last committed checkpoint pointer
in `federated/latest.json`. A checkpoint is committed after training aggregation
and round evaluation return successfully. Resume restores its global weights and
histories, then starts the next round. Each launch runs `num-server-rounds`
**additional** rounds. Configuration and client partition identities must match;
resume never silently creates a replacement experiment. Local optimizers are
initialized each round as in the original FedAvg/FedProx implementation.

Direct launcher users can set `RESUME_EXPERIMENT_ID=<id>` with
`bash federated/run_federated.sh`. Use one active execution per experiment.
