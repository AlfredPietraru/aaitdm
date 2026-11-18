# Federated training

Edit `app-state/federated_configuration.yaml` in the repository root to configure
rounds, local epochs, hospital count, FedAvg/FedProx, partitioning,
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
to `cfg.strategy_kwargs(...)` at runtime. Only the server constructs this class from Flower's resolved run configuration.

`fl_isic/federated_training_results.py` defines `FederatedTrainingResults`.
It retains the run configuration, aggregated train/validation/server metrics,
per-client metrics by round, final CPU model weights, timestamps, and terminal
status (`succeeded`, `failed`, or `interrupted`). Aggregation callbacks fill the
object while training runs, so partial histories are saved on failure or Ctrl+C.
Client round summaries and server evaluation measurements are kept separately.
Each client also logs its full local epoch history directly into its own run.
`FederatedPlotter.plot_all(results)` creates figures without writing files.
The stateless `FederatedPlotter` also exposes `plot_metrics(global_metrics, title)`,
`plot_client_training(client_id, histories)`, and `plot_distribution(distributions, split)`.
All plotting code lives in `federated_plotter.py`; callers supply the data and own
figure cleanup and MLflow uploads.
The server publishes configuration, metrics, complete JSON history, plots, and
status to the experiment's global run. The latest checkpoint is stored at
`federated/global_model.pt`. `MLFLOW_TRACKING_URI` defaults to
`http://localhost:5000`. Uploads happen after client aggregation, after global
model evaluation, and on completion, failure, or Ctrl+C. Global MLflow metric
steps correspond to federated rounds. Existing metric points are not re-logged.

`python3 main.py` automatically starts MLflow at `http://localhost:5000` and
waits until it is ready. The database and artifacts persist in `app-state/mlflow.db`
and `app-state/mlartifacts/`; startup logs go to `app-state/mlflow-server.log`.
The launcher stops the server it started when training exits. An existing healthy
server is reused and left running. `MLFLOW_TRACKING_URI` can select another server;
an unavailable remote server produces an error instead of starting one locally.

### Persistent experiments and resuming

From the repository root:

```bash
python3 main.py                    # new experiment with global and client-N runs
python3 main.py <experiment_id>    # restore that experiment and continue training
python3 main.py --centralized      # original centralized baseline
```

The server prints the new MLflow experiment ID. Each client logs every training
epoch to its persistent client run. Steps append across rounds and restarts;
`server_round` and `local_epoch` identify the round and epoch within that round.
A retried incomplete round retains its earlier epoch logs and appends new points.

The global run stores the client/run mapping in `federated/identity.json`, global
checkpoint in `federated/global_model.pt`, replacing that file after each completed
round. The checkpoint contains weights, round number, configuration, and histories. A checkpoint is committed after training aggregation
and round evaluation return successfully. Resume restores its global weights and
histories, then starts the next round. Each launch runs `num-server-rounds`
**additional** rounds. Configuration and client partition identities must match;
resume never silently creates a replacement experiment. Local optimizers are
initialized each round as in the original FedAvg/FedProx implementation.

Direct launcher users can set `RESUME_EXPERIMENT_ID=<id>` with
`bash federated/run_federated.sh`. Use one active execution per experiment.

All runs belong directly to the experiment, without a parent–child relationship.
Explicitly resuming an experiment created with the previous nested layout removes
its nesting tags while preserving run IDs, metrics, and checkpoints.

New launches retain only the latest global checkpoint, without separate model
artifacts for earlier rounds. Existing historical artifacts are left untouched;
older checkpoint layouts remain readable when resuming.

Each `client-N` run stores `training_metrics.png`, a two-panel accuracy/loss
figure updated every local epoch, plus `data_distribution/train.png` and
`data_distribution/val.png`. Its learning curves include existing logged epochs
when resuming. `data_distribution.json` stores the underlying per-class counts.
The global run keeps `federated/metrics.png` and comparison charts at
`federated/data_distribution/{train,val}.png`, with one panel per reporting client.
Bar labels show image-entry counts and percentages within that client's split;
all client panels in a comparison share class order and count scales. These
figures replace the earlier combined `per_client.png` chart for new logging.

### Global model scoring

After every aggregation the server loads the aggregated weights and evaluates
all three fixed splits itself. It needs access to the same ISIC dataset and split
configuration as the clients. Training-split evaluation uses validation transforms
and no shuffling. These are measurements of the global model, including macro F1
computed over each complete split, rather than averages of client metrics.
`federated/metrics.png` contains exactly two panels: train/validation/test loss,
and train/validation/test macro F1, against completed federated rounds. Accuracy
is also logged under `global.{train,val,test}.accuracy`; loss and macro F1 use the
same prefixes. Client averages remain separate from these global measurements.
Global evaluation histories are stored in the latest checkpoint for resuming.
Earlier checkpoints without these histories start the new curves at the first
round executed with this version. The test split is monitored every round as
requested; model selection should use validation metrics.

### Named client protocol and server-owned initialization

`fl_isic/protocol.py` defines versioned `ClientIdentity`, `ClientInitialization`,
`RoundRequest`, and `ClientResponse` structures. Flower messages contain JSON
records for metadata and `ArrayRecord` for weights. The server discovers each
node's logical identity, sends `query.initialize` once per client per launch, and
requires acknowledgements before selecting clients for training or evaluation.

Initialization contains the server-resolved `TrainingCFG` settings, train and
validation image IDs/labels/lesion groups, a dataset assignment version, and
MLflow experiment/run/URI settings. Clients store it in `Context.state` and only
reconstruct runtime components locally. They do not read `FederatedCFG`, training
YAML, or split configuration, and do not compute partitions. Each machine resolves
image filenames against its own `data/ISIC_2019_Training_Input` directory. Missing
assigned images fail initialization before training starts.

Round requests contain client ID, actual federated round, server-computed seed,
local epoch count, and proximal coefficient. They contain neither initialization
settings nor data assignments. Discovery is the only client operation that needs
`Context.node_config`; training and evaluation use the stored initialization.
`PreparedClientTask` replaces positional preparation tuples.

The global run saves `federated/training_plan.json` with resolved training
settings, full split definitions, and client assignments. Resume reuses that plan,
including for global evaluation, even if the centralized training YAML changes.
Older experiments without this artifact generate it once using their federated
settings and the currently available training YAML. Initialization is sent again
on a resumed launch because it starts new Flower client contexts.

### Required round sequence

All configured clients participate in every round. Participation fractions and
separate minimum training/evaluation counts are not configuration options. The
strategy requires the full configured client count for training and validation.
Historical experiment snapshots containing retired sampling settings remain
resumable; those settings are ignored when comparing saved configuration.

Each client trains for `local-epochs`, evaluates that locally trained model on
only its assigned validation data, logs `val/*` in its own run at the final local
epoch step, and returns both the model and validation results. The server requires
successful, valid replies from every initialized client and refuses partial
aggregation. Local validation uses the configured base loss without the FedProx
penalty. Built-in post-aggregation client evaluation is disabled internally
because validation now happens within the training request before replying.

After aggregation the server evaluates the global model on its complete fixed
train, validation, and test splits and updates the global curves/checkpoint.
Local validation measurements are kept separately from global model measurements.

### Server validation early stopping

`early-stopping-patience: 3` in `app-state/federated_configuration.yaml` stops
training after three consecutive federated rounds without a strictly lower
server validation loss. The first evaluated round establishes the baseline;
any new minimum resets patience. This setting is independent of centralized
training's early-stopping settings. `num-server-rounds` is an upper limit per
launch. Neither client metrics nor test results influence stopping.

The server evaluates the aggregated model, updates patience, saves the latest
checkpoint and histories, and stops before scheduling another client round.
Stopping is successful completion, with `stop_reason=early_stopping` and patience
state included in the global results. The latest model is retained, following the
latest-only model policy. Resume reconstructs patience from saved global
validation history; an already early-stopped experiment schedules no more rounds.
