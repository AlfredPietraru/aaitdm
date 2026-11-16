# Network training

Shared code for centralized and federated ISIC network training:

- `neural_network.py`: classifier, epoch training/evaluation, and early stopping.
- `training_configuration.py`: configuration objects and random seeds.
- `../app-state/training_configuration.yaml`: training settings, loaded outside the package.
- `losses.py`: focal loss.
- `../isic_dataloaders.py`: datasets, splits, loaders, and `ensure_isic_dataset`.
- `../main.py`: dataset preparation, `train_baseline_pipeline()`, and training entry point.
- `mlflow_logger.py`: experiment tracking.

From the repository root, activate the environment created by the startup script
and run the root entry point to check the dataset before starting training:

```bash
source venv/bin/activate
python main.py
```

`main.py` calls `ensure_isic_dataset()` from the root `isic_dataloaders.py`
module, downloading files only when the CSVs or referenced images are absent. Complete datasets are reused. Data loaders read local files without
triggering downloads. `train_baseline_pipeline()` in the same `main.py` builds
the data loaders and model, creates the MLflow run, and starts training.

Configuration and mutable MLflow storage live in `app-state/`:
`training_configuration.yaml`, `mlflow.db`, and `mlartifacts/`.

Start the local MLflow server in a separate terminal from the repository root:

```bash
source venv/bin/activate
cd app-state
mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000
```

The training pipeline connects to `http://localhost:5000` by default. Set
`MLFLOW_TRACKING_URI` before running `python main.py` to use another server.

Edit `app-state/training_configuration.yaml` to configure training. Dataset
images, ground-truth CSV, metadata CSV, and the downloaded archive live in the
repository's `data/` folder. Default dataset paths resolve from the package location.
Federated outputs remain in `federated/outputs`.

Import shared components through the package, for example:

```python
from net_training.neural_network import train_one_epoch, validate_one_epoch
from net_training.training_configuration import TrainingCFG
```

`train_one_epoch()` and `validate_one_epoch()` return mutable
`TrainingOneEpochResults` objects. `TrainingResults.add_epoch_results()` stores
independent snapshots for the entire process in `train_results`, `val_results`,
and `test_results`. `results.epoch` counts stored training epochs, and
`results.summary` is computed on demand. Status records success, failure, or user
interruption; completed epochs are retained after Ctrl+C.

The pipeline creates the MLflow experiment and run. One stateless `MLflowLogger`
instance publishes results through `logger.log(experiment_id, run_id, results)`.
Every logger method takes those same three arguments.

The experiment name is fixed to `independent-train`. The run name is the full
SHA-256 hash of the parsed training configuration, serialized as JSON with sorted
keys. YAML comments, whitespace, and mapping key order do not affect the hash;
transform list order does. Experiment and run name fields are excluded.

The centralized split is fixed to 80% training, 10% validation, and 10% test
in `main.py`'s `train_baseline_pipeline()`. Split ratios are supplied to data
loaders separately from `TrainingCFG` and are excluded from the configuration
hash. Analysis scripts require `--split-ratios TRAIN VAL TEST`; federated runs
use their existing fraction settings in `federated/pyproject.toml`.

Class resampling is controlled by `ISICDataLoaders(oversample_classes=...)`,
separately from training configuration. The baseline disables it in `main.py`;
analysis scripts enable it with `--oversample-classes`. Neither split ratios nor
resampling settings belong to `TrainingCFG` or its configuration hash.

`train(...)` creates its own results and returns `(model, results)`.
`train_baseline_pipeline()` returns the same pair. Ctrl+C also returns both,
with `results.status` set to `interrupted`; failures are logged and re-raised.

Validation and test loaders are optional in `train(...)` and default to `None`.
Without validation, all configured epochs run, final weights are retained, and
early stopping and validation-loss scheduler updates are skipped. Without a
test loader, no test evaluation runs. Pass the logger by keyword: `logger=logger`.
