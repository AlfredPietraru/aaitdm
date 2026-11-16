# Federated configuration reconstruction

Run: `old_experiments/20261111-105920`.

This is a reconstruction from saved artifacts and source history, not an
original run-time YAML snapshot. Values are classified rather than silently
presented as recovered facts.

| Setting | Reconstructed value | Evidence status |
|---|---|---|
| Aggregation | FedAvg, proximal mu 0 | Recorded in results.json |
| Clients / rounds / local epochs | 4 / 10 / 2 | Recorded in results.json |
| Partitioning | Dirichlet, alpha 0.5, seed 42 | Recorded in results.json |
| Group fractions | 0.3 / 0.1 / 0.1 | Recorded in results.json |
| Batch size / workers / learning rate | 32 / 2 / 0.0001 | Recorded in results.json |
| Training/validation counts | 7,539 / 2,667 | Recorded and reproduced from the split algorithm |
| Resampling | None in the federated data path | Source-supported: ISICDataLoaders is constructed without oversample_classes in both the original committed and cached Flower app |
| Validation transforms | Resize 512, central crop 224, scale to [0,1], ImageNet normalization | Archived centralized pipeline; supported by final-model inference comparison |
| Loss | Multiclass focal, gamma 2, mean, no alpha weighting | Strong inference from final-model focal versus cross-entropy loss comparison; see federated_validation_recovery.json |
| Optimizer | AdamW | Working inference from the completed centralized baseline family and the user's description of the shared baseline; not identifiable from model weights |
| Weight decay | 0.001 | Inferred from the common value across archived optimizer trials |
| Training transforms | Resize 512, central crop 288, random crop 224, independent flips p=0.5, 90-degree rotation p=0.5, scaling and ImageNet normalization | Inferred: identical across the six archived centralized histories and consistent with Git history |
| Scheduler / early stopping | Not used in federated local training | Source-supported local_train loop |
| Optimizer state | Recreated each client training call | Source-supported local_train loop |

## What was checked

- The supplied experiment ZIP contains centralized MLflow artifacts only.
- The cached Flower app under `~/.flwr-aitdm/apps/` preserves the federated
  loading/training code, but reads the root YAML dynamically; it does not embed it.
- The original federated commit `f831911` and cached app both construct the
  data loader without resampling. The existing split cache is not available;
  reconstructed client sample counts are checked against the saved counts.
- Committed root YAML versions include SGD as well as older AdamW settings;
  therefore Git alone does not establish the optimizer of this remote run.
- `documentation/recover_federated_config.py` evaluates the saved final model
  on the reconstructed validation split, calculating both cross entropy and
  focal loss. This does not fit the model, select training settings, or change
  the original archive. Numerical replay differences must not be hidden.

The report uses this reconstruction instead of declaring all historical
parameters unknowable. Optimizer and training-only transformations remain
explicitly inferred. Saved weights cannot uniquely identify optimizer family,
weight decay, or augmentation history.

## Validation replay result

| Hospital | Saved loss | Replayed focal loss | Replayed cross entropy |
|---|---:|---:|---:|
| 0 | 1.332160 | 1.332362 | 1.667413 |
| 1 | 1.176792 | 1.176930 | 1.505371 |
| 2 | 0.258316 | 0.258476 | 0.475246 |
| 3 | 0.613748 | 0.613806 | 0.893771 |

Maximum absolute focal-loss discrepancy: 0.000202. Each of hospitals 0–2
differs by one correct prediction from the original accuracy; hospital 3's
accuracy and macro-F1 match. The small discrepancies are retained in the JSON;
their cause is not established. This strongly supports the hypothesized
focal-loss/validation pipeline, but is not an exact reproduction of the
historical execution environment. Cross entropy does not explain the saved
loss values under this replay.
