# Stage 1 report

`report/report.tex` is the four-page English report; `report/report.pdf` is the compiled version.
`presentation/presentation.tex` is an eleven-slide, widescreen Beamer presentation;
`presentation/presentation.pdf` is the compiled slide deck (PDF, not a native PowerPoint file).
Authors: Alfred Pietraru and Stefan Dascalu. Both documents use the shared
`references.tex` bibliography containing Ahmad et al. (2026), Gayatri and
Aarthy (2024), and Cino et al. (2025), cited in the methods/related-work discussion.

The `report/` and `presentation/` folders each contain their LaTeX source and
compiled PDF. Shared evidence, bibliography, and helper scripts stay here.

## Rebuild

From this directory, use a Python environment with Matplotlib installed:

```bash
python extract_results.py
(cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex)
(cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex)
(cd presentation && pdflatex -interaction=nonstopmode -halt-on-error presentation.tex)
(cd presentation && pdflatex -interaction=nonstopmode -halt-on-error presentation.tex)
```

The extraction script reads the archived SQLite database in read-only mode and
reads the training-history artifacts and federated results JSON. It writes only
under `documentation/evidence/`. No training or external service is required.

## Sources and interpretation

- `../old_experiments/mlflow_new.db`: six centralized run identities, statuses,
  logged parameters, and latest metrics.
- `../old_experiments/mlartifacts/*/*/artifacts/training_history.txt`: historical
  configurations, epoch histories, selected checkpoint epochs, and sample counts.
- `../old_experiments/20261111-105920/results.json`: ten-round, four-client FedAvg
  experiment and individual client histories.
- `../exploratory_data_analysis/artifacts/class_distribution.csv`: full-dataset
  class totals. Its train/validation/test columns describe another EDA split;
  those columns are not used as the experiment counts.

`evidence/centralized_runs.csv` includes interrupted/incomplete runs without
inventing missing test scores. `evidence/results_summary.json` includes full
configuration and metric provenance for further analysis. The report does not
select a baseline winner on test scores.

`evidence/resampling_before_after.pdf` (and `.png`) compares actual training
supports from the archived focal runs without/with resampling; the class counts
are exported in `evidence/resampling_counts.csv`. Counts come from training
confusion-matrix row sums. The EDA resampling CSV uses a different training
fraction and is intentionally not used for this experiment comparison.

The historical federated JSON records a 30% training-group fraction and batch
size 32, versus 80% and 128 in centralized runs. These unintentional differences
are retained as experimental evidence. Historical settings are reconstructed
from archived baseline histories, federated source, and saved-model validation
replay; see `evidence/federated_configuration_reconstruction.md` for per-setting
evidence labels. AdamW and training augmentations are inferred, while focal
loss and validation transforms are supported by replay. To repeat this read-only
model evaluation, run `python recover_federated_config.py` with local ISIC data.
Federated macro-F1 is a sample-weighted mean of
client macro-F1 scores; it is not pooled macro-F1. No federated test result is
available in the archive.

Three supplied papers are included, meeting the assignment's 3–5 reference count.
Hardware details are omitted at the authors' request. The presentation follows
the same archived evidence and comparison limitations as the report.
