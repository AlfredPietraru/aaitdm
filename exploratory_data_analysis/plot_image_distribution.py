"""Plot original and oversampled class counts for configured ISIC2019 splits."""

import argparse
from collections import Counter
import csv
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_logging import configure_logging

import yaml
from matplotlib.figure import Figure

from isic_dataloaders import CLASS_NAMES, ISICDataLoaders
from training_configuration import TrainingCFG


def plot_image_distribution(
    config_path: str | Path = PROJECT_ROOT / "training_configuration.yaml",
    output_dir: str | Path = Path(__file__).resolve().parent / "artifacts",
) -> list[dict]:
    """Count loader entries, unique images, and duplicates without decoding images.

    Split fractions and seed come from the YAML. Only training is oversampled;
    validation/test counts are reported as well to make that distinction visible.
    """
    configure_logging()
    logger = logging.getLogger(__name__)
    with Path(config_path).open(encoding="utf-8") as handle:
        train_cfg = TrainingCFG(yaml.safe_load(handle))
    dataset = ISICDataLoaders(
        image_dir=PROJECT_ROOT / "ISIC_2019_Training_Input",
        labels_csv=PROJECT_ROOT / "ISIC_2019_Training_GroundTruth.csv",
        metadata_csv=PROJECT_ROOT / "ISIC_2019_Training_Metadata.csv",
        oversample_classes=train_cfg.oversample_classes,
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders_baseline_experiment(
        train_cfg=train_cfg,
    )
    rows = []
    figure = Figure(figsize=(14, 14), layout="constrained")
    axes = figure.subplots(3, 1)
    figure.suptitle(
        f"ISIC2019 class distribution — split ratios {train_cfg.split_ratios}, seed {train_cfg.seed}\n"
        + ("Training: AK, DF, VASC, SCC duplicated once; 60% of NV entries removed"
           if train_cfg.oversample_classes else "Class resampling disabled"),
    )
    for axis, split, loader in zip(
        axes, ("train", "validation", "test"), (train_loader, val_loader, test_loader)
    ):
        samples = loader.dataset.samples
        counts = Counter(label for _, label in samples)
        unique_counts = Counter(label for _, label in set(samples))
        originals = [unique_counts[i] for i in range(len(CLASS_NAMES))]
        duplicates = [counts[i] - unique_counts[i] for i in range(len(CLASS_NAMES))]
        for index, name in enumerate(CLASS_NAMES):
            rows.append({
                "split": split, "class_name": name,
                "unique_images": unique_counts[index],
                "duplicate_entries": duplicates[index],
                "total_entries": counts[index],
                "percentage_of_split": 100 * counts[index] / len(samples),
                "oversampled": duplicates[index] > 0,
            })
        axis.bar(CLASS_NAMES, originals, color="#4477AA", label="Unique images")
        axis.bar(CLASS_NAMES, duplicates, bottom=originals, color="#EEAA33",
                 label="Added duplicate entries")
        offset = max(counts.values()) * 0.02
        for index, name in enumerate(CLASS_NAMES):
            axis.text(index, counts[index] + offset, str(counts[index]), ha="center")
        axis.set(title=f"{split.capitalize()}: {len(samples):,} entries, {len(set(samples)):,} unique images",
                 ylabel="Number of entries", xlabel="Class",
                 ylim=(0, max(counts.values()) * 1.15))
        axis.legend()
        logger.info("%s class counts: %s", split, {name: counts[i] for i, name in enumerate(CLASS_NAMES)})

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "oversampled_class_distribution.png", dpi=160)
    figure.clear()
    with (output_dir / "oversampled_class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved oversampled distribution plot and CSV to %s", output_dir)
    return rows


if __name__ == "__main__":
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training_configuration.yaml")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = parser.parse_args()
    plot_image_distribution(args.config, args.output_dir)
