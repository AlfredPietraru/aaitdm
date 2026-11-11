"""Export original ISIC images and class overview figures for each data split.

Run from any directory:
    python exploratory_data_analysis/get_a_couple_images.py --samples-per-class 3
"""

import argparse
import logging
import csv
from pathlib import Path
import random
import shutil
import sys

from matplotlib.figure import Figure
from PIL import Image, ImageOps
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_logging import configure_logging

from isic_dataloaders import CLASS_NAMES, ISICDataLoaders
from training_configuration import TrainingCFG


CLASS_DESCRIPTIONS = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}


def save_samples(
    samples_per_class: int = 3,
    output_dir: str | Path = Path(__file__).resolve().parent / "artifacts" / "samples",
    config_path: str | Path = PROJECT_ROOT / "training_configuration.yaml",
) -> Path:
    """Save samples from configured splits without augmentation or normalization.

    Prefer different lesion groups within each class. Selection is reproducible
    using the configuration seed. Originals are copied unchanged; overview
    figures contain resized previews. manifest.csv records image provenance.
    """
    configure_logging()
    assert type(samples_per_class) is int and samples_per_class > 0, (
        "samples_per_class must be a positive integer."
    )
    with Path(config_path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    cfg = TrainingCFG(data)
    source = ISICDataLoaders(
        image_dir=PROJECT_ROOT / "ISIC_2019_Training_Input",
        labels_csv=PROJECT_ROOT / "ISIC_2019_Training_GroundTruth.csv",
        metadata_csv=PROJECT_ROOT / "ISIC_2019_Training_Metadata.csv",
        oversample_classes=cfg.oversample_classes,
    )
    loaders = source.get_dataloaders_baseline_experiment(train_cfg=cfg)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed)
    records = []

    for split, loader in zip(("train", "validation", "test"), loaders):
        dataset = loader.dataset
        figure = Figure(figsize=(4 * samples_per_class, 3 * len(CLASS_NAMES)),
                        layout="constrained")
        axes = figure.subplots(len(CLASS_NAMES), samples_per_class, squeeze=False)
        figure.suptitle(f"ISIC2019 — {split} — original image samples", fontsize=18)
        for label, name in enumerate(CLASS_NAMES):
            candidates = [i for i, (_, target) in enumerate(dataset.samples) if target == label]
            rng.shuffle(candidates)
            selected, seen_groups = [], set()
            for index in candidates:
                group = dataset.groups[index]
                if group not in seen_groups:
                    selected.append(index)
                    seen_groups.add(group)
                if len(selected) == samples_per_class:
                    break
            if len(selected) < samples_per_class:
                selected += [i for i in candidates if i not in selected][
                    :samples_per_class - len(selected)
                ]

            class_dir = output_dir / split / name
            class_dir.mkdir(parents=True, exist_ok=True)
            for column, axis in enumerate(axes[label]):
                axis.set_xticks([])
                axis.set_yticks([])
                if column == 0:
                    axis.set_ylabel(f"{name}\n{CLASS_DESCRIPTIONS[name]}", fontsize=11)
                if column >= len(selected):
                    axis.text(.5, .5, "No additional image available", ha="center")
                    continue
                index = selected[column]
                path, _ = dataset.samples[index]
                destination = class_dir / f"sample_{column + 1:02d}{path.suffix}"
                shutil.copy2(path, destination)
                with Image.open(path) as image:
                    preview = ImageOps.exif_transpose(image).convert("RGB")
                    preview.thumbnail((640, 640))
                axis.imshow(preview)
                axis.set_title(path.stem, fontsize=10)
                records.append({
                    "split": split, "class_index": label, "class": name,
                    "description": CLASS_DESCRIPTIONS[name], "image_id": path.stem,
                    "lesion_group": dataset.groups[index], "source_path": str(path),
                    "saved_path": str(destination.relative_to(output_dir)),
                })
        figure.savefig(output_dir / f"{split}_overview.png", dpi=120)
        figure.clear()

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    logging.getLogger(__name__).info(
        "Saved %d original images and three class overviews to %s", len(records), output_dir
    )
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=3)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "artifacts" / "samples")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "training_configuration.yaml")
    args = parser.parse_args()
    save_samples(args.samples_per_class, args.output_dir, args.config)
