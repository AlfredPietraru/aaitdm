"""Lazy image datasets and reproducible, lesion-separated ISIC2019 loaders."""

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from monai.data.utils import worker_init_fn

from training_configuration import TrainingCFG


CLASS_NAMES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")


class ISICDataset(Dataset):
    """Store image paths and labels; decode images only when requested.

    Transforms receive an RGB PIL image and must return a batchable tensor.
    Without a transform, images are resized to image_size (height, width)
    and converted to CHW float tensors in [0, 1].
    """

    class_names = CLASS_NAMES

    def __init__(self, samples, groups, transform=None, image_size=(224, 224)):
        self.samples = tuple(samples)
        self.groups = tuple(groups)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            height, width = self.image_size
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
        return image, torch.tensor(label, dtype=torch.long)


class ISICDataLoaders:
    """Read image references, labels, and full metadata once at construction.

    get_dataloaders_baseline_experiment creates splits and loaders on demand.

    Split fractions apply to lesion groups, so image counts are approximate.
    Groups are stratified by their majority class (ties use class order).
    Missing lesion IDs are treated as separate images; this cannot guarantee
    patient separation. UNK-positive rows are rejected for this eight-class task.
    """

    def __init__(
        self,
        image_dir: str | Path = "ISIC_2019_Training_Input",
        labels_csv: str | Path = "ISIC_2019_Training_GroundTruth.csv",
        metadata_csv: str | Path = "ISIC_2019_Training_Metadata.csv",
    ):
        image_dir = Path(image_dir)
        # Support both the extracted directory and the archive's nested layout.
        nested = image_dir / "ISIC_2019_Training_Input"
        if nested.is_dir():
            image_dir = nested
        with Path(metadata_csv).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            assert {'image', 'lesion_id'}.issubset(reader.fieldnames or []), (
                'Metadata must contain image and lesion_id columns.'
            )
            metadata_by_image = {}
            for row in reader:
                assert row['image'] not in metadata_by_image, (
                    f'Duplicate metadata image: {row['image']}'
                )
                metadata_by_image[row["image"]] = dict(row)

        self.samples: list[tuple[Path, int, dict[str, str]]] = []
        seen = set()
        with Path(labels_csv).open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            assert {'image', *CLASS_NAMES}.issubset(reader.fieldnames or []), (
                'Ground truth must contain image and all eight class columns.'
            )
            for row in reader:
                image_id = row["image"]
                assert image_id not in seen, (
                    f'Duplicate ground-truth image: {image_id}'
                )
                seen.add(image_id)
                values = [float(row[name]) for name in CLASS_NAMES]
                assert not (any((value not in (0, 1) for value in values)) or sum(values) != 1 or float(row.get('UNK', 0)) != 0), (
                    f'Expected one known class and no UNK label: {image_id}'
                )
                path = image_dir / f"{image_id}.jpg"
                if not path.is_file():
                    raise FileNotFoundError(path)
                assert image_id in metadata_by_image, (
                    f'Missing metadata row for {image_id}'
                )
                self.samples.append((path, values.index(1), metadata_by_image[image_id]))

    def get_dataloaders_baseline_experiment(
        self,
        *,
        train_cfg: TrainingCFG,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Split stored records and return fresh train, validation, test loaders.

        Each batch is (images, labels). Dataset samples expose (path, label)
        pairs; self.samples also retains the full CSV metadata as strings.
        TrainingCFG supplies split ratios, batch size, seed, workers, and transforms.
        Repeated calls can use different settings without rereading the CSVs.
        Ratios are fractions of all lesion groups, not relative weights. Their
        sum may be below 1; remaining groups are unused. For example,
        (0.1, 0.03, 0.03) uses approximately 16% of images. Very small splits
        must still have enough groups to represent all eight classes.
        """
        seed = train_cfg.seed
        num_workers = train_cfg.num_workers
        loader_kwargs = {
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
        }
        datasets = self.get_datasets_baseline_experiment(train_cfg=train_cfg)

        train_loader, val_loader, test_loader = (
            DataLoader(
                dataset, batch_size=train_cfg.batch_size, shuffle=(index == 0),
                **loader_kwargs,
                generator=torch.Generator().manual_seed(seed + index),
            )
            for index, dataset in enumerate(datasets)
        )

        return train_loader, val_loader, test_loader

    def get_datasets_baseline_experiment(
        self,
        *,
        train_cfg: TrainingCFG,
    ) -> tuple[ISICDataset, ISICDataset, ISICDataset]:
        """Return the lesion-separated train, validation, and test datasets.

        Uses the same split rules as get_dataloaders_baseline_experiment.
        """
        assert isinstance(train_cfg, TrainingCFG), "train_cfg must be a TrainingCFG instance."
        split_ratios = train_cfg.split_ratios
        seed = train_cfg.seed

        samples = [(path, label) for path, label, _ in self.samples]
        groups = []
        for path, _, metadata in self.samples:
            lesion = metadata["lesion_id"].strip()
            groups.append(f"lesion:{lesion}" if lesion else f"image:{path.stem}")

        group_indices = defaultdict(list)
        for index, group in enumerate(groups):
            group_indices[group].append(index)
        group_names = sorted(group_indices)
        group_labels = {}
        for group, indices in group_indices.items():
            counts = Counter(samples[index][1] for index in indices)
            group_labels[group] = min(counts, key=lambda label: (-counts[label], label))
        try:
            train_groups, heldout_groups = train_test_split(
                group_names, test_size=split_ratios[1] + split_ratios[2],
                random_state=seed, stratify=[group_labels[g] for g in group_names],
            )
            val_groups, test_groups = train_test_split(
                heldout_groups, test_size=split_ratios[2] / sum(split_ratios[1:]),
                random_state=seed, stratify=[group_labels[g] for g in heldout_groups],
            )
            if not math.isclose(sum(split_ratios), 1, rel_tol=0, abs_tol=1e-12):
                # Select the requested share of the full collection, leaving
                # the rest unused rather than assigning it all to training.
                train_count = math.floor(len(group_names) * split_ratios[0])
                train_groups, _ = train_test_split(
                    train_groups, train_size=train_count, random_state=seed,
                    stratify=[group_labels[g] for g in train_groups],
                )
        except ValueError as error:
            assert False, (
                "Cannot stratify lesion groups with these split ratios. "
                f"Each class needs enough independent groups. Details: {error}"
            )

        datasets = []
        for selected, transform in zip(
            (train_groups, val_groups, test_groups),
            (train_cfg.train_transforms, train_cfg.val_transforms, train_cfg.val_transforms),
        ):
            indices = sorted(i for group in selected for i in group_indices[group])
            assert {samples[i][1] for i in indices} == set(range(len(CLASS_NAMES))), (
                'Every split must contain all eight classes.'
            )
            datasets.append(ISICDataset(
                [samples[i] for i in indices], [groups[i] for i in indices],
                transform=transform,
            ))

        return tuple(datasets)
