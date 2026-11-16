"""Lazy image datasets and reproducible, lesion-separated ISIC2019 loaders."""

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
import zipfile

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from monai.data.utils import worker_init_fn

from net_training.training_configuration import TrainingCFG


DATA_DIR = Path(__file__).resolve().parent / "data"


DRIVE_FILE_ID = "1m_ExzDO5RffGYogf18BDXAq_9ZfhABiN"


CLASS_NAMES = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")


def ensure_isic_dataset(
    image_dir: Path = DATA_DIR / "ISIC_2019_Training_Input",
    labels_csv: Path = DATA_DIR / "ISIC_2019_Training_GroundTruth.csv",
    metadata_csv: Path = DATA_DIR / "ISIC_2019_Training_Metadata.csv",
) -> None:
    """Download only if CSVs or images referenced by ground truth are missing.

    The Drive file is expected to be a ZIP containing ISIC JPEGs and any missing
    ground-truth/metadata CSVs. Wrapper directories in the ZIP are supported.
    Existing local files are preserved. Downloads are staged and cleaned up on
    failure; individual output files are installed atomically.
    """
    nested = image_dir / "ISIC_2019_Training_Input"
    if nested.is_dir():
        image_dir = nested

    def image_names(path):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            assert "image" in (reader.fieldnames or []), "Ground truth must contain image column."
            names = [f"{row['image']}.jpg" for row in reader]
        assert names, "Ground-truth CSV must contain images."
        assert all(Path(name).name == name and "\\" not in name for name in names), (
            "Invalid image identifier in ground-truth CSV."
        )
        return names

    if labels_csv.is_file() and metadata_csv.is_file():  # noqa: SIM102
        if all((image_dir / name).is_file() for name in image_names(labels_csv)):
            return

    # Lazy import: existing datasets do not require the download dependency.
    import gdown

    with TemporaryDirectory(prefix="isic-download-") as temporary:
        archive_path = Path(temporary) / "dataset.zip"
        result = gdown.download(id=DRIVE_FILE_ID, output=str(archive_path), quiet=True)
        assert result is not None and zipfile.is_zipfile(archive_path), (
            "Google Drive download failed or did not return a ZIP archive. "
            "Check public sharing permissions and download quota."
        )
        with zipfile.ZipFile(archive_path) as archive:
            members = {}
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = PurePosixPath(member.filename).name
                members.setdefault(name, []).append(member)

            def find_member(name):
                matches = members.get(name, [])
                assert len(matches) == 1, (
                    f"Expected exactly one {name} in the Drive ZIP; found {len(matches)}."
                )
                return matches[0]

            # Inspect labels before installing any files, including image-only archives
            # when ground truth and metadata already exist locally.
            label_source = labels_csv
            if not labels_csv.is_file():
                label_source = Path(temporary) / "labels.csv"
                with archive.open(find_member(labels_csv.name)) as source, label_source.open("wb") as target:
                    shutil.copyfileobj(source, target)
            required = [p for p in (labels_csv, metadata_csv) if not p.is_file()]
            required += [image_dir / name for name in image_names(label_source)
                         if not (image_dir / name).is_file()]
            planned = [(destination, find_member(destination.name)) for destination in required]
            for destination, member in planned:
                destination.parent.mkdir(parents=True, exist_ok=True)
                staged = None
                try:
                    with NamedTemporaryFile(dir=destination.parent, prefix=".isic-", delete=False) as target:
                        staged = Path(target.name)
                        with archive.open(member) as source:
                            shutil.copyfileobj(source, target)
                    staged.replace(destination)
                finally:
                    if staged is not None:
                        staged.unlink(missing_ok=True)



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
    When oversample_classes is True, duplicate training entries for AK, DF,
    VASC, and SCC once after splitting, and drop floor(60% of NV training
    entries) using the configured seed. Validation and test remain unchanged.
    """

    def __init__(
        self,
        image_dir: str | Path = DATA_DIR / "ISIC_2019_Training_Input",
        labels_csv: str | Path = DATA_DIR / "ISIC_2019_Training_GroundTruth.csv",
        metadata_csv: str | Path = DATA_DIR / "ISIC_2019_Training_Metadata.csv",
        oversample_classes: bool = False,
    ):
        assert type(oversample_classes) is bool, "oversample_classes must be a boolean."
        self.oversample_classes = oversample_classes
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
        split_ratios: tuple[float, float, float],
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Split stored records and return fresh train, validation, test loaders.

        Each batch is (images, labels). Dataset samples expose (path, label)
        pairs; self.samples also retains the full CSV metadata as strings.
        Split ratios are supplied separately; TrainingCFG supplies loader settings and transforms.
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
            "prefetch_factor": 4 if num_workers > 0 else None,
            "pin_memory": torch.cuda.is_available(),
            "worker_init_fn": worker_init_fn,
        }
        datasets = self.get_datasets_baseline_experiment(train_cfg=train_cfg, split_ratios=split_ratios)

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
        split_ratios: tuple[float, float, float],
    ) -> tuple[ISICDataset, ISICDataset, ISICDataset]:
        """Return the lesion-separated train, validation, and test datasets.

        Uses the same split rules as get_dataloaders_baseline_experiment.
        """
        assert isinstance(train_cfg, TrainingCFG), "train_cfg must be a TrainingCFG instance."
        assert (
            isinstance(split_ratios, (list, tuple))
            and len(split_ratios) == 3
            and all(type(value) in (int, float) and 0 < value < 1 for value in split_ratios)
            and (sum(split_ratios) <= 1
                 or math.isclose(sum(split_ratios), 1, rel_tol=0, abs_tol=1e-12))
        ), "split_ratios must be three positive fractions summing to at most 1."
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
        for split_index, (selected, transform) in enumerate(zip(
            (train_groups, val_groups, test_groups),
            (train_cfg.train_transforms, train_cfg.val_transforms, train_cfg.val_transforms),
        )):
            indices = sorted(i for group in selected for i in group_indices[group])
            assert {samples[i][1] for i in indices} == set(range(len(CLASS_NAMES))), (
                'Every split must contain all eight classes.'
            )
            if split_index == 0 and self.oversample_classes:
                nv_indices = [i for i in indices if samples[i][1] == CLASS_NAMES.index("NV")]
                dropped = set(random.Random(seed).sample(nv_indices, len(nv_indices) * 3 // 5))
                indices = [i for i in indices if i not in dropped]
                oversampled_labels = {CLASS_NAMES.index(name) for name in ("AK", "DF", "VASC", "SCC")}
                indices += [i for i in indices if samples[i][1] in oversampled_labels]
            datasets.append(ISICDataset(
                [samples[i] for i in indices], [groups[i] for i in indices],
                transform=transform,
            ))

        return tuple(datasets)
