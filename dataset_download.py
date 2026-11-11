"""Recover missing ISIC2019 files from the configured public Drive ZIP."""

import csv
import logging
from pathlib import Path, PurePosixPath
import shutil
from tempfile import TemporaryDirectory, NamedTemporaryFile
import zipfile

DRIVE_FILE_ID = "1m_ExzDO5RffGYogf18BDXAq_9ZfhABiN"
LOGGER = logging.getLogger(__name__)


def ensure_isic_dataset(image_dir: Path, labels_csv: Path, metadata_csv: Path) -> None:
    """Download only if CSVs or images referenced by ground truth are missing.

    The Drive file is expected to be a ZIP containing ISIC JPEGs and any missing
    ground-truth/metadata CSVs. Wrapper directories in the ZIP are supported.
    Existing local files are preserved. Downloads are staged and cleaned up on
    failure; individual output files are installed atomically.
    """
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

    if labels_csv.is_file() and metadata_csv.is_file():
        if all((image_dir / name).is_file() for name in image_names(labels_csv)):
            return

    # Lazy import: existing datasets do not require the download dependency.
    import gdown

    LOGGER.info("Missing ISIC2019 files; downloading Drive archive %s", DRIVE_FILE_ID)
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
        LOGGER.info("Installed %d missing ISIC2019 files", len(planned))
