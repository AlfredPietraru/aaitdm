from torch import nn
from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader

from neural_network import BinaryClassifier, train
from mlflow_logger import MLflowLogger
from training_configuration import train_cfg
from reproducibility import set_seed


def train_baseline_pipeline() -> nn.Module:
    """Build the Camelyon17 baseline pipeline and return the trained classifier."""
    seed: int = 42
    set_seed(seed, pipelines=(train_cfg.train_transforms, train_cfg.val_transforms))
    dataset = get_dataset(
        dataset="camelyon17",
        root_dir="./data",
        download=True,
    )
    assert dataset is not None, "There was an error obtaining the dataset."

    train_data = dataset.get_subset("train", transform=train_cfg.train_transforms)
    val_data = dataset.get_subset("val", transform=train_cfg.val_transforms)
    test_data = dataset.get_subset("test", transform=train_cfg.val_transforms)

    train_loader = get_train_loader(
        "standard", train_data, batch_size=train_cfg.batch_size
    )
    assert train_loader is not None, "Failed to initialize the training dataloader."
    val_loader = get_eval_loader(
        "standard", val_data, batch_size=train_cfg.batch_size
    )
    assert val_loader is not None, "Failed to initialize the validation dataloader."
    test_loader = get_eval_loader(
        "standard", test_data, batch_size=train_cfg.batch_size
    )
    assert test_loader is not None, "Failed to initialize the test dataloader."

    model = BinaryClassifier()
    logger = MLflowLogger()
    return train(model, train_cfg, train_loader, val_loader, test_loader, logger)


if __name__ == "__main__":
    trained_model = train_baseline_pipeline()
