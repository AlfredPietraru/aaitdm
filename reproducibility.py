"""Reproducible random states and deterministic training settings."""

from collections.abc import Iterable
import os
import random

from monai.transforms.compose import Compose
from monai.utils.misc import set_determinism
import numpy as np
import torch


def set_seed(seed: int = 42, pipelines: Iterable[Compose] = ()) -> None:
    """Seed all RNGs before creating models, loaders, or initializing CUDA.

    Existing MONAI pipelines need their own random states reset explicitly.
    Reproducibility requires the same hardware, software, and loader settings.
    Operations without deterministic implementations fail rather than vary.
    """
    assert type(seed) is int and 0 <= seed < 2**32, "seed must be a 32-bit unsigned integer."
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    set_determinism(seed=seed, use_deterministic_algorithms=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for pipeline in pipelines:
        pipeline.set_random_state(seed=seed)
