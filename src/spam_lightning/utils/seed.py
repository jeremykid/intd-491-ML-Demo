"""Random seeding helpers."""

from __future__ import annotations

import random

import numpy as np
import pytorch_lightning as L
import torch


def set_global_seed(seed: int) -> None:
    """Set Python, NumPy, Torch, and Lightning seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)
