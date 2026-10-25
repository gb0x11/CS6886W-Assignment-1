"""Seeding, device selection and small helpers shared by the entry scripts."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG that influences training (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA / MPS default generators
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(name: str = "auto") -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU unless a device is forced."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_name(cfg: dict) -> str:
    """Readable run name that encodes the main hyper-parameters.

    Numbers use ``:g`` so 0.0 and 0 (W&B stores whole floats as ints) give the same name.
    """
    def g(v):
        return f"{float(v):g}"

    opt = cfg["optimizer"]
    if opt == "sgd":  # distinguish plain SGD from momentum SGD
        opt += f"_mom_{g(cfg['momentum'])}"
    return (f"act_{cfg['activation']}_bs_{cfg['batch_size']}_opt_{opt}"
            f"_lr_{g(cfg['lr'])}_wd_{g(cfg['weight_decay'])}_ep_{cfg['epochs']}"
            f"_bn_{cfg['batch_norm']}")
