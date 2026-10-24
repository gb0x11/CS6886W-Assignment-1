"""Optimizer and learning-rate schedule factories."""
import torch.optim as optim

OPTIMIZERS = ("sgd", "nesterov", "adam", "adamw", "adagrad", "rmsprop", "nadam")


def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    """Create an optimizer by name.

    ``sgd`` is classical (heavy-ball) momentum SGD; with ``momentum=0`` it is
    plain SGD. ``nesterov`` is SGD with Nesterov momentum. ``momentum`` is
    ignored by the adaptive methods.
    """
    name = name.lower()
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "nesterov":
        return optim.SGD(params, lr=lr, momentum=momentum or 0.9, nesterov=True,
                         weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, alpha=0.99, weight_decay=weight_decay)
    if name == "nadam":
        return optim.NAdam(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}; choose from {OPTIMIZERS}")


def build_scheduler(name, optimizer, epochs, steps_per_epoch, warmup_epochs=0):
    """Per-step LR schedule: ``none``, ``cosine`` or ``onecycle``.

    ``cosine`` optionally starts with a linear warm-up of ``warmup_epochs``.
    Returns None for a constant learning rate.
    """
    total = epochs * steps_per_epoch
    if name == "none":
        return None
    if name == "onecycle":
        max_lr = [g["lr"] for g in optimizer.param_groups]
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total,
                                             pct_start=0.25)
    if name == "cosine":
        warm = warmup_epochs * steps_per_epoch
        cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total - warm, 1))
        if warm == 0:
            return cos
        lin = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warm)
        return optim.lr_scheduler.SequentialLR(optimizer, [lin, cos], milestones=[warm])
    raise ValueError(f"unknown scheduler {name!r}")
