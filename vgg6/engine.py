"""Training and evaluation loops."""
import torch
import torch.nn.functional as F

from .data import iterate


def train_one_epoch(model, split, batch_size, optimizer, scheduler, transform, gen,
                    label_smoothing=0.0, grad_clip=0.0):
    """One pass over the training split.

    Returns mean loss and accuracy (%) measured on the augmented batches as
    they are trained on (the usual "running" training metrics).
    """
    model.train()
    loss_sum = torch.zeros((), device=split.labels.device)
    correct = torch.zeros((), device=split.labels.device)
    n = 0
    for x, y in iterate(split, batch_size, transform, shuffle=True, gen=gen):
        out = model(x)
        loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Accumulate on device; a single host sync at the end of the epoch.
        loss_sum += loss.detach() * len(y)
        correct += (out.argmax(1) == y).sum()
        n += len(y)
    return (loss_sum / n).item(), (100.0 * correct / n).item()


@torch.no_grad()
def evaluate(model, split, transform, batch_size=1000):
    """Mean cross-entropy loss and top-1 accuracy (%) on a split."""
    model.eval()
    loss_sum = torch.zeros((), device=split.labels.device)
    correct = torch.zeros((), device=split.labels.device)
    for x, y in iterate(split, batch_size, transform):
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum")
        correct += (out.argmax(1) == y).sum()
    n = len(split.labels)
    return (loss_sum / n).item(), (100.0 * correct / n).item()


@torch.no_grad()
def predict(model, split, transform, batch_size=1000):
    """Predicted class index for every image of a split (for confusion matrices)."""
    model.eval()
    preds = [model(x).argmax(1) for x, _ in iterate(split, batch_size, transform)]
    return torch.cat(preds)
