"""Evaluate a saved VGG6 checkpoint on the CIFAR-10 validation and test splits.

    python evaluate.py --checkpoint checkpoints/best_model.pt
"""
import argparse

import torch

from vgg6.data import CLASSES, GpuAugment, load_cifar10
from vgg6.engine import evaluate, predict
from vgg6.model import count_params, vgg6
from vgg6.utils import get_device


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = vgg6(cfg["activation"], cfg["batch_norm"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded {args.checkpoint} ({count_params(model):,} params)")
    print("config:", {k: cfg[k] for k in ("activation", "batch_norm", "optimizer", "lr",
                                          "batch_size", "epochs", "weight_decay", "seed")})

    _, val_set, test_set = load_cifar10(args.data_dir, device, cfg["val_size"])
    tf = GpuAugment(device, augment=False)
    val_loss, val_acc = evaluate(model, val_set, tf)
    test_loss, test_acc = evaluate(model, test_set, tf)
    print(f"validation: loss {val_loss:.4f}  top-1 {val_acc:.2f}%")
    print(f"test:       loss {test_loss:.4f}  top-1 {test_acc:.2f}%")

    preds = predict(model, test_set, tf).cpu()
    labels = test_set.labels.cpu()
    print("per-class test accuracy:")
    for c, cls in enumerate(CLASSES):
        m = labels == c
        print(f"  {cls:<11s} {100.0 * (preds[m] == c).float().mean():6.2f}%")


if __name__ == "__main__":
    main()
