"""Train VGG6 on CIFAR-10 with one configuration and log it to Weights & Biases.

Examples
--------
    # best configuration (Question 4)
    python train.py --config configs/best.yaml

    # any single configuration from the command line
    python train.py --activation gelu --optimizer nadam --lr 0.002 --epochs 30

    # as a W&B sweep agent (arguments are injected by the agent)
    wandb sweep configs/sweep.yaml && wandb agent <entity>/<project>/<sweep_id>

Every run writes
    results/runs/<run_name>.json   per-epoch history + final metrics
    checkpoints/<run_name>.pt      weights with the best validation accuracy
"""
import argparse
import json
import os
import time

import torch
import yaml

from vgg6.data import GpuAugment, load_cifar10
from vgg6.engine import evaluate, train_one_epoch
from vgg6.model import ACTIVATIONS, count_params, vgg6
from vgg6.optim import OPTIMIZERS, build_optimizer, build_scheduler
from vgg6.utils import get_device, run_name, set_seed

DEFAULTS = dict(
    activation="relu", batch_norm=True,
    optimizer="sgd", lr=0.05, momentum=0.9, weight_decay=5e-4,
    batch_size=128, epochs=30, scheduler="cosine", warmup_epochs=0,
    label_smoothing=0.0, grad_clip=0.0, cutout=16,
    val_size=5000, seed=42,
    data_dir="./data", out_dir=".", device="auto",
    wandb_project="cs6886w-vgg6-cifar10", wandb_mode=None, group=None,
    save_checkpoint=True,
)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "yes", "y")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="YAML file with any of the options below")
    p.add_argument("--activation", choices=list(ACTIVATIONS))
    p.add_argument("--batch_norm", type=str2bool)
    p.add_argument("--optimizer", choices=OPTIMIZERS)
    p.add_argument("--lr", type=float)
    p.add_argument("--momentum", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--scheduler", choices=["none", "cosine", "onecycle"])
    p.add_argument("--warmup_epochs", type=int)
    p.add_argument("--label_smoothing", type=float)
    p.add_argument("--grad_clip", type=float)
    p.add_argument("--cutout", type=int, help="cutout side length, 0 disables")
    p.add_argument("--val_size", type=int, help="images held out from train for validation")
    p.add_argument("--seed", type=int)
    p.add_argument("--data_dir")
    p.add_argument("--out_dir")
    p.add_argument("--device")
    p.add_argument("--wandb_project")
    p.add_argument("--wandb_mode", choices=["online", "offline", "disabled"])
    p.add_argument("--group", help="W&B group, e.g. the experiment family")
    p.add_argument("--save_checkpoint", type=str2bool)
    args = p.parse_args(argv)

    # Precedence: defaults < YAML file < explicit command-line flags.
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f) or {})
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k != "config"})
    return cfg


def init_wandb(cfg):
    """Start a W&B run (or a no-op stand-in if wandb is disabled / missing)."""
    mode = cfg["wandb_mode"] or os.environ.get("WANDB_MODE", "online")
    if mode == "disabled":
        return None
    import wandb
    run = wandb.init(project=cfg["wandb_project"], config=cfg, group=cfg["group"],
                     name=run_name(cfg), mode=mode, dir=cfg["out_dir"])
    # Inside a sweep the agent may override hyper-parameters through wandb.config.
    cfg.update(dict(run.config))
    run.name = run_name(cfg)
    wandb.define_metric("epoch")
    for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr"):
        wandb.define_metric(k, step_metric="epoch")
    wandb.define_metric("val_acc", summary="max")
    return run


def train(cfg):
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    name = run_name(cfg)
    run = init_wandb(cfg)
    name = run_name(cfg)  # a sweep may have changed the config

    train_set, val_set, test_set = load_cifar10(cfg["data_dir"], device, cfg["val_size"])
    train_tf = GpuAugment(device, augment=True, cutout=cfg["cutout"], seed=cfg["seed"])
    eval_tf = GpuAugment(device, augment=False)
    shuffle_gen = torch.Generator().manual_seed(cfg["seed"])

    model = vgg6(cfg["activation"], cfg["batch_norm"]).to(device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"],
                                cfg["weight_decay"], cfg["momentum"])
    steps = -(-len(train_set.labels) // cfg["batch_size"])
    scheduler = build_scheduler(cfg["scheduler"], optimizer, cfg["epochs"], steps,
                                cfg["warmup_epochs"])
    print(f"[{name}] device={device} params={count_params(model):,} "
          f"train/val/test={len(train_set.labels)}/{len(val_set.labels)}/{len(test_set.labels)}")

    ckpt_path = os.path.join(cfg["out_dir"], "checkpoints", f"{name}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    history, best_val, best_epoch, best_state = [], -1.0, -1, None
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        te = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_set, cfg["batch_size"], optimizer,
                                          scheduler, train_tf, shuffle_gen,
                                          cfg["label_smoothing"], cfg["grad_clip"])
        va_loss, va_acc = evaluate(model, val_set, eval_tf)
        row = dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, val_loss=va_loss,
                   val_acc=va_acc, lr=lr, epoch_time=time.time() - te)
        history.append(row)
        if run is not None:
            run.log(row)
        if va_acc > best_val:
            best_val, best_epoch = va_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{name}] ep {epoch:3d}/{cfg['epochs']} lr {lr:.2e} | train {tr_loss:.4f} "
              f"{tr_acc:6.2f}% | val {va_loss:.4f} {va_acc:6.2f}% | {row['epoch_time']:.1f}s",
              flush=True)
        if tr_loss != tr_loss:  # NaN: diverged, stop early
            print(f"[{name}] training diverged at epoch {epoch}")
            break

    # Final numbers are reported for the checkpoint with the best validation accuracy.
    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_set, eval_tf)
    final = dict(best_val_acc=best_val, best_epoch=best_epoch, test_acc=test_acc,
                 test_loss=test_loss, final_val_acc=history[-1]["val_acc"],
                 final_train_acc=history[-1]["train_acc"],
                 train_time_s=time.time() - t0, params=count_params(model))
    print(f"[{name}] best val {best_val:.2f}% @ep{best_epoch} | test {test_acc:.2f}%")

    if cfg["save_checkpoint"]:
        torch.save({"config": cfg, "state_dict": best_state, "metrics": final}, ckpt_path)
    res_path = os.path.join(cfg["out_dir"], "results", "runs", f"{name}.json")
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, "w") as f:
        json.dump({"name": name, "config": cfg, "history": history, "final": final}, f, indent=1)
    if run is not None:
        run.summary.update(final)
        run.finish()
    return final


if __name__ == "__main__":
    train(parse_args())
