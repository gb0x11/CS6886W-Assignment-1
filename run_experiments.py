"""Run the experiment families of Question 2 (and the sampled sweep of Question 3).

Each configuration is launched as a separate `train.py` process so that runs
are isolated and identical to what a user would run by hand. A run is skipped
when its result file already exists, so the script can be interrupted and
resumed.

    python run_experiments.py --group activation          # Q2(a)
    python run_experiments.py --group optimizer           # Q2(b)
    python run_experiments.py --group bs lr epochs        # Q2(c)
    python run_experiments.py --group refine              # follow-ups to pick the final config
    python run_experiments.py --group sweep --n 12        # random sweep (Q3)
    python run_experiments.py --group all --dry_run       # list every run
"""
import argparse
import os
import random
import subprocess
import sys

import yaml

from vgg6.utils import run_name

# Shared baseline that every family varies one axis of.
BASE = dict(activation="relu", batch_norm=True, optimizer="sgd", lr=0.05, momentum=0.9,
            weight_decay=5e-4, batch_size=128, epochs=20, scheduler="cosine", seed=42)

# Q2(b): each optimizer is tried at three learning rates around its usual default,
# since a single shared LR would be unfair to either the SGD or the adaptive family.
OPTIMIZER_LRS = {
    ("sgd", 0.0): [0.05, 0.1, 0.2],   # plain SGD (no momentum)
    ("nesterov", 0.9): [0.01, 0.05, 0.1],
    ("adam", 0.9): [0.0003, 0.001, 0.003],
    ("adagrad", 0.9): [0.003, 0.01, 0.03],
    ("rmsprop", 0.9): [0.0001, 0.0003, 0.001],
    ("nadam", 0.9): [0.0003, 0.001, 0.003],
}


def family(group, n_sweep=12, sweep_seed=0):
    """List of (group, config) pairs for one experiment family."""
    def var(**kw):
        return dict(BASE, **kw)

    if group == "activation":
        return [var(activation=a) for a in ("relu", "sigmoid", "tanh", "silu", "gelu")]
    if group == "activation_nobn":
        return [var(activation=a, batch_norm=False, lr=0.01)
                for a in ("relu", "sigmoid", "tanh", "silu", "gelu")]
    if group == "optimizer":
        return [var(optimizer=o, momentum=m, lr=lr)
                for (o, m), lrs in OPTIMIZER_LRS.items() for lr in lrs]
    if group == "lr":  # momentum SGD; 0.05 is the baseline
        return [var(lr=lr) for lr in (0.005, 0.01, 0.05, 0.1, 0.2)]
    if group == "bs":  # fixed LR, so larger batches take fewer, equally sized steps
        return [var(batch_size=b) for b in (32, 64, 128, 256, 512)]
    if group == "epochs":
        return [var(epochs=e) for e in (5, 10, 20, 40)]
    if group == "refine":  # follow-ups around the best optimizer settings of Q2(b)
        return [var(optimizer="nadam", lr=0.006), var(optimizer="nesterov", lr=0.2),
                var(optimizer="nadam", lr=0.003, activation="gelu")]
    if group == "sweep":
        with open("configs/sweep.yaml") as f:
            space = yaml.safe_load(f)["parameters"]
        rng = random.Random(sweep_seed)
        return [var(**{k: rng.choice(v["values"]) for k, v in space.items()})
                for _ in range(n_sweep)]
    raise ValueError(group)


GROUPS = ["activation", "optimizer", "lr", "bs", "epochs", "refine", "activation_nobn", "sweep"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--group", nargs="+", default=["all"], choices=GROUPS + ["all"])
    p.add_argument("--n", type=int, default=12, help="number of sampled sweep runs")
    p.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    groups = GROUPS if "all" in args.group else args.group
    seen = set()
    for g in groups:
        for cfg in family(g, args.n):
            name = run_name(cfg)
            if name in seen:  # e.g. the baseline appears in several families
                continue
            seen.add(name)
            done = os.path.exists(os.path.join("results", "runs", f"{name}.json"))
            print(f"[{g}] {name}{'  (done)' if done else ''}", flush=True)
            if done or args.dry_run:
                continue
            cmd = [sys.executable, "train.py", f"--group={g}", f"--wandb_mode={args.wandb_mode}"]
            cmd += [f"--{k}={v}" for k, v in cfg.items()]
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
