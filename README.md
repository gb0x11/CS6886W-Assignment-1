# CS6886W Assignment 1: VGG6 on CIFAR-10

This repository explores how the **activation function, optimizer, batch size, number of epochs and learning rate** affect a small VGG6 network (≈0.26 M parameters) trained on CIFAR-10. All runs are logged to Weights & Biases.

> Results and analysis are added below once the experiments finish.

## Repository layout

```
vgg6/
  data.py        CIFAR-10 loading, train/val split, normalization + augmentation
  model.py       VGG6 with a configurable activation (relu/sigmoid/tanh/silu/gelu/...)
  optim.py       optimizer factory (SGD, Nesterov, Adam, Adagrad, RMSprop, Nadam) + LR schedules
  engine.py      train_one_epoch / evaluate / predict loops
  utils.py       seeding, device selection, run naming
train.py         train ONE configuration, log to W&B, save checkpoint + JSON history
evaluate.py      evaluate a saved checkpoint on the validation and test sets
run_experiments.py  run the experiment families of Q2 and the sampled sweep (Q3)
configs/
  best.yaml      the single best configuration (Q4)
  sweep.yaml     W&B sweep definition (same search space as the offline sweep)
scripts/make_plots.py  local plots + summary table from results/runs/*.json
results/runs/    per-run JSON (config, per-epoch history, final metrics)
checkpoints/best_model.pt  trained weights of the best configuration
```

## Environment

| | |
|---|---|
| Hardware | Apple M1 Pro (16 GB), PyTorch MPS backend |
| OS | macOS (Darwin 27.0) |
| Python | 3.14.3 |
| Libraries | torch 2.14.1, torchvision 0.29.1, wandb 0.31.0, numpy 2.5.4, matplotlib 3.11.2, pandas 3.1.0, PyYAML 6.0.3 (pinned in `requirements.txt`) |

The code also runs on CUDA (selected automatically) or the CPU (`--device cpu`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login            # or: export WANDB_MODE=offline
```

CIFAR-10 is downloaded to `./data` automatically on the first run. If the download fails with an SSL error (python.org builds on macOS), either run `Install Certificates.command`, or fetch the archive manually:

```bash
mkdir -p data && curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar -xz -C data
```
