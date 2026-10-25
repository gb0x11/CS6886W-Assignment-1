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

## Reproducing the results

All commands are run from the repository root with the virtual environment active.

### 1. Best configuration (Questions 1 and 4)

```bash
python train.py --config configs/best.yaml           # trains, logs to W&B, saves the checkpoint
python evaluate.py --checkpoint checkpoints/best_model.pt
```

`train.py` writes the checkpoint with the highest validation accuracy to `checkpoints/<run_name>.pt`; the copy committed as `checkpoints/best_model.pt` is that file for `configs/best.yaml`.

### 2. Configuration study (Question 2)

```bash
python run_experiments.py --group activation          # Q2(a): relu, sigmoid, tanh, silu, gelu
python run_experiments.py --group activation_nobn     # Q2(a): same, without BatchNorm
python run_experiments.py --group optimizer           # Q2(b): 6 optimizers x 3 learning rates
python run_experiments.py --group lr bs epochs        # Q2(c)
python run_experiments.py --group all --dry_run       # list every run without training
```

Every family changes one thing relative to a shared baseline: ReLU + BatchNorm, SGD with momentum 0.9, lr 0.05, weight decay 5e-4, batch size 128, 20 epochs, cosine LR decay, seed 42. A run is skipped when `results/runs/<run_name>.json` already exists, so the script can be resumed.

### 3. W&B sweep and plots (Question 3)

```bash
wandb sweep configs/sweep.yaml                         # prints the sweep id
wandb agent <entity>/cs6886w-vgg6-cifar10/<sweep_id> --count 8
python scripts/wandb_report.py                         # W&B report with the Q3 panels
python scripts/make_plots.py                           # local figures + LaTeX tables for the PDF
```

## Seeds and determinism

* `--seed` (default **42**) seeds Python, NumPy and PyTorch, the mini-batch shuffling and the augmentation RNG. Augmentation random numbers come from a seeded CPU `torch.Generator`, so they are identical across devices.
* The 45k/5k train/validation split uses its own fixed seed (0), so every run validates on the same 5,000 images.
* On CUDA, cuDNN is put in deterministic mode. On Apple MPS some kernels are not bit-exact between runs, so a re-run can differ from the reported accuracy by a few tenths of a percent.
