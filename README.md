# CS6886W Assignment 1: VGG6 on CIFAR-10

This repository explores how the **activation function, optimizer, batch size, number of epochs and learning rate** affect a small VGG6 network (≈0.26 M parameters) trained on CIFAR-10. All runs are logged to Weights & Biases.

**Author:** Gowtham B (CS25M508)
**W&B project:** https://forge.coreweave.com/wandb/cs25m508-indian-institute-of-technology/cs6886w-vgg6-cifar10
**W&B report (Question 3 plots):** https://forge.coreweave.com/cs25m508-indian-institute-of-technology/cs6886w-vgg6-cifar10/reports/CS6886W-A1:-VGG6-on-CIFAR-10-(Question-3-plots)--VmlldzoxODE3NjQ4Mw==
**Full report (PDF):** [`report/report.pdf`](report/report.pdf)

## Best result

| | |
|---|---|
| Configuration | ReLU, BatchNorm, Nadam lr 0.003, weight decay 5e-4, batch 64, 100 epochs, cosine LR decay, seed 42 (`configs/best.yaml`) |
| Best validation accuracy | **86.36 %** (epoch 100, 5,000 held-out training images) |
| Test top-1 accuracy | **85.72 %** (10,000 CIFAR-10 test images) |
| Checkpoint | `checkpoints/best_model.pt` |

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
* On CUDA, cuDNN is put in deterministic mode. On the Apple M1 Pro (MPS) used here, two runs of `configs/best.yaml` (checked for 2 epochs) produced identical losses and accuracies to every printed digit. On a different GPU or library version, floating-point differences can move the final accuracy by a few tenths of a percent.

## How performance varies across configurations

Every Question 2 family changes one setting of the baseline (ReLU, BatchNorm, SGD momentum 0.9, lr 0.05, batch 128, 20 epochs). Numbers are best validation accuracy (%). Full tables, curves and discussion are in `report/report.pdf`.

**Activation (Q2a):** ReLU 83.16, GELU 82.98, SiLU 80.62, tanh 70.48, sigmoid 59.92.
Non-saturating activations (ReLU, GELU, SiLU) keep gradients alive through all layers and win clearly. Tanh and sigmoid saturate (sigmoid's slope is at most 0.25), and after BatchNorm they operate mostly in their nearly linear range, so the network learns slower and expresses less. Sigmoid's validation accuracy is also very unstable.

**Optimizer (Q2b, best of three learning rates each):** Nadam 84.74 (lr 3e-3), Nesterov 84.14 (0.1), Adam 84.00 (3e-3), SGD + momentum 83.74 (0.2), RMSprop 81.24 (1e-3), Adagrad 79.82 (0.03), plain SGD 79.16 (0.2).
Momentum is the key ingredient: every method with a momentum term (Nadam, Nesterov, Adam, heavy ball) reaches 83 to 85 %, the ones without (plain SGD, RMSprop, Adagrad) 79 to 81 %. Nadam also converges fastest (70 % at epoch 7). Adaptive methods need learning rates 30 to 100 times smaller than SGD. Adagrad's ever-growing denominator makes its step size decay, so it stalls unless its learning rate is large.

**Batch size (Q2c, lr fixed):** 32: 84.26, 64: 84.12, 128: 83.16, 256: 79.86, 512: 76.14.
At a fixed number of epochs, smaller batches make many more updates (1,407 vs 88 per epoch) and converge further, at a higher time per epoch (24.1 s at 32 vs 17.2 s at 512). Large batches would need a proportionally larger learning rate.

**Learning rate (Q2c, momentum SGD):** 0.005: 70.98, 0.01: 75.76, 0.05: 83.16, 0.1: 82.72, 0.2: 83.74.
Too small a rate under-trains in 20 epochs; from 0.05 upward accuracy plateaus, and larger rates make the early validation curve noisier and slower to reach 70 %.

**Epochs (Q2c):** 5: 68.82, 10: 77.18, 20: 83.16, 40: 85.88.
More training helps with diminishing returns. Training accuracy stays below validation accuracy, so the small network is not over-fitting, which is why the final model uses 100 epochs.
