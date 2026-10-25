"""Build the report figures and summary tables from results/runs/*.json.

    python scripts/make_plots.py            # writes report/figures/*.png, results/summary.csv

These are local renderings of the same metrics that are logged to W&B
(train_loss, train_acc, val_loss, val_acc per epoch), so the report can be
regenerated without a W&B account.
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_experiments import family  # noqa: E402
from vgg6.utils import run_name  # noqa: E402

FIG = "report/figures"
# Categorical palette, fixed order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#d6e6f8", "#2a78d6", "#0b2f5c"])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "lines.linewidth": 1.6,
})


def load_runs():
    runs = {}
    for p in sorted(glob.glob("results/runs/*.json")):
        with open(p) as f:
            r = json.load(f)
        r["df"] = pd.DataFrame(r["history"])
        runs[r["name"]] = r
    return runs


def members(runs, group):
    """Finished runs that belong to an experiment family, in family order."""
    return [runs[n] for n in (run_name(c) for c in family(group)) if n in runs]


def curves(rs, labels, title, path, metrics=("train_loss", "val_loss", "train_acc", "val_acc")):
    """2x2 grid of per-epoch curves, one line per run."""
    names = {"train_loss": "Training loss", "val_loss": "Validation loss",
             "train_acc": "Training accuracy (%)", "val_acc": "Validation accuracy (%)"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, m in zip(axes.flat, metrics):
        for i, (r, lab) in enumerate(zip(rs, labels)):
            ax.plot(r["df"]["epoch"], r["df"][m], color=SERIES[i % len(SERIES)], label=lab)
        ax.set_title(names[m], loc="left")
        if "loss" in m:
            v = np.concatenate([r["df"][m].values for r in rs])
            lo, hi = np.nanmin(v), min(np.nanmax(v), 2.5)  # cap diverged / sigmoid spikes
            ax.set_ylim(max(0, lo * 0.9), hi * 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=min(len(labs), 6), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def bar(values, labels, title, path, xlabel="Best validation accuracy (%)"):
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(values) + 0.9))
    y = np.arange(len(values))
    ax.barh(y, values, color=SERIES[0], height=0.6)
    for yi, v in zip(y, values):
        ax.text(v + 0.4, yi, f"{v:.2f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(values) * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summary_table(runs):
    rows = []
    for r in runs.values():
        c, f = r["config"], r["final"]
        rows.append(dict(name=r["name"], group=c.get("group"), activation=c["activation"],
                         batch_norm=c["batch_norm"], optimizer=c["optimizer"],
                         momentum=c["momentum"], lr=c["lr"], batch_size=c["batch_size"],
                         epochs=c["epochs"], weight_decay=c["weight_decay"],
                         best_val_acc=f["best_val_acc"], best_epoch=f["best_epoch"],
                         test_acc=f["test_acc"], final_train_acc=f["final_train_acc"],
                         train_time_min=f["train_time_s"] / 60,
                         sec_per_epoch=r["df"]["epoch_time"].mean(),
                         epochs_to_80=_epochs_to(r["df"], 80)))
    df = pd.DataFrame(rows).sort_values("best_val_acc", ascending=False)
    df.to_csv("results/summary.csv", index=False, float_format="%.4g")
    return df


def _epochs_to(df, thr):
    hit = df.index[df["val_acc"] >= thr]
    return int(df.loc[hit[0], "epoch"]) if len(hit) else np.nan


def latex_table(rs, labels, path, header="Setting", timing=False):
    """booktabs table of the key metrics of a family of runs (input by report/report.tex)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    best = max(r["final"]["best_val_acc"] for r in rs)
    lines = [r"\begin{tabular}{lrrrrr" + ("r}" if timing else "}"), r"\toprule",
             f"{header} & Best val (\\%) & Test (\\%) & Best ep. & Ep. to 70\\% & "
             r"Final train (\%)" + (r" & s/epoch \\" if timing else r" \\"), r"\midrule"]
    for r, lab in zip(rs, labels):
        f, d = r["final"], r["df"]
        to70 = _epochs_to(d, 70)
        val = f"{f['best_val_acc']:.2f}"
        if f["best_val_acc"] == best:
            val = rf"\textbf{{{val}}}"
        lines.append(f"{lab} & {val} & {f['test_acc']:.2f} & {f['best_epoch']} & "
                     f"{'never' if np.isnan(to70) else int(to70)} & {f['final_train_acc']:.2f}"
                     + (f" & {d['epoch_time'].median():.1f}" if timing else "") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def parallel_coordinates(df, path):
    """Parallel-coordinates plot of all runs, coloured by best validation accuracy."""
    axes_spec = [("activation", None), ("optimizer", None), ("batch_norm", None),
                 ("batch_size", "log"), ("epochs", None), ("lr", "log"),
                 ("momentum", None), ("weight_decay", None), ("best_val_acc", None)]
    d = df.copy()
    d["optimizer"] = np.where((d["optimizer"] == "sgd") & (d["momentum"] == 0), "sgd(plain)",
                              np.where(d["optimizer"] == "sgd", "sgd+mom", d["optimizer"]))
    cols, ticks = [], []
    for col, scale in axes_spec:
        v = d[col]
        if not pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
            cats = sorted(v.astype(str).unique())
            pos = v.astype(str).map({c: i / max(len(cats) - 1, 1) for i, c in enumerate(cats)})
            cols.append(pos.values)
            ticks.append([(i / max(len(cats) - 1, 1), c) for i, c in enumerate(cats)])
            continue
        x = np.log10(v.astype(float)) if scale == "log" else v.astype(float)
        lo, hi = x.min(), x.max()
        cols.append(((x - lo) / (hi - lo if hi > lo else 1)).values)
        tv = sorted(v.unique())
        if len(tv) > 8:  # label a spread-out subset of the values actually used
            tv = [tv[i] for i in np.linspace(0, len(tv) - 1, 7).round().astype(int)]
        tx = [((np.log10(t) if scale == "log" else t) - lo) / (hi - lo if hi > lo else 1) for t in tv]
        ticks.append([(p, f"{t:g}") for p, t in zip(tx, tv)])
    M = np.vstack(cols).T
    acc = d["best_val_acc"].values
    norm = plt.Normalize(acc.min(), acc.max())
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for i in np.argsort(acc):  # best runs drawn last (on top)
        ax.plot(range(len(axes_spec)), M[i], color=SEQ(norm(acc[i])), lw=1.1, alpha=0.85)
    for j, tk in enumerate(ticks):
        ax.axvline(j, color=INK2, lw=0.8)
        for p, lab in tk:
            ax.text(j + 0.03, p, lab, fontsize=6.5, color=INK, va="center")
    ax.set_xticks(range(len(axes_spec)), [a for a, _ in axes_spec])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.set_xlim(-0.2, len(axes_spec) - 0.4)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=SEQ)
    fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025, label="Best val. accuracy (%)")
    ax.set_title(f"Parallel coordinates: all {len(d)} runs", loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def all_runs_panels(runs, df, path_prefix):
    """W&B-style panels: every run's train/val loss/accuracy vs. epoch, plus val_acc vs. run."""
    order = list(df.sort_values("best_val_acc")["name"])
    norm = plt.Normalize(df["best_val_acc"].min(), df["best_val_acc"].max())
    for m, lab in [("train_acc", "Training accuracy (%)"), ("train_loss", "Training loss"),
                   ("val_acc", "Validation accuracy (%)"), ("val_loss", "Validation loss")]:
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for n in order:
            r = runs[n]
            ax.plot(r["df"]["epoch"], r["df"][m], lw=1,
                    color=SEQ(norm(r["final"]["best_val_acc"])), alpha=0.8)
        if "loss" in m:
            ax.set_ylim(0, min(3.0, np.nanmax([runs[n]["df"][m].max() for n in order]) * 1.05))
        ax.set_xlabel("Epoch")
        ax.set_title(f"{m}: all {len(order)} runs (colour = best val. accuracy)", loc="left")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=ax, pad=0.01, fraction=0.04)
        fig.tight_layout()
        fig.savefig(f"{path_prefix}_{m}.png", bbox_inches="tight")
        plt.close(fig)

    # val_acc vs. run (creation order), with the running best as a step line.
    created = sorted(runs.values(), key=lambda r: os.path.getmtime(f"results/runs/{r['name']}.json"))
    acc = np.array([r["final"]["best_val_acc"] for r in created])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.scatter(range(1, len(acc) + 1), acc, s=22, color=SERIES[0], edgecolor="white", lw=0.8, zorder=3)
    ax.step(range(1, len(acc) + 1), np.maximum.accumulate(acc), where="post", color=SERIES[1],
            lw=1.6, label="best so far")
    ax.set_xlabel("Run (in order of creation)")
    ax.set_ylabel("Best val. accuracy (%)")
    ax.set_title("val_acc vs. created", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{path_prefix}_valacc_scatter.png", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    runs = load_runs()
    df = summary_table(runs)
    print(df[["name", "best_val_acc", "test_acc", "best_epoch", "sec_per_epoch"]].to_string(index=False))

    # Q2(a) activations, with and without BatchNorm
    for g, tag in [("activation", "with BatchNorm"), ("activation_nobn", "without BatchNorm")]:
        rs = members(runs, g)
        if rs:
            curves(rs, [r["config"]["activation"] for r in rs],
                   f"Activation functions ({tag}), momentum SGD, 20 epochs", f"{FIG}/{g}.png")
            latex_table(rs, [r["config"]["activation"] for r in rs], f"report/tables/{g}.tex",
                        "Activation")

    # Q2(b) optimizers: best LR of each, then the LR sensitivity bars
    rs = members(runs, "optimizer")
    if rs:
        def opt_label(r):
            c = r["config"]
            return "SGD (plain)" if c["optimizer"] == "sgd" else {
                "nesterov": "Nesterov SGD", "adam": "Adam", "adagrad": "Adagrad",
                "rmsprop": "RMSprop", "nadam": "Nadam"}[c["optimizer"]]
        best = {}
        for r in rs:
            k = opt_label(r)
            if k not in best or r["final"]["best_val_acc"] > best[k]["final"]["best_val_acc"]:
                best[k] = r
        sgdm = [r for r in members(runs, "lr")]
        if sgdm:
            best["SGD + momentum"] = max(sgdm, key=lambda r: r["final"]["best_val_acc"])
        keys = list(best)
        curves([best[k] for k in keys], [f"{k} (lr={best[k]['config']['lr']:g})" for k in keys],
               "Optimizers at their best learning rate, ReLU + BN, 20 epochs", f"{FIG}/optimizer.png")
        opt_names = {r["name"] for r in rs}
        bar([r["final"]["best_val_acc"] for r in rs + sgdm],
            [f"{opt_label(r) if r['name'] in opt_names else 'SGD + momentum'}  lr={r['config']['lr']:g}"
             for r in rs + sgdm],
            "Optimizer x learning rate", f"{FIG}/optimizer_lr_bars.png")
        all_opt = rs + sgdm
        latex_table(all_opt, [f"{opt_label(r) if r['name'] in opt_names else 'SGD + momentum'}, "
                              f"lr={r['config']['lr']:g}" for r in all_opt],
                    "report/tables/optimizer.tex", "Optimizer, learning rate")

    # Q2(c) batch size / learning rate / epochs
    for g, key, title in [("bs", "batch_size", "Batch size (lr=0.05 fixed)"),
                          ("lr", "lr", "Learning rate (momentum SGD, bs=128)"),
                          ("epochs", "epochs", "Number of epochs (cosine schedule)")]:
        rs = members(runs, g)
        if rs:
            curves(rs, [f"{key}={r['config'][key]:g}" for r in rs], title, f"{FIG}/{g}.png")
            latex_table(rs, [f"{r['config'][key]:g}" for r in rs], f"report/tables/{g}.tex",
                        key.replace("_", " "), timing=(g == "bs"))

    # Q1 best run
    best_name = df.iloc[0]["name"]
    if os.path.exists("configs/best.yaml"):
        import yaml
        from train import DEFAULTS
        with open("configs/best.yaml") as f:
            best_name = run_name(dict(DEFAULTS, **yaml.safe_load(f)))
    if best_name in runs:
        curves([runs[best_name]], ["best configuration"], f"Best run: {best_name}", f"{FIG}/best_run.png")

    # Q3 W&B-style panels
    parallel_coordinates(df, f"{FIG}/parallel_coordinates.png")
    all_runs_panels(runs, df, f"{FIG}/all_runs")


if __name__ == "__main__":
    main()
