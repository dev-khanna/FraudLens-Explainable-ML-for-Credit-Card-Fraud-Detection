"""Matplotlib/seaborn figures with a shared, quiet dashboard palette."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

INK = "#dce5f4"
MUTED = "#99a9be"
PANEL = "#121d30"
ACCENT = "#5dd6c0"
RED = "#ff7d8b"
AMBER = "#efbf70"


def apply_theme():
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        "figure.facecolor": PANEL, "axes.facecolor": PANEL,
        "axes.edgecolor": "#2c3c54", "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": "#25344a", "grid.alpha": .45, "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.facecolor": PANEL, "legend.edgecolor": "#2c3c54",
        "savefig.facecolor": PANEL,
    })


def risk_distribution(scores, threshold):
    apply_theme()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.hist(np.asarray(scores) * 100, bins=np.linspace(0, 100, 31), color=ACCENT, alpha=.85)
    ax.axvline(threshold * 100, color=RED, linestyle="--", linewidth=1.6,
               label=f"Flag threshold · {threshold:.0%}")
    ax.set(xlabel="Model risk score (%)", ylabel="Transactions", xlim=(0, 100))
    ax.legend(loc="upper center")
    fig.tight_layout()
    return fig


def confusion_figure(metrics):
    apply_theme()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    sns.heatmap(matrix, annot=True, fmt=",d", cmap="crest", cbar=False, ax=ax,
                xticklabels=["Not flagged", "Flagged"],
                yticklabels=["Legitimate", "Fraud"], linewidths=1,
                annot_kws={"fontsize": 13})
    ax.set(xlabel="Model result", ylabel="Known dataset label")
    fig.tight_layout()
    return fig


def performance_curves(curves):
    apply_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, key, xlabel, ylabel in [
        (axes[0], "pr", "Recall", "Precision"),
        (axes[1], "roc", "False positive rate", "True positive rate"),
    ]:
        ax.plot(curves[key]["x"], curves[key]["y"], color=ACCENT, linewidth=2)
        ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.03))
    axes[0].set_title("Precision–recall")
    axes[1].set_title("ROC")
    axes[1].plot([0, 1], [0, 1], color=MUTED, linestyle="--", alpha=.5)
    fig.tight_layout()
    return fig


def global_importance(values, names):
    apply_theme()
    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[-15:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(np.asarray(names)[order], importance[order], color=ACCENT)
    ax.set_xlabel("Mean absolute SHAP contribution (log-odds)")
    fig.tight_layout()
    return fig


def reference_comparison(raw_row, features, reference):
    apply_theme()
    fig, axes = plt.subplots(len(features), 1, figsize=(9, .85 * len(features) + .6), squeeze=False)
    for ax, feature in zip(axes[:, 0], features):
        stats = reference[feature]
        value = float(raw_row[feature])
        ax.plot([stats["p01"], stats["p99"]], [0, 0], color=MUTED, linewidth=3)
        ax.plot([stats["p25"], stats["p75"]], [0, 0], color=ACCENT, linewidth=9, alpha=.5)
        ax.scatter([stats["median"]], [0], color=INK, marker="|", s=180, zorder=3)
        ax.scatter([value], [0], color=RED, marker="D", s=45, zorder=4)
        ax.set_yticks([])
        ax.set_ylabel(feature, rotation=0, ha="right", va="center", labelpad=18)
        ax.set_ylim(-.6, .6)
        ax.grid(axis="x")
        ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


def waterfall_figure(explanation):
    import shap
    apply_theme()
    plt.figure(figsize=(9, 5.5))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    # SHAP creates twin axes; keep their labels readable on the dark surface.
    for ax in fig.axes:
        ax.tick_params(colors=MUTED)
        for text in ax.texts:
            if isinstance(text.get_color(), str) and text.get_color() in ("#999999", "#bbbbbb"):
                text.set_color(MUTED)
    return fig
