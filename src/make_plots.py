"""
Generates the figures for the paper from combined_raw.csv.gz (and, for
Figure 4, reconstructed_trials.csv).

Usage:
    python make_plots.py
    # expects combined_raw.csv.gz in the same directory, and
    # reconstructed_trials.csv for fig4 (skipped with a warning if absent)

Outputs (both .pdf for LaTeX \\includegraphics and .png for quick viewing):
    fig1_accuracy_by_level.pdf/.png    -- word-type accuracy vs interference
                                           level, one panel per model, one
                                           line per quantization level
                                           [UNCHANGED from v1]
    fig2_numeric_control.pdf/.png      -- same layout, numeric-category
                                           (control) attributes
                                           [UNCHANGED from v1]
    fig3_key_level_comparison.pdf/.png -- grouped bar chart of fp16/int8/int4
                                           accuracy at each model's most-
                                           powered high-interference level,
                                           with 95% CI error bars.
                                           [CHANGED from v1: significance
                                           brackets now use McNemar's paired
                                           test (matching the paper's
                                           updated primary analysis) instead
                                           of Fisher's exact test, AND now
                                           show FP16-vs-INT8 significance in
                                           addition to FP16-vs-INT4 -- the
                                           paired test revealed a real INT8
                                           effect in 2/3 models that the
                                           unpaired Fisher's test had missed.]
    fig4_intrusion_rate.pdf/.png       -- [NEW] same-key intrusion rate
                                           (fraction of ALL word-type trials
                                           where a wrong answer exactly
                                           matches an earlier overwritten
                                           value) by quantization level,
                                           pooled across models. Visualizes
                                           the mechanism-level finding that
                                           quantization doesn't just lower
                                           accuracy, it specifically raises
                                           the rate of semantically-
                                           confusable intrusions.
"""

import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

RESULTS_PATH = "combined_raw.csv.gz"
RECON_PATH = "reconstructed_trials.csv"

MODEL_ORDER = ["qwen2.5-7b", "mistral-7b", "phi3.5-mini"]
MODEL_LABELS = {
    "qwen2.5-7b": "Qwen2.5-7B-Instruct",
    "mistral-7b": "Mistral-7B-Instruct-v0.3",
    "phi3.5-mini": "Phi-3.5-mini-instruct",
}
QUANT_ORDER = ["fp16", "int8", "int4"]
QUANT_LABELS = {"fp16": "FP16", "int8": "INT8", "int4": "INT4"}
QUANT_COLORS = {"fp16": "#2563eb", "int8": "#16a34a", "int4": "#dc2626"}

# Each model's most statistically powered high-interference level, chosen to
# avoid ceiling effects (too easy, no room for a gap to show) and floor
# effects (too hard, no room for a gap to show) -- see paper Section 4.3.
KEY_LEVELS = {"qwen2.5-7b": 64, "mistral-7b": 16, "phi3.5-mini": 8}


def load_data():
    with gzip.open(RESULTS_PATH, "rt") as f:
        df = pd.read_csv(f)
    return df


def stars_for_p(p_val):
    return "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."


def paired_mcnemar_pvalue(df, model, level, quant_a, quant_b, kind="word"):
    """Paired McNemar's test between two quant modes, at a given model and
    interference level, using (seed, attribute, trial_idx) as the pairing
    key -- matches the paper's updated primary significance test."""
    sub = df[(df.model == model) & (df.interference_level == level) & (df.kind == kind)]
    key_cols = ["seed", "attribute", "trial_idx"]
    a = sub[sub.quant_mode == quant_a].set_index(key_cols)["correct"]
    b = sub[sub.quant_mode == quant_b].set_index(key_cols)["correct"]
    joined = a.to_frame("a").join(b.to_frame("b"), how="inner")
    n11 = ((joined.a == 1) & (joined.b == 1)).sum()
    n10 = ((joined.a == 1) & (joined.b == 0)).sum()
    n01 = ((joined.a == 0) & (joined.b == 1)).sum()
    n00 = ((joined.a == 0) & (joined.b == 0)).sum()
    table = [[n11, n10], [n01, n00]]
    result = mcnemar(table, exact=(min(n10, n01) < 25), correction=True)
    return result.pvalue


def plot_accuracy_by_level(df, kind, out_stem, ylabel):
    """UNCHANGED from v1 -- raw accuracy curves are unaffected by which
    significance test is used to annotate Figure 3, so this plot doesn't
    need to change."""
    sub_all = df[df.kind == kind]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        sub = sub_all[sub_all.model == model]
        for quant in QUANT_ORDER:
            g = (
                sub[sub.quant_mode == quant]
                .groupby("interference_level")["correct"]
                .mean()
            )
            ax.plot(
                g.index, g.values, marker="o", markersize=4,
                label=QUANT_LABELS[quant], color=QUANT_COLORS[quant], linewidth=1.8,
            )
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.set_xlabel("Interference level (# prior updates)")
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="lower left", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_stem}.pdf")
    plt.savefig(f"{out_stem}.png", dpi=200)
    plt.close()
    print(f"Saved {out_stem}.pdf / .png")


def _bracket(ax, x1, x2, y, h, text, fontsize=9):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=fontsize)


def plot_key_level_comparison(df, out_stem="fig3_key_level_comparison"):
    """CHANGED from v1: significance now computed via paired McNemar's test
    (matching the paper's revised primary analysis) instead of Fisher's
    exact test, and BOTH the FP16-vs-INT8 and FP16-vs-INT4 comparisons are
    now annotated -- the paired test showed a real, previously-invisible
    INT8 effect in Mistral and Phi that the unpaired test in v1 missed."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bar_width = 0.25
    x = np.arange(len(MODEL_ORDER))

    bar_x = {}  # (model, quant) -> bar x-position, needed for bracket placement
    for qi, quant in enumerate(QUANT_ORDER):
        means, cis = [], []
        for model in MODEL_ORDER:
            level = KEY_LEVELS[model]
            sub = df[
                (df.model == model) & (df.kind == "word") &
                (df.interference_level == level) & (df.quant_mode == quant)
            ]["correct"]
            p, n = sub.mean(), len(sub)
            ci = 1.96 * np.sqrt(p * (1 - p) / n)
            means.append(p)
            cis.append(ci)
        offset = (qi - 1) * bar_width
        ax.bar(
            x + offset, means, bar_width, yerr=cis, capsize=3,
            label=QUANT_LABELS[quant], color=QUANT_COLORS[quant],
        )
        for xi, model in enumerate(MODEL_ORDER):
            bar_x[(model, quant)] = x[xi] + offset

    for xi, model in enumerate(MODEL_ORDER):
        level = KEY_LEVELS[model]

        # bar heights (needed to place brackets above the tallest bar + its CI)
        heights = {}
        for quant in QUANT_ORDER:
            sub = df[(df.model == model) & (df.kind == "word") &
                     (df.interference_level == level) & (df.quant_mode == quant)]["correct"]
            p, n = sub.mean(), len(sub)
            heights[quant] = p + 1.96 * np.sqrt(p * (1 - p) / n)

        p_int8 = paired_mcnemar_pvalue(df, model, level, "fp16", "int8")
        p_int4 = paired_mcnemar_pvalue(df, model, level, "fp16", "int4")

        base_y = max(heights.values()) + 0.05
        # lower bracket: FP16 vs INT8 (adjacent bars)
        _bracket(ax, bar_x[(model, "fp16")], bar_x[(model, "int8")],
                 base_y, 0.03, stars_for_p(p_int8))
        # upper bracket: FP16 vs INT4 (spans all three bars)
        _bracket(ax, bar_x[(model, "fp16")], bar_x[(model, "int4")],
                 base_y + 0.12, 0.03, stars_for_p(p_int4))

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{MODEL_LABELS[m]}\n(level={KEY_LEVELS[m]})" for m in MODEL_ORDER], fontsize=9
    )
    ax.set_ylabel("Accuracy (word-type attributes)")
    ax.set_ylim(0, 1.35)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    plt.tight_layout()
    plt.savefig(f"{out_stem}.pdf")
    plt.savefig(f"{out_stem}.png", dpi=200)
    plt.close()
    print(f"Saved {out_stem}.pdf / .png  (brackets: lower=FP16 vs INT8, upper=FP16 vs INT4, McNemar's test)")


def plot_intrusion_rate(df, out_stem="fig4_intrusion_rate"):
    """NEW figure. Requires reconstructed_trials.csv (distractor sequences)
    to classify wrong answers as same-key intrusions. Pooled across all
    three models and all four word-type attributes -- the point here is the
    aggregate mechanism-level effect, not per-model breakdown (Figure 3
    already covers per-model accuracy)."""
    if not os.path.exists(RECON_PATH):
        print(f"[fig4] Skipped: {RECON_PATH} not found in this directory.")
        return

    recon = pd.read_csv(RECON_PATH)
    key = ["model", "seed", "attribute", "interference_level", "trial_idx"]
    merged = df.merge(recon[key + ["distractor_sequence"]], on=key, how="left")
    word = merged[merged.kind == "word"].copy()

    def is_same_key_intrusion(row):
        if pd.isna(row.correct) or row.correct == 1:
            return False
        if pd.isna(row.model_extracted) or row.model_extracted == "":
            return False
        prior_values = row.distractor_sequence.split("|")[:-1]
        return str(row.model_extracted).lower() in [v.lower() for v in prior_values]

    word["is_intrusion"] = word.apply(is_same_key_intrusion, axis=1)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    rates, cis = [], []
    for quant in QUANT_ORDER:
        sub = word[word.quant_mode == quant]["is_intrusion"]
        p, n = sub.mean(), len(sub)
        rates.append(p)
        cis.append(1.96 * np.sqrt(p * (1 - p) / n))

    bars = ax.bar(QUANT_ORDER, rates, yerr=cis, capsize=4,
                   color=[QUANT_COLORS[q] for q in QUANT_ORDER])
    ax.set_xticks(range(len(QUANT_ORDER)))
    ax.set_xticklabels([QUANT_LABELS[q] for q in QUANT_ORDER])
    ax.set_ylabel("Same-key intrusion rate\n(fraction of all word-type trials)")
    ax.grid(alpha=0.25, axis="y")

    # FP16 vs INT4 paired McNemar test on the is_intrusion indicator,
    # pooled across models/seeds/attributes/levels
    key_cols = ["model", "seed", "attribute", "interference_level", "trial_idx"]
    a = word[word.quant_mode == "fp16"].set_index(key_cols)["is_intrusion"]
    b = word[word.quant_mode == "int4"].set_index(key_cols)["is_intrusion"]
    joined = a.to_frame("a").join(b.to_frame("b"), how="inner")
    n11 = ((joined.a) & (joined.b)).sum()
    n10 = ((joined.a) & (~joined.b)).sum()
    n01 = ((~joined.a) & (joined.b)).sum()
    n00 = ((~joined.a) & (~joined.b)).sum()
    result = mcnemar([[n11, n10], [n01, n00]], exact=(min(n10, n01) < 25), correction=True)

    y_top = max(r + c for r, c in zip(rates, cis)) + 0.015
    _bracket(ax, 0, 2, y_top, 0.01, stars_for_p(result.pvalue))
    ax.set_ylim(0, y_top + 0.05)

    plt.tight_layout()
    plt.savefig(f"{out_stem}.pdf")
    plt.savefig(f"{out_stem}.png", dpi=200)
    plt.close()
    print(f"Saved {out_stem}.pdf / .png  (FP16 vs INT4 McNemar p={result.pvalue:.3g})")


if __name__ == "__main__":
    df = load_data()
    plot_accuracy_by_level(
        df, kind="word", out_stem="fig1_accuracy_by_level",
        ylabel="Accuracy (word-type attributes)",
    )
    plot_accuracy_by_level(
        df, kind="numeric", out_stem="fig2_numeric_control",
        ylabel="Accuracy (numeric attributes, control)",
    )
    plot_key_level_comparison(df)
    plot_intrusion_rate(df)
    print("\nAll figures generated. Copy the .pdf files next to your .tex file "
          "and compile.")