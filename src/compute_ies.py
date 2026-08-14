"""
Interference Endurance Score (IES) for FP16 / INT8 / INT4.

IES, as defined in the PI-LLM paper (Wang & Sun, arXiv:2506.08184, Section
2.2.2 / Figure 15): "the area under the curve (AUC) of retrieval accuracy,
calculated across log-scaled update counts." Higher IES = greater
robustness to interference.

This script computes IES for each (model, quant_mode) pair using the
full interference-level sweep already present in the main paper's
existing dataset (combined_raw.csv.gz) -- no new experiments needed,
since fp16/int8/int4 were run at full scope (all interference levels)
in the main paper protocol.

Definitions used here (stated explicitly so choices are auditable):
  - x-axis: log2(interference_level) (our levels are ~powers of 2:
    1, 2, 4, 8, 16, 32, 64, 96)
  - y-axis: retrieval accuracy (word-type attributes, the condition
    where the paper's main effect lives; numeric/control attributes are
    also reported separately for completeness)
  - AUC: trapezoidal integration (np.trapz) over the log-scaled x-axis
  - We report BOTH:
      * raw AUC (matches the PI-LLM paper's AUC-in-log-space quantity
        directly, for citing "our IES computed the same way as PI-LLM")
      * normalized AUC (raw AUC divided by the log-x range, giving a
        value in [0, 1] on the same scale as accuracy -- this makes IES
        directly comparable across models/arms even if their tested
        interference-level grids differ slightly, which matters if you
        later add IES for the int4_forced_lmhead arm at its reduced
        interference-level set)
  - Quantization penalty: delta_IES = IES(fp16) - IES(quant_mode), reported
    both in absolute IES units and as a percentage of IES(fp16).

Self-contained: only needs combined_raw.csv.gz (auto-detected in the
current directory, or pass --baseline explicitly). Produces a CSV of
per-level accuracy, a CSV of IES results, a markdown summary snippet,
and a chart.

Usage:
    python compute_ies.py
    python compute_ies.py --baseline ./combined_raw.csv.gz --out-dir ./ies_report
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# numpy >=2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


REQUIRED_COLS = [
    "model", "quant_mode", "seed", "attribute", "kind",
    "interference_level", "trial_idx", "subject", "gold",
    "model_raw_response", "model_extracted", "correct", "latency_sec",
]

QUANT_ARMS = ["fp16", "int8", "int4"]
REFERENCE_ARM = "fp16"


def find_baseline_file(explicit_path):
    if explicit_path:
        return explicit_path
    for c in ["./combined_raw.csv.gz", "./combined_raw.csv",
              "./combined_all_arms.csv.gz", "./combined_all_arms.csv"]:
        if os.path.exists(c):
            return c
    return None


def load_baseline(path):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            "No baseline file found. This script needs the main paper's "
            "combined_raw.csv.gz (fp16/int8/int4 arms across all "
            "interference levels). Put it in the current directory or "
            "pass --baseline /path/to/combined_raw.csv.gz."
        )
    compression = "gzip" if path.endswith(".gz") else None
    df = pd.read_csv(path, compression=compression)
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"baseline file missing expected columns: {missing}")
    df["correct"] = (
        df["correct"].astype(str).str.lower()
        .map({"true": True, "false": False})
        .fillna(df["correct"])
    )
    return df


def accuracy_by_level(df, model, quant_mode, kind):
    sub = df[(df["model"] == model) & (df["quant_mode"] == quant_mode) & (df["kind"] == kind)]
    if sub.empty:
        return pd.DataFrame(columns=["interference_level", "accuracy", "n_trials"])
    grouped = sub.groupby("interference_level")["correct"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["interference_level", "accuracy", "n_trials"]
    return grouped.sort_values("interference_level")


def compute_ies(level_acc_df):
    """Trapezoidal AUC of accuracy over log2(interference_level).
    Returns (raw_auc, normalized_auc, log_x_range) or (None, None, None)
    if fewer than 2 points are available (AUC undefined)."""
    if len(level_acc_df) < 2:
        return None, None, None
    x = np.log2(level_acc_df["interference_level"].values.astype(float))
    y = level_acc_df["accuracy"].values.astype(float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    raw_auc = float(_trapz(y, x))
    log_range = float(x.max() - x.min())
    normalized_auc = raw_auc / log_range if log_range > 0 else None
    return raw_auc, normalized_auc, log_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=None,
                         help="path to combined_raw.csv.gz; auto-detected if omitted")
    parser.add_argument("--out-dir", default="./ies_report")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    baseline_path = find_baseline_file(args.baseline)
    print(f"Using baseline file: {baseline_path}")
    df = load_baseline(baseline_path)

    models = sorted(df["model"].unique())
    kinds = sorted(df["kind"].unique())  # typically ["numeric", "word"]

    per_level_rows = []
    ies_rows = []

    for model in models:
        for kind in kinds:
            for quant_mode in QUANT_ARMS:
                level_acc = accuracy_by_level(df, model, quant_mode, kind)
                if level_acc.empty:
                    print(f"[warn] no data for model={model} quant_mode={quant_mode} kind={kind} -- skipping")
                    continue
                for _, row in level_acc.iterrows():
                    per_level_rows.append({
                        "model": model, "quant_mode": quant_mode, "kind": kind,
                        "interference_level": int(row["interference_level"]),
                        "accuracy": round(row["accuracy"], 4),
                        "n_trials": int(row["n_trials"]),
                    })
                raw_auc, norm_auc, log_range = compute_ies(level_acc)
                ies_rows.append({
                    "model": model, "quant_mode": quant_mode, "kind": kind,
                    "n_levels": len(level_acc),
                    "levels_tested": ",".join(str(int(l)) for l in level_acc["interference_level"]),
                    "IES_raw_auc": round(raw_auc, 4) if raw_auc is not None else None,
                    "IES_normalized": round(norm_auc, 4) if norm_auc is not None else None,
                    "log2_x_range": round(log_range, 4) if log_range is not None else None,
                })

    per_level_df = pd.DataFrame(per_level_rows)
    ies_df = pd.DataFrame(ies_rows)

    per_level_df.to_csv(os.path.join(args.out_dir, "accuracy_by_interference_level.csv"), index=False)
    ies_df.to_csv(os.path.join(args.out_dir, "ies_results.csv"), index=False)

    print("\n=== Accuracy by interference level ===")
    print(per_level_df.to_string(index=False))
    print("\n=== IES results ===")
    print(ies_df.to_string(index=False))

    # ---- Quantization penalty: delta_IES relative to fp16 ----
    penalty_rows = []
    for model in models:
        for kind in kinds:
            sub = ies_df[(ies_df["model"] == model) & (ies_df["kind"] == kind)]
            fp16_row = sub[sub["quant_mode"] == REFERENCE_ARM]
            if fp16_row.empty or fp16_row["IES_normalized"].isna().all():
                continue
            fp16_ies = fp16_row["IES_normalized"].values[0]
            for quant_mode in QUANT_ARMS:
                if quant_mode == REFERENCE_ARM:
                    continue
                q_row = sub[sub["quant_mode"] == quant_mode]
                if q_row.empty or q_row["IES_normalized"].isna().all():
                    continue
                q_ies = q_row["IES_normalized"].values[0]
                delta = fp16_ies - q_ies
                pct = (delta / fp16_ies * 100) if fp16_ies != 0 else None
                penalty_rows.append({
                    "model": model, "kind": kind, "quant_mode": quant_mode,
                    "IES_fp16": round(fp16_ies, 4), "IES_quant": round(q_ies, 4),
                    "delta_IES": round(delta, 4),
                    "pct_IES_lost": round(pct, 2) if pct is not None else None,
                })
    penalty_df = pd.DataFrame(penalty_rows)
    penalty_df.to_csv(os.path.join(args.out_dir, "ies_quantization_penalty.csv"), index=False)
    print("\n=== Quantization penalty (delta_IES vs FP16) ===")
    print(penalty_df.to_string(index=False))

    # ---- Chart: accuracy vs log-scaled interference level, per model, word-type ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        word_models = [m for m in models if not per_level_df[
            (per_level_df["model"] == m) & (per_level_df["kind"] == "word")].empty]
        colors = {"fp16": "#4C72B0", "int8": "#8172B2", "int4": "#C44E52"}

        if word_models:
            fig, axes = plt.subplots(1, len(word_models), figsize=(5 * len(word_models), 4.5), sharey=True)
            if len(word_models) == 1:
                axes = [axes]
            for ax, model in zip(axes, word_models):
                for quant_mode in QUANT_ARMS:
                    sub = per_level_df[
                        (per_level_df["model"] == model) &
                        (per_level_df["kind"] == "word") &
                        (per_level_df["quant_mode"] == quant_mode)
                    ].sort_values("interference_level")
                    if sub.empty:
                        continue
                    ax.plot(sub["interference_level"], sub["accuracy"], marker="o",
                            label=quant_mode, color=colors.get(quant_mode))
                ax.set_xscale("log", base=2)
                ax.set_xlabel("Interference level (log2 scale)")
                ax.set_ylabel("Accuracy")
                ax.set_ylim(0, 1)
                ax.set_title(model)
                ax.legend()
            fig.suptitle("Word-type retrieval accuracy vs. interference level (IES curves)")
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "fig_ies_curves.png"), dpi=150)
            print(f"\nSaved chart to {os.path.join(args.out_dir, 'fig_ies_curves.png')}")
    except ImportError:
        print("[warn] matplotlib not available; skipping chart.")

    # ---- Markdown summary ----
    lines = []
    lines.append("# Interference Endurance Score (IES): FP16 vs INT8 vs INT4\n")
    lines.append(
        "IES follows the PI-LLM definition (Wang & Sun, arXiv:2506.08184): "
        "the area under the accuracy-vs-interference-level curve, with the "
        "x-axis log-scaled. We report both the raw AUC (log2-x space, "
        "directly matching PI-LLM's quantity) and a normalized AUC (raw "
        "AUC divided by the log2 x-range, giving a value in [0,1] on the "
        "same scale as accuracy -- use this version when comparing arms "
        "tested over different interference-level ranges).\n"
    )
    lines.append("## IES by model, quant mode, and attribute kind\n")
    lines.append(ies_df.to_markdown(index=False))
    lines.append("\n\n## Quantization penalty (delta_IES relative to FP16)\n")
    lines.append(penalty_df.to_markdown(index=False) if not penalty_df.empty else "(no data)")
    lines.append(
        "\n\n## Suggested paper text (draft -- edit before pasting)\n\n"
        "> To allow standardized comparison with PI-LLM (Wang & Sun, 2025) "
        "and to quantify the quantization penalty in a single metric, we "
        "computed the Interference Endurance Score (IES) -- the area under "
        "the accuracy-vs-interference-level curve on a log-scaled x-axis "
        "-- for each model under FP16, INT8, and INT4. [EDIT: insert the "
        "specific delta_IES / pct_IES_lost numbers from the table above once "
        "reviewed; do not paste this placeholder as-is.]\n"
    )
    with open(os.path.join(args.out_dir, "ies_report.md"), "w") as f:
        f.write("\n".join(lines))

    print(f"\nAll outputs written to {args.out_dir}/")
    print("Share these files back to finish the write-up:")
    print(f"  - {args.out_dir}/accuracy_by_interference_level.csv")
    print(f"  - {args.out_dir}/ies_results.csv")
    print(f"  - {args.out_dir}/ies_quantization_penalty.csv")
    print(f"  - {args.out_dir}/ies_report.md")
    print(f"  - {args.out_dir}/fig_ies_curves.png")


if __name__ == "__main__":
    main()