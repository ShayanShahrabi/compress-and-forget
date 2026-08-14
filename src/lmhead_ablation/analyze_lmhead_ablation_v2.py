"""
Analysis for the int4_forced_lmhead ablation (V2, standalone).

Self-contained: no imports from other project files, no separate packing
step required -- this script directly globs every per-seed result CSV
out of --results-dir itself.

It also tries to auto-find a baseline file (fp16/int4/int8 arms) so you
don't have to remember a flag: it looks for combined_raw.csv.gz (or
.csv) in the current directory and in --results-dir's parent, unless
you pass --baseline explicitly. If none is found, it still runs and
produces a descriptive-only report, clearly marking baseline
comparisons as unavailable -- it will NOT fabricate or estimate missing
baseline numbers.

Usage:
    python analyze_lmhead_ablation_v2.py --results-dir ./results_int4_forced_lmhead
    python analyze_lmhead_ablation_v2.py --results-dir ./results_int4_forced_lmhead --baseline ./combined_raw.csv.gz --extra-results-dir ./results_int4_nolmhead
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

KEY_LEVELS = {
    "qwen2.5-7b": 64,
    "mistral-7b": 16,
    "phi3.5-mini": 8,
}

REQUIRED_COLS = [
    "model", "quant_mode", "seed", "attribute", "kind",
    "interference_level", "trial_idx", "subject", "gold",
    "model_raw_response", "model_extracted", "correct", "latency_sec",
]

REFERENCE_ARMS = ["fp16", "int4"]


# ----------------------------------------------------------------------
# Auto-discovery
# ----------------------------------------------------------------------

def find_baseline_file(results_dir):
    """Looks for a baseline file in a few sensible places without being
    asked. Returns the first match, or None."""
    candidates = [
        "./combined_raw.csv.gz", "./combined_raw.csv",
        os.path.join(os.path.dirname(os.path.abspath(results_dir)), "combined_raw.csv.gz"),
        os.path.join(os.path.dirname(os.path.abspath(results_dir)), "combined_raw.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def load_new_results(results_dirs):
    """Globs every results_v2_*_seed*.csv in the given directories.
    Any quant_mode is picked up automatically; the quant_mode column
    disambiguates them."""
    if isinstance(results_dirs, str):
        results_dirs = [results_dirs]

    paths = []
    for d in results_dirs:
        if not os.path.isdir(d):
            print(f"[warn] results dir not found, skipping: {d}")
            continue
        paths.extend(sorted(glob.glob(os.path.join(d, "results_v2_*_seed*.csv"))))

    if not paths:
        raise FileNotFoundError(
            f"No results_v2_*_seed*.csv files found in {results_dirs}. "
            f"Check that the experiment run actually produced output there."
        )
    print(f"Found {len(paths)} result files:")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        missing = set(REQUIRED_COLS) - set(df.columns)
        if missing:
            print(f"  [warn] {p} missing columns {missing} -- skipping")
            continue
        print(f"  {p}: {len(df)} rows")
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid result files found (all had missing columns).")

    combined = pd.concat(dfs, ignore_index=True)
    combined["correct"] = (
        combined["correct"].astype(str).str.lower()
        .map({"true": True, "false": False})
        .fillna(combined["correct"])
    )
    print(f"  quant_modes found: {sorted(combined['quant_mode'].unique())}")
    return combined


def load_baseline(path):
    if path is None or not os.path.exists(path):
        return None
    compression = "gzip" if path.endswith(".gz") else None
    df = pd.read_csv(path, compression=compression)
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"baseline file missing expected columns: {missing}")
    return df


# ----------------------------------------------------------------------
# Descriptive stats
# ----------------------------------------------------------------------

def descriptive_table(df, key_levels=None):
    key_levels = key_levels or KEY_LEVELS
    rows = []
    for model, level in key_levels.items():
        sub = df[(df["model"] == model) & (df["interference_level"] == level) & (df["kind"] == "word")]
        if sub.empty:
            continue
        for qmode, g in sub.groupby("quant_mode"):
            rows.append({
                "model": model, "interference_level": level, "quant_mode": qmode,
                "n_trials": len(g), "accuracy": round(g["correct"].mean(), 4),
                "n_correct": int(g["correct"].sum()),
            })
    return pd.DataFrame(rows)


def paired_mcnemar(df, model, level, mode_a, mode_b):
    from statsmodels.stats.contingency_tables import mcnemar

    key_cols = ["seed", "attribute", "trial_idx"]
    a = df[(df["model"] == model) & (df["interference_level"] == level) &
           (df["kind"] == "word") & (df["quant_mode"] == mode_a)]
    b = df[(df["model"] == model) & (df["interference_level"] == level) &
           (df["kind"] == "word") & (df["quant_mode"] == mode_b)]

    merged = a.merge(b, on=key_cols, suffixes=("_a", "_b"))
    if merged.empty:
        return None, {
            "n_paired": 0,
            "note": f"No paired trials found between {mode_a!r} and {mode_b!r} "
                    f"at {model}/{level}. Check that trial-generation RNG order "
                    f"matched (same seeds, category order, levels) between arms.",
        }

    both_correct = int(((merged["correct_a"] == True) & (merged["correct_b"] == True)).sum())
    both_wrong = int(((merged["correct_a"] == False) & (merged["correct_b"] == False)).sum())
    a_only = int(((merged["correct_a"] == True) & (merged["correct_b"] == False)).sum())
    b_only = int(((merged["correct_a"] == False) & (merged["correct_b"] == True)).sum())

    table = [[both_correct, a_only], [b_only, both_wrong]]
    discordant_min = min(a_only, b_only)
    result = mcnemar(table, exact=(discordant_min < 25))

    return result, {
        "n_paired": len(merged),
        "acc_" + mode_a: round(merged["correct_a"].mean(), 4),
        "acc_" + mode_b: round(merged["correct_b"].mean(), 4),
        "both_correct": both_correct, "both_wrong": both_wrong,
        f"{mode_a}_only_correct": a_only, f"{mode_b}_only_correct": b_only,
        "statistic": float(result.statistic), "pvalue": float(result.pvalue),
        "exact_used": discordant_min < 25,
    }


def interpret(fp16_acc, lmhead_unquantized_acc, lmhead_quantized_acc):
    backbone_gap = fp16_acc - lmhead_unquantized_acc
    lmhead_gap = lmhead_unquantized_acc - lmhead_quantized_acc

    if backbone_gap <= 1e-9 and lmhead_gap <= 1e-9:
        return "no_penalty_detected", {"backbone_gap": round(backbone_gap, 4),
                                        "lmhead_gap": round(lmhead_gap, 4)}

    total_int4_penalty = fp16_acc - lmhead_quantized_acc
    lmhead_share = (lmhead_gap / total_int4_penalty) if total_int4_penalty > 1e-9 else None

    if lmhead_share is None:
        label = "undetermined"
    elif lmhead_share >= 0.5:
        label = "output_projection_driven"
    elif lmhead_share <= 0.1:
        label = "backbone_driven"
    else:
        label = "partial_contribution"

    return label, {
        "backbone_only_gap_vs_fp16": round(backbone_gap, 4),
        "additional_gap_from_quantizing_lmhead": round(lmhead_gap, 4),
        "total_gap_fp16_to_full_int4": round(total_int4_penalty, 4),
        "fraction_of_total_penalty_from_lmhead": round(lmhead_share, 3) if lmhead_share is not None else None,
    }


def make_bar_chart(desc_df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in KEY_LEVELS if m in desc_df["model"].unique()]
    if not models:
        print("[warn] no models with data at their key level -- skipping chart.")
        return
    quant_order = ["fp16", "int8", "int4", "int4_nolmhead", "int4_forced_lmhead"]
    default_colors = ["#4C72B0", "#8172B2", "#C44E52", "#55A868", "#DD8452"]
    colors = dict(zip(quant_order, default_colors))

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        level = KEY_LEVELS[model]
        sub = desc_df[(desc_df["model"] == model) & (desc_df["interference_level"] == level)]
        present = [q for q in quant_order if q in sub["quant_mode"].values]
        present += [q for q in sub["quant_mode"].values if q not in present]
        accs = [sub[sub["quant_mode"] == q]["accuracy"].values[0] for q in present]
        bar_colors = [colors.get(q, "#999999") for q in present]
        bars = ax.bar(present, accs, color=bar_colors)
        ax.set_title(f"{model}\n(level={level}, word-type)")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=9)

    fig.suptitle("Word-type retrieval accuracy across quantization arms")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def write_markdown_report(desc_df, mcnemar_results, out_path, baseline_used):
    lines = []
    lines.append("# Ablation: forced lm_head quantization (V2)\n")

    if baseline_used:
        lines.append(f"Baseline file used: `{baseline_used}`\n")
    else:
        lines.append("**No baseline file found/supplied** -- this report contains "
                     "descriptive stats for the new arm(s) only. Comparisons against "
                     "fp16/int4 are unavailable until a baseline is supplied.\n")

    lines.append("## Descriptive accuracy at each model's key interference level "
                  "(word-type attributes)\n")
    lines.append(desc_df.to_markdown(index=False) if not desc_df.empty else "(no data)")
    lines.append("\n\n## Paired McNemar's tests\n")

    interpretations = []
    for model, level in KEY_LEVELS.items():
        if model not in mcnemar_results:
            continue
        lines.append(f"\n### {model} (level={level})\n")
        for pair_name, stats in mcnemar_results.get(model, {}).items():
            lines.append(f"**{pair_name}**\n")
            lines.append("```")
            lines.append(json.dumps(stats, indent=2))
            lines.append("```\n")

        sub = desc_df[(desc_df["model"] == model) & (desc_df["interference_level"] == level)]
        acc_map = dict(zip(sub["quant_mode"], sub["accuracy"]))
        lmhead_unquantized_acc = acc_map.get("int4_nolmhead", acc_map.get("int4"))
        lmhead_quantized_acc = acc_map.get("int4_forced_lmhead")

        if "fp16" in acc_map and lmhead_unquantized_acc is not None and lmhead_quantized_acc is not None:
            label, detail = interpret(acc_map["fp16"], lmhead_unquantized_acc, lmhead_quantized_acc)
            lines.append(f"**Interpretation for {model}:** {label}\n")
            lines.append("```")
            lines.append(json.dumps(detail, indent=2))
            lines.append("```\n")
            interpretations.append((model, label, detail))
        else:
            lines.append(f"**Interpretation for {model}:** incomplete data -- "
                         f"need fp16, an lm_head-unquantized int4 arm, and "
                         f"int4_forced_lmhead all present. "
                         f"available quant_modes: {sorted(acc_map.keys())}\n")

    lines.append("\n## Suggested Limitations-section paragraph (draft -- edit before pasting)\n")
    if interpretations:
        summary_bits = "; ".join(f"{m}: {l}" for m, l, d in interpretations)
        lines.append(
            "> We conducted a follow-up ablation directly quantizing the output "
            "projection (`lm_head`) to 4-bit precision, having found that our "
            "primary `int4` condition left `lm_head` at full precision by "
            "default. Comparing FP16, backbone-only INT4, and INT4 with "
            "`lm_head` also quantized isolates the output projection's "
            f"contribution to the interference penalty. Results were "
            f"{summary_bits}. [EDIT: convert this into 1-2 narrative sentences "
            "once you've reviewed the numbers above; do not paste the bracketed "
            "placeholder as-is.]\n"
        )
    else:
        lines.append("> [Incomplete arms for a full comparison -- fill in once "
                      "all arms are present.]\n")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown report to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True,
                         help="directory with results_v2_*_seed*.csv from run_lmhead_ablation_v2.py")
    parser.add_argument("--extra-results-dir", action="append", default=[],
                         help="additional results directories to include (e.g. an "
                              "earlier int4_nolmhead run), can be passed multiple times")
    parser.add_argument("--baseline", default=None,
                         help="path to combined_raw.csv.gz; auto-detected if omitted")
    parser.add_argument("--out-dir", default="./ablation_report_v2")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_dirs = [args.results_dir] + args.extra_results_dir
    new_df = load_new_results(all_dirs)

    baseline_path = args.baseline or find_baseline_file(args.results_dir)
    baseline_df = load_baseline(baseline_path)
    if baseline_df is not None:
        print(f"Using baseline file: {baseline_path}")
    else:
        print("No baseline file found or supplied -- proceeding with new-arm-only analysis. "
              "Pass --baseline /path/to/combined_raw.csv.gz to enable fp16/int4 comparisons.")

    if baseline_df is not None:
        common_cols = [c for c in REQUIRED_COLS if c in baseline_df.columns and c in new_df.columns]
        full_df = pd.concat([baseline_df[common_cols], new_df[common_cols]], ignore_index=True)
    else:
        full_df = new_df

    full_df.to_csv(os.path.join(args.out_dir, "combined_all_arms.csv.gz"),
                    index=False, compression="gzip")

    models_present = [m for m in KEY_LEVELS if m in full_df["model"].unique()]
    desc = descriptive_table(full_df)
    desc.to_csv(os.path.join(args.out_dir, "descriptive_key_level_accuracy.csv"), index=False)
    print("\n=== Descriptive accuracy at key levels ===")
    print(desc.to_string(index=False) if not desc.empty else "(no data)")

    mcnemar_results = {}
    for model, level in KEY_LEVELS.items():
        if model not in models_present:
            continue
        mcnemar_results[model] = {}
        quant_modes_present = set(full_df[full_df["model"] == model]["quant_mode"].unique())
        non_reference_arms = sorted(quant_modes_present - set(REFERENCE_ARMS))

        for arm in non_reference_arms:
            for ref in REFERENCE_ARMS:
                pair_key = f"{arm}_vs_{ref}"
                if ref in quant_modes_present:
                    _, stats = paired_mcnemar(full_df, model, level, ref, arm)
                    mcnemar_results[model][pair_key] = stats
                else:
                    mcnemar_results[model][pair_key] = {
                        "note": f"'{ref}' arm not present -- supply --baseline to enable this comparison"
                    }

        int4_family = sorted(a for a in quant_modes_present if a.startswith("int4") and a not in REFERENCE_ARMS)
        for i in range(len(int4_family)):
            for j in range(i + 1, len(int4_family)):
                a, b = int4_family[i], int4_family[j]
                pair_key = f"{a}_vs_{b}"
                _, stats = paired_mcnemar(full_df, model, level, b, a)
                mcnemar_results[model][pair_key] = stats

    with open(os.path.join(args.out_dir, "mcnemar_results.json"), "w") as f:
        json.dump(mcnemar_results, f, indent=2)
    print("\n=== McNemar results ===")
    print(json.dumps(mcnemar_results, indent=2))

    try:
        make_bar_chart(desc, os.path.join(args.out_dir, "fig_lmhead_ablation_comparison.png"))
    except ImportError:
        print("[warn] matplotlib not available; skipping chart generation.")

    write_markdown_report(desc, mcnemar_results,
                           os.path.join(args.out_dir, "ablation_report_v2.md"),
                           baseline_used=baseline_path)

    print(f"\nAll outputs written to {args.out_dir}/")
    print("Share these files back to finish the write-up:")
    print(f"  - {args.out_dir}/descriptive_key_level_accuracy.csv")
    print(f"  - {args.out_dir}/mcnemar_results.json")
    print(f"  - {args.out_dir}/ablation_report_v2.md")
    print(f"  - {args.out_dir}/fig_lmhead_ablation_comparison.png")


if __name__ == "__main__":
    main()