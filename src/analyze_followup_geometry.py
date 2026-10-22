"""Summarize matched within-condition geometry and make a layer-wise plot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

QUANT_ORDER = ["fp16", "int4"]
QUANT_LABELS = {"fp16": "FP16", "int4": "NF4 INT4"}
QUANT_COLORS = {"fp16": "#2563eb", "int4": "#dc2626"}


def summarize(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for path in input_dir.glob("*/geometry_*.csv"):
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No geometry_*.csv files found under {input_dir}")
    data = pd.concat(frames, ignore_index=True)
    duplicate_rows = int(data.duplicated().sum())
    if duplicate_rows:
        print(f"Removing {duplicate_rows} exact duplicate rows; source CSVs are unchanged.")
        data = data.drop_duplicates(ignore_index=True)
    key = ["model", "seed", "attribute", "interference_level", "trial_idx",
           "left_value", "right_value", "layer_index"]
    if data.duplicated(key + ["quant_mode"]).any():
        raise ValueError("Duplicate trial/value-pair/layer/quant rows found")
    wide = data.pivot(index=key, columns="quant_mode", values="cosine_similarity").reset_index()
    if not set(QUANT_ORDER).issubset(wide.columns):
        raise ValueError("Both FP16 and INT4 matched data are required for comparison")
    wide["cosine_delta_int4_minus_fp16"] = wide["int4"] - wide["fp16"]
    wide.to_csv(output_dir / "matched_geometry_pairs.csv", index=False)
    summary = (data.groupby(["model", "quant_mode", "layer_index"], as_index=False)
               .agg(mean_cosine=("cosine_similarity", "mean"),
                    sd_cosine=("cosine_similarity", "std"), n_pairs=("cosine_similarity", "size")))
    summary.to_csv(output_dir / "geometry_layer_summary.csv", index=False)
    delta = (wide.groupby(["model", "layer_index"], as_index=False)
             .agg(mean_delta=("cosine_delta_int4_minus_fp16", "mean"),
                  sd_delta=("cosine_delta_int4_minus_fp16", "std"),
                  n_matched_pairs=("cosine_delta_int4_minus_fp16", "size")))
    delta.to_csv(output_dir / "geometry_layer_matched_delta.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    models = ["qwen2.5-7b", "mistral-7b", "phi3.5-mini"]
    labels = {"qwen2.5-7b": "Qwen2.5-7B", "mistral-7b": "Mistral-7B", "phi3.5-mini": "Phi-3.5-mini"}
    for ax, model in zip(axes, models):
        subset = summary[summary.model == model]
        for quant in QUANT_ORDER:
            line = subset[subset.quant_mode == quant].sort_values("layer_index")
            ax.plot(line.layer_index, line.mean_cosine, color=QUANT_COLORS[quant],
                    linewidth=1.8, label=QUANT_LABELS[quant])
        ax.set_title(labels[model])
        ax.set_xlabel("Transformer hidden-state index")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    axes[0].set_ylabel("Mean within-prompt value-pair cosine similarity")
    axes[-1].legend(frameon=False)
    fig.suptitle("Semantic-value representation geometry by precision")
    fig.text(0.5, 0.01, "Exploratory descriptive comparison; not causal evidence.", ha="center", color="#52514e")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(output_dir / "geometry_layer_cosine.png", dpi=200)
    fig.savefig(output_dir / "geometry_layer_cosine.pdf")
    plt.close(fig)
    return summary, delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="followup_results/representation")
    parser.add_argument("--output-dir", default="followup_results/representation/analysis")
    args = parser.parse_args()
    summary, delta = summarize(args.input_dir, args.output_dir)
    print(f"Wrote summaries for {len(summary)} model/quant/layer rows and {len(delta)} matched model/layer rows")


if __name__ == "__main__":
    main()
