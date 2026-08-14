"""
Run this ON THE CLUSTER after your experiment runs finish, in the same
folder as your results_v2_*.csv files. It produces just two files to
upload, no matter how many model x quant x seed combinations you ran:

    combined_raw.csv.gz     -- every trial, all columns, gzip-compressed
                                (full fidelity, needed if we want to re-run
                                any statistical test at the trial level)
    summary_aggregated.csv  -- one row per (model, quant, seed, attribute,
                                interference_level): n, correct, accuracy
                                (tiny, a few KB, enough for sanity checks
                                and most plots)

Usage:
    python combine_results.py
    # then upload combined_raw.csv.gz and summary_aggregated.csv
"""

import glob
import gzip
import os

import pandas as pd


def main():
    files = sorted(glob.glob("results_v2_*.csv"))
    if not files:
        print("No results_v2_*.csv files found in the current directory.")
        return

    print(f"Found {len(files)} result files:")
    for fpath in files:
        print(f"  {fpath}  ({os.path.getsize(fpath)/1024:.1f} KB)")

    dfs = []
    for fpath in files:
        d = pd.read_csv(fpath)
        # Backfill a 'seed' column for any older files that predate seed tracking
        if "seed" not in d.columns:
            d["seed"] = -1  # -1 marks "seed unknown / pre-seed-tracking run"
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal combined rows: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Quant modes: {sorted(df['quant_mode'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    # ---- Full-fidelity combined file (compressed) ----
    raw_out = "combined_raw.csv.gz"
    with gzip.open(raw_out, "wt", newline="") as f:
        df.to_csv(f, index=False)
    print(f"\nSaved {raw_out}  ({os.path.getsize(raw_out)/1024:.1f} KB)")

    # ---- Small aggregated summary ----
    group_cols = ["model", "quant_mode", "seed", "attribute", "kind", "interference_level"]
    summary = (
        df.groupby(group_cols)["correct"]
        .agg(n="count", n_correct="sum")
        .reset_index()
    )
    summary["accuracy"] = summary["n_correct"] / summary["n"]
    summary_out = "summary_aggregated.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved {summary_out}  ({os.path.getsize(summary_out)/1024:.1f} KB, {len(summary)} rows)")

    print("\nDone. Upload just these two files:")
    print(f"  {raw_out}")
    print(f"  {summary_out}")


if __name__ == "__main__":
    main()