"""
Revision statistics computed from the released trial-level data (no GPU needed).

 1. Per-level PAIRED accuracy gap (FP16 - INT4, percentage points) with 95% Wald CIs
    from the discordant pairs + exact McNemar p, for word-type and numeric trials.
 2. Trial-clustered logistic regression (GEE, exchangeable, robust SE; cluster = the
    prompt shared by the three precisions) of INT4 x log2(k), pooled and per model.
    (An earlier variational-Bayes mixed model treated the three precisions of one prompt
    as independent, which overstates precision.)
 3. Sensitivity of the key-level McNemar tests to trials in which a vocabulary entry
    occurs twice in the prompt (duplicate strings in the candidate pools).
 4. Per-seed direction check of the key-level FP16 > INT4 effect.

Usage (repo root):  python src/revision_stats.py --data data --out results_stats
Requires: pandas numpy scipy statsmodels
"""
import argparse, json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest
import statsmodels.api as sm
import statsmodels.formula.api as smf

KEY = {"qwen2.5-7b": 64, "mistral-7b": 16, "phi3.5-mini": 8}
NAMES = {"qwen2.5-7b": "Qwen2.5-7B", "mistral-7b": "Mistral-7B", "phi3.5-mini": "Phi-3.5-mini"}
MODELS = list(KEY)
LEVELS = [1, 2, 4, 8, 16, 32, 64, 96]
K = ["model", "seed", "attribute", "interference_level", "trial_idx"]


def validate_inputs(df, rt):
    """Validate paired observations and reconstruction; return the trial-wide data.

    Inputs are not modified. Numeric gold values are compared as strings, as in
    the released mixed word/numeric CSV columns. Repeated history values are valid.
    """
    required_df = K + ["kind", "subject", "gold", "quant_mode", "correct"]
    required_rt = K + ["subject", "gold", "distractor_sequence"]
    for name, frame, required in (("observations", df, required_df),
                                  ("reconstruction", rt, required_rt)):
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise ValueError(f"{name}: missing required columns {missing}")
        if frame.empty:
            raise ValueError(f"{name}: no trials")
        if frame[required].isna().any().any():
            raise ValueError(f"{name}: null required keys or metadata")
        if not frame.model.isin(MODELS).all():
            raise ValueError(f"{name}: unknown model")
        levels = pd.to_numeric(frame.interference_level, errors="coerce")
        if not (np.isfinite(levels) & (levels > 0) & (levels % 1 == 0)
                & frame.interference_level.eq(levels)
                & ~frame.interference_level.map(lambda x: isinstance(x, (bool, np.bool_)))).all():
            raise ValueError(f"{name}: interference_level must be a positive finite integer")
    if not df.kind.isin(["word", "numeric"]).all():
        raise ValueError("observations: unknown kind")
    if not df.quant_mode.isin(["fp16", "int8", "int4"]).all():
        raise ValueError("observations: unknown precision")
    if not df.correct.isin([0, 1]).all():
        raise ValueError("observations: correct must be binary (0 or 1)")
    if df.duplicated(K + ["quant_mode"]).any():
        raise ValueError("observations: duplicate trial/precision")
    if rt.duplicated(K).any():
        raise ValueError("reconstruction: duplicate trial")
    if not df.groupby(K).quant_mode.nunique().eq(3).all():
        raise ValueError("observations: each trial requires fp16, int8, and int4")
    metadata = df[K + ["kind", "subject", "gold"]].copy()
    metadata[["subject", "gold"]] = metadata[["subject", "gold"]].astype(str)
    if metadata.groupby(K)[["kind", "subject", "gold"]].nunique().ne(1).any().any():
        raise ValueError("observations: inconsistent kind, subject, or gold within trial")
    metadata = metadata.drop_duplicates(K)
    reconstructed = rt[required_rt].copy()
    reconstructed[["subject", "gold"]] = reconstructed[["subject", "gold"]].astype(str)
    matched = metadata.merge(reconstructed, on=K, how="outer", validate="one_to_one",
                             indicator=True, suffixes=("_obs", "_rt"))
    if not matched._merge.eq("both").all():
        raise ValueError("reconstruction: trial coverage must exactly match observations")
    if not (matched.subject_obs.eq(matched.subject_rt)
            & matched.gold_obs.eq(matched.gold_rt)).all():
        raise ValueError("reconstruction: subject or gold does not match observations")
    seq = reconstructed.distractor_sequence.astype(str).str.split("|")
    if not (seq.map(lambda s: all(value.strip() for value in s))
            & seq.str.len().eq(reconstructed.interference_level)
            & seq.str[-1].eq(reconstructed.gold)).all():
        raise ValueError("reconstruction: history must have k nonempty values ending in gold")
    reconstructed["dup"] = seq.map(lambda s: len(s) != len(set(s)))
    # Unlike pivot_table, pivot cannot silently average repeated observations.
    wide = df.pivot(index=K + ["kind"], columns="quant_mode", values="correct")
    wide = wide.reset_index().sort_values(K + ["kind"]).reset_index(drop=True)
    return wide.merge(reconstructed[K + ["dup"]], on=K, validate="one_to_one", sort=False)


def paired(s, q):
    n = len(s)
    b = int(((s.fp16 == 1) & (s[q] == 0)).sum()); c = int(((s.fp16 == 0) & (s[q] == 1)).sum())
    d = (b - c) / n
    se = np.sqrt(max(b + c - (b - c) ** 2 / n, 0)) / n
    p = binomtest(b, b + c, .5).pvalue if b + c else 1.0
    return dict(n=n, fp16=s.fp16.mean(), q=s[q].mean(), gap=d, lo=d - 1.96 * se, hi=d + 1.96 * se, n10=b, n01=c, p=p)


def sci(p):
    if p >= 1e-2: return f"{p:.3f}"
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results_stats")
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.data, "combined_raw.csv.gz"))
    rt = pd.read_csv(os.path.join(a.data, "reconstructed_trials.csv"))
    w = validate_inputs(df, rt)
    rt["seq"] = rt.distractor_sequence.astype(str).str.split("|")
    rt["dup"] = rt.seq.apply(lambda s: len(s) != len(set(s)))
    rt["gold_prior"] = rt.seq.apply(lambda s: s[-1] in s[:-1])
    os.makedirs(a.out, exist_ok=False)

    # ---- 1. per-level gaps
    rows = []
    for (m, kind, l), s in w.groupby(["model", "kind", "interference_level"]):
        rows.append(dict(model=m, kind=kind, level=l, **paired(s, "int4")))
    gaps = pd.DataFrame(rows); gaps.to_csv(os.path.join(a.out, "per_level_gaps_int4.csv"), index=False)
    for kind in ("word", "numeric"):
        lines = [r"\begin{tabular}{lccc}", r"\toprule", r"$k$ & " + " & ".join(NAMES[m] for m in MODELS) + r" \\", r"\midrule"]
        for l in LEVELS:
            cells = []
            for m in MODELS:
                r = gaps[(gaps.model == m) & (gaps.kind == kind) & (gaps.level == l)]
                if r.empty: cells.append("--"); continue
                r = r.iloc[0]
                star = "$^{*}$" if r.p < 0.05 else ""
                cells.append(f"{100*r.gap:+.1f} [{100*r.lo:+.1f}, {100*r.hi:+.1f}]{star}")
            if all(c == "--" for c in cells):
                continue  # level not tested for this condition (e.g. k=96 word-type)
            lines.append(f"{l} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(a.out, f"tab_gap_{kind}.tex"), "w").write("\n".join(lines) + "\n")

    # ---- 2. GEE
    d = df.copy(); d["lk"] = np.log2(d.interference_level)
    d["trial"] = d[K].astype(str).agg("|".join, axis=1); d["q"] = pd.Categorical(d.quant_mode, ["fp16", "int8", "int4"])
    g = []
    for kind in ("word", "numeric"):
        s = d[d.kind == kind].sort_values("trial")
        for scope in ["pooled"] + MODELS:
            sub = s if scope == "pooled" else s[s.model == scope]
            f = "correct ~ C(q)*lk" + ("+C(model)" if scope == "pooled" else "")
            r = smf.gee(f, groups="trial", data=sub, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
            for qq in ("int8", "int4"):
                t = [x for x in r.params.index if qq in x and "lk" in x][0]
                g.append(dict(kind=kind, scope=scope, quant=qq, beta=r.params[t], se=r.bse[t], p=r.pvalues[t],
                              lo=r.conf_int().loc[t, 0], hi=r.conf_int().loc[t, 1]))
    gee = pd.DataFrame(g); gee.to_csv(os.path.join(a.out, "gee_interactions.csv"), index=False)
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Scope & Word-type INT4$\times\log_2 k$ & Numeric INT4$\times\log_2 k$ \\", r"\midrule"]
    for scope in ["pooled"] + MODELS:
        cells = []
        for kind in ("word", "numeric"):
            r = gee[(gee.kind == kind) & (gee.scope == scope) & (gee.quant == "int4")].iloc[0]
            cells.append(f"{r.beta:+.3f} ({r.se:.3f}), $p$={sci(r.p)}")
        lines.append(("Pooled (model fixed effects)" if scope == "pooled" else NAMES[scope]) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tab_gee.tex"), "w").write("\n".join(lines) + "\n")

    # ---- 3. duplicate-entry sensitivity
    summ = dict(n_trials=int(len(rt)), n_dup=int(rt.dup.sum()), n_gold_prior=int(rt.gold_prior.sum()),
                dup_by_model_attr={f"{k[0]}/{k[1]}": int(v) for k, v in rt[rt.dup].groupby(["model", "attribute"]).size().items()})
    rows = []
    for m, l in KEY.items():
        s = w[(w.model == m) & (w.interference_level == l) & (w.kind == "word")]
        for lab, ss in (("all", s), ("excl", s[~s.dup])):
            for q in ("int8", "int4"):
                rows.append(dict(model=m, level=l, subset=lab, quant=q, **paired(ss, q)))
    dup = pd.DataFrame(rows); dup.to_csv(os.path.join(a.out, "dup_sensitivity.csv"), index=False)
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Model ($k$) & $n$ (all / excl.) & INT4 $n_{10}/n_{01}$ & $p$ & INT8 $n_{10}/n_{01}$ & $p$ \\", r"\midrule"]
    for m, l in KEY.items():
        a_ = dup[(dup.model == m) & (dup.subset == "all") & (dup.quant == "int4")].iloc[0]
        e4 = dup[(dup.model == m) & (dup.subset == "excl") & (dup.quant == "int4")].iloc[0]
        e8 = dup[(dup.model == m) & (dup.subset == "excl") & (dup.quant == "int8")].iloc[0]
        lines.append(f"{NAMES[m]} ({l}) & {int(a_.n)} / {int(e4.n)} & {int(e4.n10)} / {int(e4.n01)} & {sci(e4.p)} & {int(e8.n10)} / {int(e8.n01)} & {sci(e8.p)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tab_dup.tex"), "w").write("\n".join(lines) + "\n")

    # ---- 4. per-seed direction at key level
    seeds = []
    for m, l in KEY.items():
        s = w[(w.model == m) & (w.interference_level == l) & (w.kind == "word")]
        for sd, ss in s.groupby("seed"):
            seeds.append(dict(model=m, seed=sd, fp16=ss.fp16.mean(), int4=ss.int4.mean(), int8=ss.int8.mean()))
    seeds = pd.DataFrame(seeds); seeds.to_csv(os.path.join(a.out, "per_seed_key_level.csv"), index=False)
    summ["seeds_int4_below_fp16"] = int((seeds.int4 < seeds.fp16).sum()); summ["n_seed_comparisons"] = int(len(seeds))
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2)); print(gee.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
