"""
Effect sizes + mixed-effects analysis for the quantization x interference
paper. Addresses two reviewer-facing improvements:

  1. Report odds ratios with 95% CIs, not just p-values, for every
     comparison (per-model key-level comparisons AND the pooled test).
  2. Replace the per-model "most-powered single level" analysis with a
     mixed-effects logistic regression using ALL interference levels at
     once, with model and seed as random effects. This directly answers
     the post-hoc-level-selection concern: the interaction term
     quant_mode x log2(interference_level) tests whether the quantization
     penalty grows with interference, using the full dataset rather than
     one hand-picked level per model.

Usage:
    python effect_sizes_and_mixedmodel.py
    # expects combined_raw.csv.gz in the same directory
"""

import gzip

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.stats.contingency_tables import StratifiedTable, Table2x2

RESULTS_PATH = "combined_raw.csv.gz"

KEY_LEVELS = {"qwen2.5-7b": 64, "mistral-7b": 16, "phi3.5-mini": 8}


def load_data():
    with gzip.open(RESULTS_PATH, "rt") as f:
        return pd.read_csv(f)


# ---------------------------------------------------------------------
# Part 1: effect sizes (odds ratios) with 95% CIs
# ---------------------------------------------------------------------

def effect_size_table(df):
    print("=" * 78)
    print("PART 1: Effect sizes (odds ratios, FP16 vs INT4) with 95% CIs")
    print("=" * 78)

    tables_for_pooling = []
    for model, level in KEY_LEVELS.items():
        sub = df[(df.model == model) & (df.kind == "word") & (df.interference_level == level)]
        fp16 = sub[sub.quant_mode == "fp16"]["correct"]
        int4 = sub[sub.quant_mode == "int4"]["correct"]
        table = [[fp16.sum(), len(fp16) - fp16.sum()], [int4.sum(), len(int4) - int4.sum()]]
        t2x2 = Table2x2(table)
        or_val = t2x2.oddsratio
        ci_lo, ci_hi = t2x2.oddsratio_confint()
        p_val = scipy_stats.fisher_exact(table)[1]
        tables_for_pooling.append(table)
        print(f"\n{model} (level={level}, n={len(fp16)} per arm):")
        print(f"  FP16 accuracy: {fp16.mean():.3f}   INT4 accuracy: {int4.mean():.3f}")
        print(f"  Odds ratio (FP16 vs INT4): {or_val:.3f}  95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  Fisher exact p-value: {p_val:.4g}")

    st = StratifiedTable(tables_for_pooling)
    pooled_or = st.oddsratio_pooled
    pooled_ci = st.oddsratio_pooled_confint()
    mh_result = st.test_null_odds()
    print(f"\nPooled (Mantel-Haenszel) across all 3 models:")
    print(f"  Common odds ratio: {pooled_or:.3f}  95% CI [{pooled_ci[0]:.3f}, {pooled_ci[1]:.3f}]")
    print(f"  MH test statistic: {mh_result.statistic:.3f}, p-value: {mh_result.pvalue:.3g}")


# ---------------------------------------------------------------------
# Part 2: mixed-effects logistic regression, ALL interference levels
# ---------------------------------------------------------------------

def mixed_effects_model(df, kind, label, random_slopes=False):
    print("\n" + "=" * 78)
    slope_note = " (random-slopes robustness variant)" if random_slopes else ""
    print(f"PART 2: Mixed-effects logistic regression -- {label}{slope_note}")
    print("=" * 78)
    print("Model: correct ~ quant_mode * log2(interference_level)")
    if random_slopes:
        print("Random intercepts: model, seed, attribute")
        print("Random slopes: log2(interference_level) by model, by attribute")
    else:
        print("Random intercepts: model, seed, attribute")
    print("(Uses ALL interference levels simultaneously -- no per-model level selection.)\n")

    sub = df[df.kind == kind].copy()
    sub["log2_level"] = np.log2(sub["interference_level"])
    sub["quant_mode"] = pd.Categorical(sub["quant_mode"], categories=["fp16", "int8", "int4"])
    sub["model"] = sub["model"].astype(str)
    sub["seed"] = sub["seed"].astype(str)
    sub["attribute"] = sub["attribute"].astype(str)

    formula = 'correct ~ C(quant_mode, Treatment("fp16")) * log2_level'
    vc_formula = {"model": "0 + C(model)", "seed": "0 + C(seed)", "attribute": "0 + C(attribute)"}
    if random_slopes:
        vc_formula["model_slope"] = "0 + C(model):log2_level"
        vc_formula["attribute_slope"] = "0 + C(attribute):log2_level"

    glmm = BinomialBayesMixedGLM.from_formula(formula, vc_formula, data=sub)
    result = glmm.fit_vb()

    names = result.model.exog_names
    means = result.fe_mean
    sds = result.fe_sd
    print(f"{'Term':<55} {'Coef':>8} {'SD':>7} {'OR':>7} {'95% CI':>18} {'z':>7} {'p':>9}")
    for name, mean, sd in zip(names, means, sds):
        z = mean / sd
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        or_val = np.exp(mean)
        ci_lo, ci_hi = np.exp(mean - 1.96 * sd), np.exp(mean + 1.96 * sd)
        print(f"{name:<55} {mean:>8.3f} {sd:>7.3f} {or_val:>7.3f} "
              f"[{ci_lo:>5.3f}, {ci_hi:>5.3f}] {z:>7.2f} {p:>9.3g}")

    print("\nInterpretation guide:")
    print("  - The 'int4:log2_level' interaction term is the key test: a negative,")
    print("    significant coefficient means the INT4 accuracy penalty (relative to")
    print("    FP16) GROWS as interference level increases -- i.e. quantization")
    print("    specifically hurts retrieval more under heavier interference, using")
    print("    the full dataset rather than a single hand-picked level per model.")
    print("  - The 'int8:log2_level' interaction term is the same test for INT8,")
    print("    expected to be small/non-significant given the 'cliff at INT4' pattern.")

    return result


if __name__ == "__main__":
    df = load_data()
    effect_size_table(df)
    # Primary model (paper Table 5): random intercepts for model, seed, attribute.
    mixed_effects_model(df, kind="word", label="word-type attributes (main hypothesis)")
    mixed_effects_model(df, kind="numeric", label="numeric attributes (specificity control)")
    # Robustness check (paper Appendix E): adds random slopes for log2(level) by model and attribute.
    mixed_effects_model(df, kind="word", label="word-type attributes (main hypothesis)", random_slopes=True)
    mixed_effects_model(df, kind="numeric", label="numeric attributes (specificity control)", random_slopes=True)