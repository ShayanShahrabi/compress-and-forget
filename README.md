# Compress and Forget: Evaluating Quantized LLMs on Interference-Heavy Retrieval

Official code and reproducibility artifacts for:

> **Compress and Forget: Evaluating Quantized LLMs on Interference-Heavy Retrieval**
---

## Overview

This repository contains the experimental pipeline used to study how post-training quantization affects proactive interference (PI) in instruction-tuned language models.

The main evaluation compares:

- **FP16**
- **INT8** using `bitsandbytes` LLM.int8
- **INT4/NF4** using `bitsandbytes` with double quantization

across:

- Qwen2.5-7B-Instruct
- Mistral-7B-Instruct-v0.3
- Phi-3.5-mini-instruct

The benchmark uses repeated key rebinding: an attribute is overwritten multiple times and the model must retrieve only the most recent value. Word-type attributes use semantically confusable distractors, while numeric attributes provide a control condition.

The experiment is paired at the trial level: for a fixed model and seed, FP16, INT8, and INT4 use the same generated trials. The released analysis therefore compares quantization conditions without item-difficulty variation as a confound.

## Main experimental design

The main task contains four word-type attributes:

- mood
- favorite color
- favorite animal
- occupation

and three numeric control attributes:

- temperature
- stock price
- page count

The interference level `k` counts all chronological updates, including the final correct value; `k−1` values are stale. Tested levels are {1, 2, 4, 8, 16, 32, 64, 96}, subject to model/attribute vocabulary caps.

Word-type candidates are filtered by checking their actual token spans in the filled prompt template. This avoids assuming that tokenization of a word in isolation is equivalent to tokenization of that word in context.

Each condition uses five random seeds. The released records contain 23,075 paired trial identities (69,225 precision-specific observations). Exact regeneration also depends on the ordered vocabulary and evaluator version: released manifests and available evaluator copies differ. Use the released reconstructed histories and matching manifests as replay references; see [reproducibility limits](docs/REPRODUCIBILITY.md).

## Installation

The original main experiments were run on an NVIDIA RTX 3090 (24 GB).

Create a clean environment and install the dependencies:

```bash
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the PyTorch build appropriate for your system first if the default pip resolution does not provide the desired CUDA build.

The environment used for the main experiments was approximately:

```text
Python      3.x
PyTorch     2.5.1+cu121
Transformers 5.12.1
bitsandbytes 0.49.2
GPU         NVIDIA RTX 3090 24 GB
```

The evaluator writes an environment report so deviations can be recorded.

## Running the main experiment

The evaluator supports:

```text
fp16
int8
int4
```

and the three model identifiers:

```text
qwen2.5-7b
mistral-7b
phi3.5-mini
```

Example:

```bash
python src/pi_llm_eval_v2.py \
    --model qwen2.5-7b \
    --quant int4 \
    --seed 42 \
    --out results_v2_qwen2.5-7b_int4_seed42.csv
```

Repeat for the desired model × precision × seed combinations.

The script downloads the model/tokenizer from Hugging Face on first use. Model weights are **not** included in this repository.

### Reproducing the full main experiment

The paper's main experiment consists of:

```text
3 models × 3 precision conditions × 5 seeds
```

with the same task generator and paired trial design.

Running the complete experiment requires substantial GPU time and disk space. The released `data/combined_raw.csv.gz` contains the corresponding trial-level results, so the paper's analyses can be reproduced without rerunning all model inference.

## Reproducing the revised manuscript analyses

Use the released `data/reconstructed_trials.csv`; do not overwrite it with a fresh evaluator replay. CPU analysis does not require model downloads. Install NumPy, pandas, SciPy, statsmodels and Matplotlib in your CPU environment (the full requirements also include GPU inference dependencies).

From the repository root, choose fresh output directories:

```bash
python3 -I analysis/audit_primary.py --out generated/primary
```

```bash
python3 -I src/revision_stats.py --data data --out generated/statistics
```

```bash
python3 -I analysis/build_figures.py --out generated/figures
```

The audit checks complete pairing and primary summaries. `revision_stats.py` computes paired per-level gaps, exact McNemar tests, trial-clustered GEE, repeated-entry sensitivity and seed summaries. The figure builder produces five PDF/PNG figures and CSV companions; its geometry panel uses a **preserved cleaned summary**, not raw hidden-state re-extraction. References are in `analysis/verified/` and `analysis/figure-data/`.

```bash
python3 -I -B -m unittest discover -s src -p 'test_*cpu.py'
```

```bash
python3 -I -B -m unittest discover -s src -p 'test_revision_stats.py'
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for artifact provenance, comparison tolerances and limits. The historical `reconstruct_trials.py`, `combine_all_results.py`, `compute_ies.py`, `effect_sizes_and_mixedmodel.py` and `make_plots.py` remain available for inspection. Some expect input files in the working directory. They are not interchangeable with the revised paired statistics and cleaned geometry summaries; the older mixed-model analysis is not the revised primary inference.

## Exploratory follow-up code

The repository also includes the scripts and protocol for the paper's exploratory sampled-decoding and hidden-state geometry pilots. The runners reuse the primary evaluator's prompts, tokenizer-filtered vocabularies, trial construction, model loading, and scoring.

- `src/run_sampling_followup.py`: paired FP16/INT8/INT4 sampled decoding.
- `src/run_representational_followup.py`: exploratory FP16/INT4 hidden-state geometry collection.
- `src/analyze_followup_geometry.py`: analysis of geometry CSVs produced by a run.
- `src/test_followup_cpu.py`: CPU-only checks for shared manifest and prompt helpers.
- `src/FOLLOWUP_EXPERIMENTS.md`: runtime requirements, commands, outputs, and interpretation caveats.

The released pilot completion CSVs and manifests are now included under `data/followup/sampling/`. Representation manifests and a compact cleaned summary are included, but raw geometry CSVs and centered-probe NPZ arrays are not. The existing runners write manifests and environment reports for new runs; their presence does not prove exact historical regeneration. Completion repeats are clustered within prompts; geometry results are descriptive and do not establish causal mediation. See the follow-up guide before running these exploratory experiments.

## LM-head ablation

The main INT4 loading path used by the project left the language-model head in full precision under the relevant bitsandbytes configuration. A follow-up experiment explicitly quantized the LM head as a sensitivity check. Its small additional net accuracy changes do not establish equivalence or a causal decomposition of the INT4 penalty.

The standalone V2 implementation is in:

```text
src/lmhead_ablation/
```

Run the experiment with:

```bash
python src/lmhead_ablation/run_lmhead_ablation_v2.py --help
```

and analyze the generated results with:

```bash
python src/lmhead_ablation/analyze_lmhead_ablation_v2.py --help
```

The released ablation results are under:

```text
data/ablation/
```

## Released results

The repository includes the main trial-level results in:

```text
data/combined_raw.csv.gz
```

as well as:

- tokenizer vocabulary manifests
- reconstructed trial sequences
- numeric token-length analysis
- LM-head ablation results
- paper figures
- IES report
- statistical analysis outputs

Primary paired statistics can be recomputed from released tables without repeating model inference. Follow-up audit summaries are preserved exports with the limitations described in the reproducibility guide; not every reported quantity has its exact historical producer or raw input available.

## Reproducibility notes

The main experiments use:

- greedy decoding (`do_sample=False`)
- explicit random seeds
- identical trial sequences across quantization conditions
- tokenizer-verified single-token word candidates
- exact-match style answer extraction
- paired statistical comparisons where appropriate

The main quantization configurations are:

### INT8

`bitsandbytes` LLM.int8 loading.

### INT4

`bitsandbytes` 4-bit loading with:

```text
quant_type = NF4
double_quant = True
compute_dtype = float16
```

No model weights are redistributed here; users should obtain the models from their respective official Hugging Face repositories and comply with their licenses.


## License

The code in this repository is released under the MIT License. Model weights and third-party software remain subject to their respective licenses.
