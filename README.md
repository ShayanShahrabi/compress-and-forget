# Compress and Forget: bitsandbytes Quantization Amplifies Proactive Interference in LLMs

Official code and reproducibility artifacts for:

> **Compress and Forget: bitsandbytes Quantization Amplifies Proactive Interference in LLMs**  
> Shayan Shahrabi-Farahani, Dara Rahmati  

**Status:** research artifact for the current paper version. The main results use bitsandbytes FP16/INT8/INT4 (NF4).

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

## Repository structure

```text
pi-llm-quantization/
├── src/
│   ├── pi_llm_eval_v2.py              # Main FP16/INT8/INT4 evaluator
│   ├── reconstruct_trials.py          # Reconstructs deterministic trial sequences
│   ├── combine_all_results.py        # Combines per-run CSV files
│   ├── compute_ies.py                 # Interference Endurance Score
│   ├── compute_numeric_token_lengths.py
│   ├── effect_sizes_and_mixedmodel.py # Statistical analysis
│   ├── make_plots.py                  # Paper figures
│   └── lmhead_ablation/
│       ├── run_lmhead_ablation_v2.py
│       └── analyze_lmhead_ablation_v2.py
│
├── data/
│   ├── combined_raw.csv.gz            # Released main trial-level results
│   ├── reconstructed_trials.csv      # Deterministically reconstructed trials
│   ├── numeric_token_lengths.csv     # Numeric token-length analysis
│   ├── vocab_manifest_*.json         # Tokenizer-verified vocabularies
│   └── ablation/                     # LM-head ablation results
│
├── results/
│   ├── figures/                       # Figures used by the paper
│   └── reports/                       # IES/statistical/ablation reports
│
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

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

Interference levels are drawn from powers of two, with the maximum level determined by the available tokenizer-filtered vocabulary for each word-type attribute/model.

Word-type candidates are filtered by checking their actual token spans in the filled prompt template. This avoids assuming that tokenization of a word in isolation is equivalent to tokenization of that word in context.

Each condition uses five random seeds. Trial generation is deterministic for a fixed seed, and the same trial sequence is used across quantization levels.

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

## Reproducing the analyses

From the repository root:

### 1. Reconstruct trials

```bash
python src/reconstruct_trials.py
```

This creates:

```text
reconstructed_trials.csv
```

The reconstruction is used by the same-key intrusion analysis. The reconstructed subject/gold values should match the corresponding entries in the released raw results.

### 2. Combine result files

If you have generated per-condition result CSVs:

```bash
python src/combine_all_results.py
```

### 3. Compute IES

```bash
python src/compute_ies.py
```

The Interference Endurance Score is computed from the existing trial-level accuracy curves using trapezoidal integration over a log2-scaled interference axis.

### 4. Statistical analysis

```bash
python src/effect_sizes_and_mixedmodel.py
```

This contains the effect-size and mixed-effects analyses used for the paper.

### 5. Generate paper figures

Place/use `combined_raw.csv.gz` and `reconstructed_trials.csv` in the working directory and run:

```bash
python src/make_plots.py
```

The script generates:

- accuracy vs. interference level
- numeric-control accuracy
- key high-interference comparison
- same-key intrusion rate

in PDF and PNG formats.

## LM-head ablation

The main INT4 loading path used by the project left the language-model head in full precision under the relevant bitsandbytes configuration. A follow-up experiment explicitly quantized the LM head to determine whether the observed INT4 penalty was attributable to the output projection.

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

This is intended to make the reported analyses reproducible without requiring every model inference run to be repeated.

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

## Citation

If you use this code or benchmark, please cite the paper:

```bibtex
@article{shahrabi2026compress,
  title   = {Compress and Forget: bitsandbytes Quantization Amplifies Proactive Interference in LLMs},
  author  = {Shahrabi-Farahani, Shayan and Rahmati, Dara},
  journal = {arXiv preprint},
  year    = {2026}
}
```

A `CITATION.cff` file is also included.

## License

The code in this repository is released under the MIT License. Model weights and third-party software remain subject to their respective licenses.
