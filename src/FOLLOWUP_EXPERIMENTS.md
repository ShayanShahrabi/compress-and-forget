# Follow-up experiments: representation and sampled decoding

These standalone experiments are designed for a CUDA Linux runtime such as the RTX 6000 Blackwell Molab environment. They do not modify `pi_llm_eval_v2.py` or the published paper. The scripts live in `src/` beside `pi_llm_eval_v2.py`; run commands from the repository root using the paths shown below. The actual model runs are GPU-intensive and should be run on Molab, not on the local development machine. For environments importing modules as a package, `src` may also be used as a package name.

## Protocol and outputs

Both experiments use the primary evaluator's model registry, task prompts, tokenizer-filtered semantic word pools, key-interference levels (Qwen 64, Mistral 16, Phi 8), and five existing seeds (42, 123, 999, 7, 2024). For each selected key-level trial, the manifest builder replays the primary evaluator's entire eligible RNG schedule in insertion order: word categories, then numeric categories; levels in protocol order; all trials at each eligible level. It selects key-level word trials only after consuming RNG for every preceding eligible trial, matching `pi_llm_eval_v2.py` even when resumed runs would skip inference. Each model-specific `trial_manifest.json` persists exact prompts, gold answers, pool and prompt SHA-256 hashes, and the replay schedule. Reuse that same manifest for every precision arm. Do not edit or replace it between arms. Separate manifests are intentional because tokenizer-filtered pools differ by model.

The first run downloads model weights into the Hugging Face cache and writes artifacts beneath `followup_results/` by default. Ensure the runtime has sufficient persistent storage for model caches and outputs, and copy the complete output directory off the ephemeral runtime before it is reset.

## Environment prerequisites

Use a Python environment with CUDA-enabled PyTorch, Transformers, Accelerate, bitsandbytes, pandas, and Matplotlib. The existing project does not pin Molab package versions. Install versions compatible with the offered CUDA runtime and Blackwell GPU, and record the script's `environment_report.json`; do not silently assume the RTX 3090 versions are compatible. The scripts check that CUDA is available, record GPU/software metadata, and fail before model loading if no CUDA GPU is visible. They use the same `load_model()` implementation as the primary evaluator: FP16, bitsandbytes INT8, and NF4 INT4 with double quantization and FP16 compute.

Confirm the selected Molab runtime exposes the GPU to a shell, has enough persistent storage, and can access Hugging Face model files. If a model requires access approval in Hugging Face, complete that outside the scripts before launch. Keep a copy of the terminal logs together with the artifacts.

## 1. Representational geometry (FP16 vs NF4 INT4)

Start with a small pilot for one model and five trials per semantic category:

```bash
cd llms_inference
python src/run_representational_followup.py --model mistral-7b --quant both --trials-per-category 5
```

After confirming artifacts and memory behavior, run all three models with a larger sample. The default selects five trials per category; use up to each model's full key-level budget (Mistral 60, Qwen 60, Phi 25) as time allows:

```bash
python src/run_representational_followup.py --model all --quant both --trials-per-category 60
```

Phi's key level has 25 trials in the primary protocol, so requesting 60 simply produces the available 25. For each prompt and precision, the script averages the contextual hidden states over the subword span of each value in every hidden-state output (embedding index 0 and transformer block outputs), then records every within-prompt value-pair cosine. Result files are under `followup_results/representation/<model>/geometry_{fp16,int4}.csv`, with manifest, vocabulary and environment sidecars. To make the layer-wise plot and matched summary after the runs:

```bash
python src/analyze_followup_geometry.py --input-dir followup_results/representation --output-dir followup_results/representation/analysis
```

The analysis writes trial/value-pair matched cosine data, layer summaries and PDF/PNG plots. Interpret the descriptive plot as exploratory evidence about representational geometry, not as proof that a geometric change causes behavioral PI. Hidden-state cosine is sensitive to tokenizer alignment and prompt context; preserve the manifest and the original CSVs.

## 2. Sampled decoding sensitivity (FP16, INT8, INT4)

Pilot all three quantization arms for one model and a small trial count:

```bash
python src/run_sampling_followup.py --model mistral-7b --quant all --trials-per-category 5 --repeats 5
```

Then scale up. The original Mistral and Qwen key levels allow 60 trials/category; Phi allows 25:

```bash
python src/run_sampling_followup.py --model all --quant all --trials-per-category 60 --repeats 5
```

Each completion uses `do_sample=True`, temperature 0.7 and top-p 0.9. The generation seed is deterministically derived from model, trial identity and repeat index and is shared across precision arms. Results are in `followup_results/sampling/<model>/sampling_{fp16,int8,int4}.csv`. They contain full trial identity, prompt hash, repeat seed, decoded response, extracted answer, correctness and latency. Five completions per prompt are repeated measurements clustered within prompt, not five independent prompts. Statistical inference must account for prompt-level clustering (e.g. average accuracy over the five completions per trial and then pair precision arms by trial; use an appropriate clustered or hierarchical method for population-level claims).

## Resume and safe operation

The experiment scripts append flushed CSV rows and skip keys already in that CSV when restarted. Resume only with the same manifest and exactly the same model/precision/output directory. If changing trial count, repeat count or protocol, use a new output directory (or archive the old run first). Never merge two runs whose manifests differ. Do not delete partial outputs until their contents have been copied and checked. If a process stops, restart the same command; validate row counts, unique pairing keys, and the prompt hashes before analyzing.

Run quantization arms sequentially in one process; the scripts release model references and empty the CUDA cache after each arm. A crash or out-of-memory event is not grounds to reduce precision-specific settings silently: first record it and retry a smaller pilot or fewer trials with an explicitly named fresh output directory.

## Transfer artifacts back for analysis

Copy the entire `followup_results/` directory, including each model's manifest, vocabulary manifest, environment report and raw CSVs, plus terminal logs if available. Do not send only figures or aggregated means. On the local checkout, run the CPU-only tests and geometry analysis, then share the resulting files for scientific review before revising `latex/main.tex`. The paper should not be changed until the runs have completed and paired data have passed validation.

## Local CPU-only checks (no model downloads)

From the repository root:

```bash
python -m unittest discover -s src -p 'test_followup_cpu.py'
python -m py_compile src/followup_common.py src/run_representational_followup.py src/run_sampling_followup.py src/analyze_followup_geometry.py
python src/run_representational_followup.py --help
python src/run_sampling_followup.py --help
python src/analyze_followup_geometry.py --help
```

These checks validate helpers and command-line interfaces only. They do not load weights, verify Molab compatibility, or substitute for a GPU pilot.
