"""Exploratory FP16-vs-NF4 hidden-state geometry experiment."""

import argparse
import csv
import json
import itertools
import time
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from .followup_common import (
        environment_report, filtered_vocab, load_or_create_manifest, load_quantized_model,
        prompt_inputs, require_cuda, token_span, update_values,
    )
except ImportError:
    from followup_common import (
    environment_report, filtered_vocab, load_or_create_manifest, load_quantized_model,
    prompt_inputs, require_cuda, token_span, update_values,
)


def run_arm(model_key, quant, trials, out_path):
    tokenizer, model = load_quantized_model(model_key, quant)
    fields = ["model", "quant_mode", "seed", "attribute", "interference_level", "trial_idx",
              "prompt_sha256", "left_value", "right_value", "layer_index", "cosine_similarity",
              "n_value_tokens_left", "n_value_tokens_right"]
    done = set()
    if out_path.exists():
        with out_path.open(newline="", encoding="utf-8") as handle:
            done = {(r["seed"], r["attribute"], r["trial_idx"], r["left_value"],
                     r["right_value"], r["layer_index"]) for r in csv.DictReader(handle)}
    write_header = not out_path.exists()
    started = time.time()
    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for trial_num, row in enumerate(trials, 1):
            rendered, input_ids, attention_mask, offsets = prompt_inputs(tokenizer, row["prompt"], model.device)
            spans = []
            history = update_values(row)
            for value in history:
                spans.append(token_span(rendered, row["prompt"], value, offsets))
            with torch.inference_mode():
                output = model(input_ids=input_ids, attention_mask=attention_mask,
                               output_hidden_states=True, use_cache=False, return_dict=True)
            hidden_states = output.hidden_states
            # HF tuple index 0 is the embedding output; block outputs are 1..N.
            for layer_idx, hidden in enumerate(hidden_states):
                reps = [hidden[0, positions, :].mean(dim=0) for positions in spans]
                for left, right in itertools.combinations(range(len(history)), 2):
                    key = (str(row["seed"]), row["attribute"], str(row["trial_idx"]),
                           history[left], history[right], str(layer_idx))
                    if key in done:
                        continue
                    similarity = F.cosine_similarity(reps[left].float()[None, :],
                                                     reps[right].float()[None, :], dim=-1).item()
                    writer.writerow({
                        "model": model_key, "quant_mode": quant, "seed": row["seed"],
                        "attribute": row["attribute"], "interference_level": row["interference_level"],
                        "trial_idx": row["trial_idx"], "prompt_sha256": row["prompt_sha256"],
                        "left_value": history[left], "right_value": history[right],
                        "layer_index": layer_idx, "cosine_similarity": similarity,
                        "n_value_tokens_left": len(spans[left]), "n_value_tokens_right": len(spans[right]),
                    })
                handle.flush()
            del output, hidden_states
            if trial_num % 10 == 0:
                print(f"[{model_key}/{quant}] {trial_num}/{len(trials)} prompts; {time.time()-started:.1f}s")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen2.5-7b", "mistral-7b", "phi3.5-mini", "all"], default="all")
    parser.add_argument("--quant", choices=["fp16", "int4", "both"], default="both")
    parser.add_argument("--trials-per-category", type=int, default=5)
    parser.add_argument("--out-dir", default="followup_results/representation")
    args = parser.parse_args()
    if args.trials_per_category < 1:
        parser.error("--trials-per-category must be positive")
    require_cuda()
    out_dir = Path(args.out_dir)
    environment_report(out_dir)
    models = ["qwen2.5-7b", "mistral-7b", "phi3.5-mini"] if args.model == "all" else [args.model]
    quants = ["fp16", "int4"] if args.quant == "both" else [args.quant]
    for model_key in models:
        # Tokenizer filtering is condition-independent; use baseline tokenizer to build the fixed manifest.
        tokenizer, model = load_quantized_model(model_key, "fp16")
        pools, vocab_manifest = filtered_vocab(tokenizer)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        manifest_path = out_dir / model_key / "trial_manifest.json"
        trials = load_or_create_manifest(manifest_path, model_key, pools, args.trials_per_category)
        (out_dir / model_key / "vocab_manifest.json").write_text(json.dumps(vocab_manifest, indent=2) + "\n")
        for quant in quants:
            run_arm(model_key, quant, trials, out_dir / model_key / f"geometry_{quant}.csv")
        del tokenizer
    print("Finished representation runs. Analyze within-condition pairwise cosine by layer;")
    print("compare matched pair values between FP16 and INT4. This is exploratory geometry evidence.")


if __name__ == "__main__":
    main()
