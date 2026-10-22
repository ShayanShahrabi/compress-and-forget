"""Matched stochastic-decoding sensitivity experiment for FP16/INT8/INT4."""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

try:
    from .followup_common import (
        environment_report, filtered_vocab, load_or_create_manifest, load_quantized_model,
        prompt_inputs, require_cuda, stable_generation_seed, require_protocol,
    )
except ImportError:
    from followup_common import (
    environment_report, filtered_vocab, load_or_create_manifest, load_quantized_model,
    prompt_inputs, require_cuda, stable_generation_seed, require_protocol,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen2.5-7b", "mistral-7b", "phi3.5-mini", "all"], default="all")
    parser.add_argument("--quant", choices=["fp16", "int8", "int4", "all"], default="all")
    parser.add_argument("--trials-per-category", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out-dir", default="followup_results/sampling")
    args = parser.parse_args()
    if args.trials_per_category < 1 or args.repeats < 1:
        parser.error("trial and repeat counts must be positive")
    require_cuda()
    protocol = require_protocol()
    out_dir = Path(args.out_dir)
    environment_report(out_dir)
    models = ["qwen2.5-7b", "mistral-7b", "phi3.5-mini"] if args.model == "all" else [args.model]
    quants = ["fp16", "int8", "int4"] if args.quant == "all" else [args.quant]
    for model_key in models:
        tokenizer, model = load_quantized_model(model_key, "fp16")
        pools, vocab_manifest = filtered_vocab(tokenizer)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_dir = out_dir / model_key
        manifest_path = model_dir / "trial_manifest.json"
        trials = load_or_create_manifest(manifest_path, model_key, pools, args.trials_per_category)
        (model_dir / "vocab_manifest.json").write_text(json.dumps(vocab_manifest, indent=2) + "\n")
        del tokenizer
        for quant in quants:
            tokenizer, model = load_quantized_model(model_key, quant)
            path = model_dir / f"sampling_{quant}.csv"
            fields = ["model", "quant_mode", "seed", "attribute", "kind", "interference_level",
                      "trial_idx", "subject", "gold", "prompt_sha256", "repeat_idx",
                      "generation_seed", "temperature", "top_p", "do_sample", "raw_response",
                      "model_extracted", "correct", "latency_sec"]
            done = set()
            if path.exists():
                with path.open(newline="", encoding="utf-8") as handle:
                    done = {(r["seed"], r["attribute"], r["trial_idx"], r["repeat_idx"])
                            for r in csv.DictReader(handle)}
            write_header = not path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                if write_header:
                    writer.writeheader()
                for trial_num, row in enumerate(trials, 1):
                    _, input_ids, attention_mask, _ = prompt_inputs(tokenizer, row["prompt"], model.device)
                    if (input_ids.shape[1] + 12) > model.config.max_position_embeddings:
                        raise ValueError(f"Prompt exceeds model context length for trial {row['trial_idx']}")
                    for repeat_idx in range(args.repeats):
                        key = (str(row["seed"]), row["attribute"], str(row["trial_idx"]), str(repeat_idx))
                        if key in done:
                            continue
                        generation_seed = stable_generation_seed(row, repeat_idx)
                        torch.manual_seed(generation_seed)
                        torch.cuda.manual_seed_all(generation_seed)
                        started = time.time()
                        with torch.inference_mode():
                            output = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=12,
                                do_sample=True,
                                temperature=0.7,
                                top_p=0.9,
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
                        latency = time.time() - started
                        correct, extracted = protocol.score_response(response, row["gold"], "word")
                        writer.writerow({
                            "model": model_key, "quant_mode": quant, "seed": row["seed"],
                            "attribute": row["attribute"], "kind": "word",
                            "interference_level": row["interference_level"], "trial_idx": row["trial_idx"],
                            "subject": row["subject"], "gold": row["gold"],
                            "prompt_sha256": row["prompt_sha256"], "repeat_idx": repeat_idx,
                            "generation_seed": generation_seed, "temperature": 0.7, "top_p": 0.9,
                            "do_sample": True, "raw_response": response.replace("\n", " ")[:500],
                            "model_extracted": extracted, "correct": int(correct),
                            "latency_sec": round(latency, 3),
                        })
                        handle.flush()
                    if trial_num % 25 == 0:
                        print(f"[{model_key}/{quant}] {trial_num}/{len(trials)} prompts")
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("Completed matched sampling runs. Treat repeat completions as clustered within prompt;")
    print("do not analyze the five completions as independent prompts.")


if __name__ == "__main__":
    main()
