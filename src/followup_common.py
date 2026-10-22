"""Shared reproducibility helpers for the Molab follow-up experiments."""

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

try:
    import pi_llm_eval_v2 as protocol
except ModuleNotFoundError as exc:
    if exc.name not in {"transformers", "bitsandbytes"}:
        raise
    protocol = None


def require_protocol():
    if protocol is None:
        raise RuntimeError(
            "The experiment runtime needs Transformers and bitsandbytes installed "
            "before importing the primary evaluator."
        )
    return protocol

SEEDS = [42, 123, 999, 7, 2024]
KEY_LEVELS = {"qwen2.5-7b": 64, "mistral-7b": 16, "phi3.5-mini": 8}


def environment_report(out_dir):
    import transformers
    try:
        import bitsandbytes
        bnb_version = bitsandbytes.__version__
    except Exception as exc:
        bnb_version = f"unavailable: {exc}"
    report = {
        "python": sys.version,
        "platform": sys.platform,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bnb_version,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_runtime": torch.version.cuda,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / "environment_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Run this experiment in the Molab GPU runtime.")


def filtered_vocab(tokenizer):
    protocol = require_protocol()
    pools, manifest = protocol.build_filtered_vocab(tokenizer)
    return pools, manifest


def manifest_trials(model_key, word_pools, trials_per_category, seeds=SEEDS, protocol_override=None):
    """Replay the primary evaluator's full RNG schedule, then select key-level trials."""
    protocol = protocol_override or require_protocol()
    if model_key not in KEY_LEVELS:
        raise ValueError(f"No key interference level configured for {model_key!r}")
    if trials_per_category < 1:
        raise ValueError("trials_per_category must be positive")

    # pi_llm_eval_v2.main iterates insertion order: all word categories, then
    # all numeric categories. Preserve that order because it determines RNG draws.
    categories = list(protocol.WORD_CATEGORIES) + list(protocol.NUMERIC_CATEGORIES)
    selected_level = KEY_LEVELS[model_key]
    trials = []
    for seed in seeds:
        rng = random.Random(seed)
        for category in categories:
            is_word = category in protocol.WORD_CATEGORIES
            pool_size = (len(word_pools[category]) if is_word
                         else len(protocol.NUMERIC_CATEGORIES[category]["pool"]))
            for level in protocol.INTERFERENCE_LEVELS:
                # Match the primary evaluator's pool-cap skip exactly: skipped
                # levels consume no random draws; every eligible trial does.
                if level > pool_size:
                    continue
                n_trials = protocol.TRIALS_PER_LEVEL.get(level, 20)
                for trial_idx in range(n_trials):
                    trial = protocol.build_trial(category, level, rng, word_pools)
                    if trial is None:
                        continue
                    if (not is_word or level != selected_level
                            or trial_idx >= trials_per_category):
                        continue
                    identity = {
                        "model": model_key,
                        "seed": seed,
                        "attribute": category,
                        "interference_level": level,
                        "trial_idx": trial_idx,
                        "subject": trial["subject"],
                        "gold": trial["gold"],
                        "prompt": trial["prompt"],
                        "prompt_sha256": hashlib.sha256(trial["prompt"].encode()).hexdigest(),
                        "kind": trial["kind"],
                    }
                    trials.append(identity)
    return trials


def load_or_create_manifest(path, model_key, word_pools, trials_per_category, protocol_override=None):
    protocol = protocol_override or require_protocol()
    path = Path(path)
    if path.exists():
        payload = json.loads(path.read_text())
        if (payload.get("model") != model_key
                or payload.get("trials_per_category") != trials_per_category
                or payload.get("word_pools") != word_pools
                or payload.get("seeds") != SEEDS
                or payload.get("protocol_schedule") != "pi_llm_eval_v2 category/level/trial insertion order, full eligible schedule replay"):
            raise ValueError(f"Existing manifest settings do not match requested run or tokenizer pools: {path}")
        trials = payload["trials"]
        for row in trials:
            expected = hashlib.sha256(row["prompt"].encode()).hexdigest()
            if row.get("prompt_sha256") != expected:
                raise ValueError(f"Manifest prompt hash mismatch for {row}")
        return trials
    trials = manifest_trials(model_key, word_pools, trials_per_category, protocol_override=protocol)
    for row in trials:
        row["prompt_sha256"] = hashlib.sha256(row["prompt"].encode()).hexdigest()
    payload = {
        "schema_version": 1,
        "model": model_key,
        "hf_model_id": protocol.MODEL_REGISTRY[model_key]["hf_id"],
        "key_interference_level": KEY_LEVELS[model_key],
        "seeds": SEEDS,
        "protocol_schedule": "pi_llm_eval_v2 category/level/trial insertion order, full eligible schedule replay",
        "trials_per_category": trials_per_category,
        "word_pools": word_pools,
        "trials": trials,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp_path, path)
    return trials


def load_quantized_model(model_key, quant):
    protocol = require_protocol()
    model_cfg = protocol.MODEL_REGISTRY[model_key]
    return protocol.load_model(model_cfg["hf_id"], quant, model_cfg["trust_remote_code"])


def rendered_prompt(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_inputs(tokenizer, prompt, device):
    rendered = rendered_prompt(tokenizer, prompt)
    encoded = tokenizer(rendered, return_tensors="pt", return_offsets_mapping=True,
                       add_special_tokens=False)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return rendered, input_ids, attention_mask, offsets


def token_span(rendered, prompt, value, offsets):
    prompt_start = rendered.find(prompt)
    if prompt_start < 0:
        raise ValueError("Could not locate task prompt in rendered chat template")
    local_start = prompt.rfind(value)
    if local_start < 0:
        raise ValueError(f"Could not locate final value {value!r} in prompt")
    start = prompt_start + local_start
    end = start + len(value)
    positions = [i for i, (left, right) in enumerate(offsets)
                if right > start and left < end]
    if not positions:
        raise ValueError(f"No tokenizer offsets overlap value span {value!r}")
    return positions


def stable_generation_seed(row, repeat_idx):
    key = f"{row['model']}|{row['seed']}|{row['attribute']}|{row['interference_level']}|{row['trial_idx']}|{repeat_idx}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31 - 1)


def update_values(row, protocol_override=None):
    """Recover each assignment value from the canonical category template."""
    protocol = protocol_override or require_protocol()
    category = protocol.WORD_CATEGORIES[row["attribute"]]
    template = category["template"]
    prefix_template, suffix_template = template.split("{val}")
    prefix = prefix_template.format(subj=row["subject"])
    suffix = suffix_template.format(subj=row["subject"])
    update_text = row["prompt"].split("\n\n", 1)[0]
    values = []
    for line in update_text.splitlines():
        if not line.startswith(prefix) or not line.endswith(suffix):
            raise ValueError(f"Unexpected update line for {row['attribute']}: {line!r}")
        end = len(line) - len(suffix) if suffix else len(line)
        values.append(line[len(prefix):end])
    if not values or values[-1] != row["gold"]:
        raise ValueError("Manifest update values do not end with the recorded gold value")
    return values
