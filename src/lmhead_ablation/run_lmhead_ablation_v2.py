"""
PI-LLM quantization ablation: int4_forced_lmhead arm (V2, standalone).

Self-contained: does not import from any other project file. Everything
needed -- task constants, trial generation, tokenizer filtering, scoring,
manual lm_head quantization, and model loading -- lives in this one file.

Context: the standard BitsAndBytesConfig leaves `lm_head` at full
precision by default (confirmed empirically for this model set -- the
original `int4` arm never actually quantized `lm_head`). This script
manually converts `lm_head` into a genuine bnb.nn.Linear4bit layer after
loading, and verifies the conversion via quant_state inspection (not
just class name) before running any trials.

Usage:
    python run_lmhead_ablation_v2.py --model qwen2.5-7b --scope minimal
    python run_lmhead_ablation_v2.py --model all --scope full
"""

import argparse
import csv
import inspect
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass


# ----------------------------------------------------------------------
# Environment report
# ----------------------------------------------------------------------

def print_and_save_env_report(out_dir):
    import torch
    import transformers
    try:
        import bitsandbytes as bnb
        bnb_version = bnb.__version__
    except Exception as e:
        bnb_version = f"IMPORT_FAILED: {e}"

    gpu_name = "NO_GPU_DETECTED"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    report = {
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "bitsandbytes_version": bnb_version,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "expected_torch": "2.5.1+cu121",
        "expected_transformers": "5.12.1",
        "expected_bitsandbytes": "0.49.2",
        "expected_gpu": "NVIDIA RTX 3090 (24GB)",
    }
    print("=" * 70)
    print("ENVIRONMENT REPORT (paste into paper's deviations-from-protocol note)")
    print("=" * 70)
    for k, v in report.items():
        print(f"  {k}: {v}")
    if report["torch_version"] != report["expected_torch"]:
        print("  >>> DEVIATION: torch version differs from spec")
    if report["transformers_version"] != report["expected_transformers"]:
        print("  >>> DEVIATION: transformers version differs from spec")
    if not isinstance(bnb_version, str) or bnb_version != report["expected_bitsandbytes"]:
        print("  >>> DEVIATION: bitsandbytes version differs from spec")
    if gpu_name != report["expected_gpu"] and "3090" not in gpu_name:
        print("  >>> DEVIATION: GPU differs from spec (report this explicitly)")
    print("=" * 70)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "environment_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


# ----------------------------------------------------------------------
# Task constants
# ----------------------------------------------------------------------

SUBJECTS = [
    "Adam", "Maria", "Sam", "Elena", "Marcus", "Priya", "Noah", "Ines",
    "Diego", "Yuki", "Omar", "Hannah", "Leo", "Fatima", "Chen", "Nora",
    "Victor", "Amara", "Felix", "Sana",
]

WORD_TEMPLATES = {
    "mood": "{subj}'s mood is now {val}.",
    "favorite_color": "{subj}'s favorite color is now {val}.",
    "favorite_animal": "{subj}'s favorite animal is now the {val}.",
    "occupation": "{subj}'s job is now {val}.",
}

NUMERIC_TEMPLATES = {
    "temperature": "The temperature reading for {subj}'s greenhouse is now {val} degrees.",
    "stock_price": "{subj}'s tracked stock price is now ${val}.",
    "page_count": "The page count for {subj}'s document is now {val} pages.",
}

NUMERIC_POOLS = {
    "temperature": list(range(10, 999)),
    "stock_price": list(range(5, 999)),
    "page_count": list(range(3, 999)),
}

RETRIEVAL_Q_WORD = "Based only on the most recent update above, what is {subj}'s {attr_phrase} right now? Reply with a single word and nothing else."
RETRIEVAL_Q_NUMERIC = "Based only on the most recent update above, what is the current {attr_phrase} for {subj}? Reply with a single number and nothing else."

ATTR_PHRASE = {
    "mood": "mood",
    "favorite_color": "favorite color",
    "favorite_animal": "favorite animal",
    "occupation": "job",
    "temperature": "temperature reading",
    "stock_price": "tracked stock price",
    "page_count": "page count",
}

MOOD_CANDIDATES = [
    "happy", "sad", "anxious", "excited", "calm", "angry", "nervous", "joyful",
    "tired", "confused", "proud", "frustrated", "relieved", "curious", "bored",
    "hopeful", "irritated", "content", "worried", "cheerful", "gloomy",
    "restless", "grateful", "embarrassed", "eager", "lonely", "surprised",
    "peaceful", "jealous", "motivated", "annoyed", "sleepy", "optimistic",
    "disappointed", "amused", "nostalgic", "grumpy", "playful", "shy",
    "confident", "insecure", "energetic", "exhausted", "hesitant", "brave",
    "timid", "cautious", "carefree", "stressed", "relaxed", "impatient",
    "patient", "determined", "uncertain", "satisfied", "regretful", "hurt",
    "thrilled", "terrified", "delighted", "puzzled", "hostile", "friendly",
    "distracted", "focused", "overwhelmed", "inspired", "numb", "vulnerable",
    "guilty", "hopeless", "eager", "wistful", "serene", "tense", "giddy",
    "melancholy", "spiteful", "affectionate", "indifferent", "suspicious",
    "trusting", "resentful", "empowered", "helpless", "amazed", "startled",
]
COLOR_CANDIDATES = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "teal",
    "maroon", "navy", "gold", "silver", "turquoise", "crimson", "indigo",
    "beige", "lavender", "olive", "coral", "charcoal", "amber", "ivory",
    "magenta", "mint", "rust", "azure", "emerald", "ruby", "cyan", "khaki",
    "taupe", "ochre", "sienna", "fuchsia", "burgundy", "scarlet", "cobalt",
    "sapphire", "jade", "lilac", "peach", "salmon", "mustard", "plum",
    "rose", "bronze", "copper", "pewter", "slate", "cream", "tan", "brown",
    "black", "white", "gray", "grey", "violet", "cerise", "denim", "flax",
    "honey", "walnut", "chestnut", "clay", "sand", "moss", "forest",
    "sky", "wine", "berry", "cherry", "lime", "lemon", "apricot", "grape",
]
ANIMAL_CANDIDATES = [
    "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "squirrel",
    "otter", "beaver", "raccoon", "badger", "hedgehog", "mole", "mouse",
    "rat", "hamster", "ferret", "weasel", "skunk", "sloth", "koala",
    "kangaroo", "panda", "elephant", "giraffe", "zebra", "hippo", "rhino",
    "camel", "llama", "horse", "donkey", "cow", "bull", "pig", "sheep",
    "goat", "chicken", "duck", "goose", "turkey", "swan", "peacock",
    "ostrich", "penguin", "eagle", "hawk", "falcon", "owl", "crow", "raven",
    "sparrow", "robin", "cardinal", "pigeon", "parrot", "toucan", "flamingo",
    "pelican", "heron", "stork", "dolphin", "whale", "shark", "octopus",
    "squid", "crab", "lobster", "shrimp", "jellyfish", "starfish", "turtle",
    "tortoise", "lizard", "gecko", "iguana", "chameleon", "snake", "cobra",
    "python", "crocodile", "alligator", "frog", "toad", "bee", "wasp",
    "ant", "beetle", "butterfly", "moth", "dragonfly", "cricket", "spider",
    "scorpion", "snail", "worm", "goldfish", "salmon", "trout", "eel",
]
OCCUPATION_CANDIDATES = [
    "doctor", "teacher", "lawyer", "engineer", "nurse", "dentist", "pilot",
    "chef", "baker", "farmer", "plumber", "electrician", "mechanic",
    "carpenter", "painter", "architect", "accountant", "banker",
    "journalist", "photographer", "musician", "actor", "dancer", "writer",
    "poet", "scientist", "professor", "librarian", "translator", "therapist",
    "psychologist", "surgeon", "paramedic", "firefighter", "soldier",
    "sailor", "astronaut", "athlete", "coach", "referee", "tailor",
    "florist", "butcher", "waiter", "bartender", "barista", "consultant",
    "analyst", "developer", "designer", "editor", "publisher", "curator",
    "historian", "biologist", "chemist", "physicist", "geologist",
    "economist", "linguist", "zoologist", "botanist", "astronomer",
    "cardiologist", "neurologist", "radiologist", "optometrist",
    "chiropractor", "nutritionist", "plumber", "electrician", "welder",
    "surveyor", "cartoonist", "sculptor", "composer", "conductor",
]

WORD_CANDIDATE_LISTS = {
    "mood": MOOD_CANDIDATES,
    "favorite_color": COLOR_CANDIDATES,
    "favorite_animal": ANIMAL_CANDIDATES,
    "occupation": OCCUPATION_CANDIDATES,
}

INTERFERENCE_LEVELS = [1, 2, 4, 8, 16, 32, 64, 96]
TRIALS_PER_LEVEL = {1: 15, 2: 15, 4: 20, 8: 25, 16: 60, 32: 50, 64: 60, 96: 60}
SEEDS = [42, 123, 999, 7, 2024]

ALL_CATEGORIES = list(WORD_TEMPLATES.keys()) + list(NUMERIC_TEMPLATES.keys())

CHECK_SUBJ = "Sam"

MODEL_REGISTRY = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}

KEY_LEVELS_MINIMAL = {
    "qwen2.5-7b": [32, 64, 96],
    "mistral-7b": [8, 16, 32],
    "phi3.5-mini": [4, 8, 16],
}


# ----------------------------------------------------------------------
# Tokenizer-verified single-token filtering
# ----------------------------------------------------------------------

def is_single_token_in_context(tokenizer, template, subj, word):
    prefix_template, suffix_template = template.split("{val}")
    prefix = prefix_template.format(subj=subj)
    suffix = suffix_template.format(subj=subj)
    text = prefix + word + suffix
    start, end = len(prefix), len(prefix) + len(word)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    overlapping = [i for i, (s, e) in enumerate(offsets) if not (e <= start or s >= end)]
    return len(overlapping) == 1


def build_filtered_word_pools(tokenizer):
    pools = {}
    for cat, candidates in WORD_CANDIDATE_LISTS.items():
        template = WORD_TEMPLATES[cat]
        filtered = [
            w for w in candidates
            if is_single_token_in_context(tokenizer, template, CHECK_SUBJ, w)
        ]
        pools[cat] = filtered
        print(f"  [{cat}] {len(filtered)}/{len(candidates)} candidates survive "
              f"single-token filtering")
    return pools


def pool_for(category_key, word_pools):
    if category_key in WORD_TEMPLATES:
        return word_pools[category_key]
    else:
        return NUMERIC_POOLS[category_key]


def max_pool_size_for(category_key, word_pools):
    return len(pool_for(category_key, word_pools))


# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

def build_prompt(category_key, subj, values):
    is_word = category_key in WORD_TEMPLATES
    template = WORD_TEMPLATES[category_key] if is_word else NUMERIC_TEMPLATES[category_key]
    lines = [template.format(subj=subj, val=v) for v in values]
    attr_phrase = ATTR_PHRASE[category_key]
    if is_word:
        question = RETRIEVAL_Q_WORD.format(subj=subj, attr_phrase=attr_phrase)
    else:
        question = RETRIEVAL_Q_NUMERIC.format(subj=subj, attr_phrase=attr_phrase)
    prompt = "\n".join(lines) + "\n\n" + question
    return prompt, is_word


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zA-Z]+")
_NUM_RE = re.compile(r"[-+]?\d+")


def normalize_response(raw):
    s = raw.strip().lower()
    if s.endswith("."):
        s = s[:-1]
    return s.strip()


def score_word(raw, gold):
    s = normalize_response(raw)
    m = _WORD_RE.search(s)
    extracted = m.group(0) if m else ""
    correct = extracted == gold.lower()
    return extracted, correct


def score_numeric(raw, gold):
    s = normalize_response(raw)
    m = _NUM_RE.search(s)
    extracted = m.group(0) if m else ""
    try:
        correct = str(int(extracted)) == str(int(gold))
    except (ValueError, TypeError):
        correct = False
    return extracted, correct


# ----------------------------------------------------------------------
# Trial generation (exact nested-loop RNG replay order)
# ----------------------------------------------------------------------

@dataclass
class Trial:
    seed: int
    category_key: str
    kind: str
    level: int
    trial_idx: int
    subj: str
    values: list
    gold: object


def generate_trials(word_pools, key_levels=None):
    trials = []
    for seed in SEEDS:
        rng = random.Random(seed)
        for category_key in ALL_CATEGORIES:
            is_word = category_key in WORD_TEMPLATES
            pool = pool_for(category_key, word_pools)
            for level in INTERFERENCE_LEVELS:
                if level > max_pool_size_for(category_key, word_pools):
                    continue
                n_trials = TRIALS_PER_LEVEL[level]
                for trial_idx in range(n_trials):
                    subj = rng.choice(SUBJECTS)
                    values = rng.sample(pool, level)
                    gold = values[-1]
                    keep = True
                    if key_levels is not None:
                        keep = (is_word and level in key_levels)
                    if keep:
                        trials.append(Trial(
                            seed=seed, category_key=category_key,
                            kind="word" if is_word else "numeric",
                            level=level, trial_idx=trial_idx,
                            subj=subj, values=values, gold=gold,
                        ))
                    # rng calls above must happen even when `keep` is False,
                    # to stay trial-paired with the original dataset's RNG stream.
    return trials


# ----------------------------------------------------------------------
# Manual lm_head quantization + verification
# ----------------------------------------------------------------------

def manually_quantize_lm_head(model, compute_dtype, quant_type="nf4", double_quant=True):
    import torch
    import bitsandbytes as bnb

    old_lm_head = model.get_output_embeddings()
    if old_lm_head is None:
        raise RuntimeError("model.get_output_embeddings() returned None -- "
                            "cannot locate lm_head to quantize.")

    if "Linear4bit" in type(old_lm_head).__name__:
        print("  lm_head is already Linear4bit -- nothing to do.")
        return model

    in_features = old_lm_head.in_features
    out_features = old_lm_head.out_features
    has_bias = old_lm_head.bias is not None
    device = old_lm_head.weight.device

    print(f"  Converting lm_head ({in_features} -> {out_features}, bias={has_bias}) "
          f"to bnb.nn.Linear4bit (quant_type={quant_type}, "
          f"compute_dtype={compute_dtype}, double_quant={double_quant})")

    sig = inspect.signature(bnb.nn.Linear4bit.__init__)
    params = set(sig.parameters.keys())
    kwargs = dict(bias=has_bias, compute_dtype=compute_dtype, quant_type=quant_type)
    if "compress_statistics" in params:
        kwargs["compress_statistics"] = double_quant
    else:
        print("  [warn] bnb.nn.Linear4bit has no 'compress_statistics' param in this "
              "version -- double_quant setting will not be applied to lm_head "
              "(note this deviation in the report).")

    new_lm_head = bnb.nn.Linear4bit(in_features, out_features, **kwargs)

    new_lm_head.weight = bnb.nn.Params4bit(
        old_lm_head.weight.data.clone(),
        requires_grad=False,
        quant_type=quant_type,
    )
    if has_bias:
        new_lm_head.bias = torch.nn.Parameter(old_lm_head.bias.data.clone())

    new_lm_head = new_lm_head.to(device)  # triggers quantization

    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(new_lm_head)
    else:
        model.lm_head = new_lm_head

    return model


def verify_lm_head_quantized(model):
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head
    cls_name = type(lm_head).__name__
    weight_type = type(getattr(lm_head, "weight", None)).__name__

    print(f"lm_head class: {cls_name}, weight param class: {weight_type}")

    is_quantized = "Linear4bit" in cls_name and "Params4bit" in weight_type
    if not is_quantized:
        raise RuntimeError(
            f"VERIFICATION FAILED: lm_head is NOT a quantized bnb layer "
            f"(class={cls_name}, weight={weight_type}). The manual "
            f"conversion did not take. Do not proceed -- results would "
            f"silently duplicate the unquantized-lm_head arms."
        )

    quant_state = getattr(lm_head.weight, "quant_state", None)
    if quant_state is None:
        raise RuntimeError(
            "VERIFICATION FAILED: lm_head.weight has no quant_state -- "
            "quantization did not actually execute (weight was likely "
            "never moved to the target device to trigger Params4bit "
            "quantization). Do not proceed."
        )

    print("  -> verified: lm_head is a real quantized bnb.nn.Linear4bit layer "
          f"(quant_state present, quant_type={getattr(quant_state, 'quant_type', 'unknown')}).")
    return {"lm_head_class": cls_name, "lm_head_weight_class": weight_type,
            "quant_type": str(getattr(quant_state, "quant_type", "unknown"))}


# ----------------------------------------------------------------------
# Model loading + inference
# ----------------------------------------------------------------------

def load_model_and_tokenizer_forced(hf_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        quantization_config=base_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()

    current_lm_head = model.get_output_embeddings()
    print(f"Pre-conversion lm_head class: {type(current_lm_head).__name__}, "
          f"dtype: {getattr(current_lm_head, 'weight', None).dtype if hasattr(current_lm_head, 'weight') else 'n/a'}")

    model = manually_quantize_lm_head(
        model, compute_dtype=torch.float16, quant_type="nf4", double_quant=True
    )
    verification = verify_lm_head_quantized(model)

    return model, tokenizer, verification


def run_inference(model, tokenizer, prompt):
    import torch
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=12,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    gen_tokens = out[0][inputs["input_ids"].shape[1]:]
    raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return raw_response, latency


# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                         choices=list(MODEL_REGISTRY.keys()) + ["all"])
    parser.add_argument("--scope", default="minimal", choices=["minimal", "full"])
    parser.add_argument("--out-dir", default="./results_int4_forced_lmhead")
    args = parser.parse_args()

    env_report = print_and_save_env_report(args.out_dir)

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        hf_id = MODEL_REGISTRY[model_key]
        print(f"\n{'#'*70}\n# Loading {model_key} ({hf_id})\n{'#'*70}")

        model, tokenizer, verification_info = load_model_and_tokenizer_forced(hf_id)

        print("Building single-token-filtered word pools for this tokenizer...")
        word_pools = build_filtered_word_pools(tokenizer)

        key_levels = None
        if args.scope == "minimal":
            key_levels = set(KEY_LEVELS_MINIMAL[model_key])
            print(f"Reduced scope: word-type only, levels {sorted(key_levels)}, all seeds.")
        else:
            print("Full scope: all categories, all levels, all seeds.")

        trials = generate_trials(word_pools, key_levels=key_levels)
        print(f"Generated {len(trials)} trials for {model_key}.")

        by_seed = {}
        for t in trials:
            by_seed.setdefault(t.seed, []).append(t)

        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, f"meta_{model_key}_int4forcedlmhead.json"), "w") as f:
            json.dump({
                "model_key": model_key,
                "hf_id": hf_id,
                "scope": args.scope,
                "verification": verification_info,
                "n_trials_total": len(trials),
                "env": env_report,
                "note": "lm_head manually converted to bnb.nn.Linear4bit post-load; "
                        "the standard BitsAndBytesConfig leaves lm_head unquantized "
                        "by default, confirmed empirically for this model set.",
            }, f, indent=2)

        for seed, seed_trials in by_seed.items():
            out_path = os.path.join(
                args.out_dir, f"results_v2_{model_key}_int4forcedlmhead_seed{seed}.csv"
            )
            print(f"  Running {len(seed_trials)} trials for seed={seed} -> {out_path}")
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "model", "quant_mode", "seed", "attribute", "kind",
                    "interference_level", "trial_idx", "subject", "gold",
                    "model_raw_response", "model_extracted", "correct",
                    "latency_sec",
                ])
                for i, t in enumerate(seed_trials):
                    prompt, is_word = build_prompt(t.category_key, t.subj, t.values)
                    raw_response, latency = run_inference(model, tokenizer, prompt)

                    if is_word:
                        extracted, correct = score_word(raw_response, t.gold)
                    else:
                        extracted, correct = score_numeric(raw_response, t.gold)

                    writer.writerow([
                        model_key, "int4_forced_lmhead", t.seed, t.category_key, t.kind,
                        t.level, t.trial_idx, t.subj, t.gold,
                        raw_response.replace("\n", "\\n"), extracted, correct,
                        f"{latency:.4f}",
                    ])

                    if (i + 1) % 25 == 0:
                        print(f"    ...{i+1}/{len(seed_trials)}")
                        f.flush()

        del model
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    print("\nDone. Files written to:", args.out_dir)


if __name__ == "__main__":
    main()