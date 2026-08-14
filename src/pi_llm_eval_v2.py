"""
PI-LLM-style Proactive Interference Evaluation Under Quantization (v2)
========================================================================

Research question:
    Does quantization (FP16 -> INT8 -> INT4) selectively amplify proactive
    interference for SEMANTICALLY similar distractors (word-type attributes)
    more than for arbitrary distractors (numeric attributes)?

Scope: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, and Phi-3.5-mini-instruct.

Design notes:
    - Word-type vocabulary pools (mood, color, animal, occupation) are built
      from large candidate lists, then FILTERED using the actual model
      tokenizer to keep only single-token entries, via baseline subtraction:
      len(encode(" "+word)) - len(encode(" ")) == 1. This is robust across
      tokenizer families -- the naive "len(encode(" "+word)) == 1" check
      previously zeroed out ALL of Phi's word pools, because Phi's
      SentencePiece-based tokenizer adds a constant extra leading token to
      every encoding regardless of add_special_tokens, making every word
      look like 2 tokens. Baseline subtraction cancels that constant out.
    - Interference levels: 1, 2, 4, 8, 16, 32, 64, 96 (word pools cap out
      wherever the filtered pool size allows; numeric pools support all of
      these).
    - Trial budget is weighted toward level 64 (Qwen's informative range --
      significant fp16-vs-int4 gap found there) and level 16 (Mistral's
      informative mid-curve range, away from ceiling and from the floor
      effect seen at level 32).
    - A vocab_manifest_<model>_<quant>.json is written per run recording the
      exact filtered pool per category + token counts, for reproducibility.

Usage (run once per model x quant x seed combination -- separate processes so
GPU memory is fully released between runs; safe to run sequentially on one GPU):
    python pi_llm_eval_v2.py --model phi3.5-mini --quant fp16 --seed 42
    python pi_llm_eval_v2.py --model phi3.5-mini --quant fp16 --seed 123
    python pi_llm_eval_v2.py --model phi3.5-mini --quant fp16 --seed 999
    ... (3 models x 3 quant levels x 3 seeds = 27 runs total)

Or use run_all.sh to run all 27 combinations sequentially with logging and
continue-on-error (so one crash doesn't stop the rest).

Resumable: re-running the same --model/--quant/--seed triple skips already-
completed rows -- safe to re-run run_all.sh if a run gets interrupted.

NOTE on existing Phi data: your earlier Phi runs (before this fix) only have
numeric-category rows -- all word-category trials were silently skipped
because the vocab pools were incorrectly computed as empty. Re-running with
this fixed script will NOT duplicate the numeric rows (already marked done)
and will fill in the missing word-category rows for the same files.
"""

import argparse
import csv
import json
import os
import random
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_REGISTRY = {
    "qwen2.5-7b":  {"hf_id": "Qwen/Qwen2.5-7B-Instruct",           "trust_remote_code": False},
    "mistral-7b":  {"hf_id": "mistralai/Mistral-7B-Instruct-v0.3", "trust_remote_code": False},
    # trust_remote_code=False for Phi is intentional: transformers has native,
    # maintained Phi3 support since v4.43+. trust_remote_code=True instead pulls
    # Microsoft's own custom modeling file from the HF cache, which can go stale
    # relative to your installed transformers' internal cache API -- that
    # mismatch (a stale '.seen_tokens' attribute access) is what caused the
    # 'DynamicCache has no attribute seen_tokens' crash previously.
    "phi3.5-mini": {"hf_id": "microsoft/Phi-3.5-mini-instruct",    "trust_remote_code": False},
    # OLMo 2 is natively supported in transformers (no trust_remote_code
    # needed) as of the version required here; fully open weights AND
    # training data (AI2), a distinct training pedigree from the other
    # three models -- useful for the "generalizes across training
    # approaches" argument, not just architecture.
    # Gemma 2 was attempted but ran into a persistent download hang on the
    # cluster (Xet storage backend stalling regardless of auth/network
    # health -- confirmed via curl that connectivity and auth were both
    # fine). Swapped for Falcon3, fully open, no gate, different training
    # org (TII), natively supported in transformers.
    "falcon3-7b":  {"hf_id": "tiiuae/Falcon3-7B-Instruct",          "trust_remote_code": False},
}

# Interference levels. Word pools cap out wherever the filtered pool size
# allows (checked at runtime); numeric pools are large enough for all of these.
INTERFERENCE_LEVELS = [1, 2, 4, 8, 16, 32, 64, 96]

# Trial budget per interference level -- weighted toward high levels, since
# that's where the real signal is: level 64 for Qwen (significant fp16 vs
# int4 gap found in the pilot), level 16 for Mistral (informative middle of
# its interference curve, away from ceiling AND away from the floor effect
# seen at level 32).
TRIALS_PER_LEVEL = {
    1: 15, 2: 15, 4: 20, 8: 25, 16: 60, 32: 50, 64: 60, 96: 60,
}

SEED = 42

SUBJECTS = [
    "Adam", "Maria", "Sam", "Elena", "Marcus", "Priya", "Noah", "Ines",
    "Diego", "Yuki", "Omar", "Hannah", "Leo", "Fatima", "Chen", "Nora",
    "Victor", "Amara", "Felix", "Sana",
]

# --------------------------------------------------------------------------
# Candidate word lists (word-type / semantic-interference categories).
# These are large on purpose -- they get filtered down to single-token
# entries by the tokenizer at runtime, so the FINAL pool size depends on
# Qwen's vocabulary, not on this list length. Kept to common, everyday
# words throughout (no manual "reach for obscure words" padding).
# --------------------------------------------------------------------------

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

WORD_CATEGORIES = {
    "mood": {
        "template": "{subj}'s mood is now {val}.",
        "question": "Based only on the most recent update above, what is {subj}'s mood right now? Reply with a single word and nothing else.",
        "candidates": MOOD_CANDIDATES,
    },
    "favorite_color": {
        "template": "{subj}'s favorite color is now {val}.",
        "question": "Based only on the most recent update above, what is {subj}'s favorite color right now? Reply with a single word and nothing else.",
        "candidates": COLOR_CANDIDATES,
    },
    "favorite_animal": {
        "template": "{subj}'s favorite animal is now the {val}.",
        "question": "Based only on the most recent update above, what is {subj}'s favorite animal right now? Reply with a single word and nothing else.",
        "candidates": ANIMAL_CANDIDATES,
    },
    "occupation": {
        "template": "{subj}'s job is now {val}.",
        "question": "Based only on the most recent update above, what is {subj}'s job right now? Reply with a single word and nothing else.",
        "candidates": OCCUPATION_CANDIDATES,
    },
}

NUMERIC_CATEGORIES = {
    "temperature": {
        "template": "The temperature reading for {subj}'s greenhouse is now {val} degrees.",
        "question": "Based only on the most recent update above, what is the temperature reading right now? Reply with only the number and nothing else.",
        "pool": [str(n) for n in range(10, 999)],
    },
    "stock_price": {
        "template": "{subj}'s tracked stock price is now ${val}.",
        "question": "Based only on the most recent update above, what is the tracked stock price right now? Reply with only the number (no $ sign) and nothing else.",
        "pool": [str(n) for n in range(5, 999)],
    },
    "page_count": {
        "template": "The page count for {subj}'s document is now {val} pages.",
        "question": "Based only on the most recent update above, what is the document's page count right now? Reply with only the number and nothing else.",
        "pool": [str(n) for n in range(3, 999)],
    },
}


def is_single_token_in_context(tokenizer, template, subj, word):
    """Determine whether `word`, as it actually appears inside the filled-in
    template sentence, occupies exactly one token -- by tokenizing the real
    sentence and checking how many tokens' character spans overlap the
    word's own character span (via the fast tokenizer's offset mapping).

    This replaces an earlier, flawed approach that tried to infer token
    count via arithmetic on separately-encoded fragments (e.g.
    tokens(" "+word) - tokens(" ")). That approach silently assumed spaces
    compose the same way across tokenizer families, which is false: for
    BPE tokenizers (e.g. Qwen) a leading space fuses INTO the word's token,
    while for SentencePiece tokenizers (e.g. Mistral, Phi) a leading space
    can add a constant extra token. The arithmetic approach worked for one
    family and silently broke for the other. Reading offsets directly off
    the real tokenized sentence has no such assumption baked in.
    """
    prefix_template, suffix_template = template.split("{val}")
    prefix = prefix_template.format(subj=subj)
    suffix = suffix_template.format(subj=subj)
    text = prefix + word + suffix
    start, end = len(prefix), len(prefix) + len(word)

    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    overlapping = [i for i, (s, e) in enumerate(offsets) if not (e <= start or s >= end)]
    return len(overlapping) == 1


def build_filtered_vocab(tokenizer):
    """Filter each word-category's candidate list down to single-token
    entries using the real tokenizer, checked IN CONTEXT (inside that
    category's actual template sentence). Returns dict + writes a manifest.
    A fixed representative subject name is used for the check; token count
    for the value word does not meaningfully depend on which short common
    name precedes it."""
    filtered = {}
    manifest = {}
    check_subj = "Sam"
    for name, spec in WORD_CATEGORIES.items():
        template = spec["template"]
        kept = [w for w in spec["candidates"]
                if is_single_token_in_context(tokenizer, template, check_subj, w)]
        dropped = [w for w in spec["candidates"] if w not in kept]
        filtered[name] = kept
        manifest[name] = {
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "kept": kept,
            "dropped": dropped,
        }
        print(f"[vocab] {name}: {len(kept)}/{len(spec['candidates'])} candidates are single-token "
              f"(max testable interference level = {len(kept)})")
    return filtered, manifest


def build_trial(category_key, interference_level, rng, filtered_word_pools):
    if category_key in WORD_CATEGORIES:
        spec = WORD_CATEGORIES[category_key]
        pool = filtered_word_pools[category_key]
        kind = "word"
    else:
        spec = NUMERIC_CATEGORIES[category_key]
        pool = spec["pool"]
        kind = "numeric"

    n = interference_level
    if n > len(pool):
        return None
    subj = rng.choice(SUBJECTS)
    values = rng.sample(pool, n)
    lines = [spec["template"].format(subj=subj, val=v) for v in values]
    question = spec["question"].format(subj=subj)
    prompt = "\n".join(lines) + "\n\n" + question
    gold = values[-1]
    return {"subject": subj, "prompt": prompt, "gold": gold, "kind": kind}


def load_model(model_id, quant_mode, trust_remote_code=False):
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    common_kwargs = dict(device_map={"": 0}, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code)
    if quant_mode == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, **common_kwargs
        )
    elif quant_mode == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, **common_kwargs
        )
    elif quant_mode == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, **common_kwargs
        )
    else:
        raise ValueError(f"Unknown quant_mode: {quant_mode}")
    model.eval()
    print(f"GPU memory allocated after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return tokenizer, model


def generate_answer(tokenizer, model, prompt, max_new_tokens=12):
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def score_response(response, gold, kind):
    resp_clean = response.strip().lower().strip(".").strip()
    if kind == "numeric":
        match = re.search(r"-?\d+", resp_clean)
        model_val = match.group(0) if match else None
        correct = (model_val == gold)
    else:
        match = re.search(r"[a-zA-Z]+", resp_clean)
        model_val = match.group(0) if match else None
        correct = (model_val == gold.lower())
    return correct, model_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), required=True)
    parser.add_argument("--quant", choices=["fp16", "int8", "int4"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=None)
    args, _ = parser.parse_known_args()

    model_cfg = MODEL_REGISTRY[args.model]
    model_id = model_cfg["hf_id"]
    trust_remote_code = model_cfg["trust_remote_code"]

    out_path = args.out or f"results_v2_{args.model}_{args.quant}_seed{args.seed}.csv"
    manifest_path = f"vocab_manifest_{args.model}_{args.quant}.json"
    rng = random.Random(args.seed)

    done = set()
    write_header = not os.path.exists(out_path)
    if not write_header:
        with open(out_path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["attribute"], int(row["interference_level"]), int(row["trial_idx"])))

    print(f"Loading {model_id} ({args.model}) in {args.quant} mode...")
    tokenizer, model = load_model(model_id, args.quant, trust_remote_code=trust_remote_code)
    print("Model loaded.")

    filtered_word_pools, vocab_manifest = build_filtered_vocab(tokenizer)
    with open(manifest_path, "w") as fh:
        json.dump(vocab_manifest, fh, indent=2)
    print(f"Vocab manifest written to {manifest_path}")

    all_categories = list(WORD_CATEGORIES.keys()) + list(NUMERIC_CATEGORIES.keys())

    fieldnames = [
        "model", "quant_mode", "seed", "attribute", "kind", "interference_level", "trial_idx",
        "subject", "gold", "model_raw_response", "model_extracted",
        "correct", "latency_sec",
    ]
    f = open(out_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    skipped, run_count = 0, 0
    t_start = time.time()
    for category_key in all_categories:
        for level in INTERFERENCE_LEVELS:
            max_pool = (len(filtered_word_pools[category_key]) if category_key in WORD_CATEGORIES
                        else len(NUMERIC_CATEGORIES[category_key]["pool"]))
            if level > max_pool:
                continue
            n_trials = TRIALS_PER_LEVEL.get(level, 20)
            for trial_idx in range(n_trials):
                key = (category_key, level, trial_idx)
                if key in done:
                    skipped += 1
                    continue

                trial = build_trial(category_key, level, rng, filtered_word_pools)
                if trial is None:
                    continue

                t0 = time.time()
                response = generate_answer(tokenizer, model, trial["prompt"])
                latency = time.time() - t0
                correct, extracted = score_response(response, trial["gold"], trial["kind"])
                run_count += 1

                writer.writerow({
                    "model": args.model,
                    "quant_mode": args.quant,
                    "seed": args.seed,
                    "attribute": category_key,
                    "kind": trial["kind"],
                    "interference_level": level,
                    "trial_idx": trial_idx,
                    "subject": trial["subject"],
                    "gold": trial["gold"],
                    "model_raw_response": response.replace("\n", " ")[:200],
                    "model_extracted": extracted,
                    "correct": int(correct),
                    "latency_sec": round(latency, 3),
                })
                f.flush()

                if run_count % 25 == 0:
                    elapsed = time.time() - t_start
                    print(f"[{args.model}/{args.quant}/seed{args.seed}] {run_count} trials done in {elapsed/60:.1f} min "
                          f"({elapsed/run_count:.2f} sec/trial avg)")

                print(f"[{args.model}/{args.quant}/seed{args.seed}] {category_key:16s} level={level:3d} trial={trial_idx:3d} "
                      f"gold={trial['gold']:>10s} pred={str(extracted):>10s} correct={correct}")

    f.close()
    total_elapsed = time.time() - t_start
    print(f"\nDone [{args.model}/{args.quant}/seed{args.seed}]. {skipped} rows skipped (already completed), "
          f"{run_count} new trials run in {total_elapsed/60:.1f} min. Results saved to {out_path}")


if __name__ == "__main__":
    main()