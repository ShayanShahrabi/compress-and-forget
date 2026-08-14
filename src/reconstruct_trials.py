"""
Run this ON THE CLUSTER (needs tokenizer access, no GPU needed), in the
same environment as your original eval script.
Produces reconstructed_trials.csv -- upload that file back.

Regenerates, for every (model, seed, attribute, interference_level,
trial_idx), the EXACT subject + full ordered value sequence used in that
trial -- by replaying the same tokenizer-filtered vocab construction and
the same seeded RNG draw sequence, in the same nested-loop order, as the
original pi_llm_eval_v2.py. This works because we already confirmed (via
the pairing-integrity check) that every run completed in a single pass
with no resume/skip, so the RNG draw order is fully deterministic from the
seed alone.

This does NOT re-run any model inference -- it only reconstructs the
prompt-generation side, which is CPU-only and fast.
"""
import csv
import json

from transformers import AutoTokenizer

# ---- copied verbatim from the original eval script for exact fidelity ----
MODEL_REGISTRY = {
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}

INTERFERENCE_LEVELS = [1, 2, 4, 8, 16, 32, 64, 96]
TRIALS_PER_LEVEL = {1: 15, 2: 15, 4: 20, 8: 25, 16: 60, 32: 50, 64: 60, 96: 60}
SEEDS = [42, 123, 999, 7, 2024]

SUBJECTS = [
    "Adam", "Maria", "Sam", "Elena", "Marcus", "Priya", "Noah", "Ines",
    "Diego", "Yuki", "Omar", "Hannah", "Leo", "Fatima", "Chen", "Nora",
    "Victor", "Amara", "Felix", "Sana",
]

# Paste the candidate lists exactly as in pi_llm_eval_v2_2_.py
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
    "mood": {"template": "{subj}'s mood is now {val}.", "candidates": MOOD_CANDIDATES},
    "favorite_color": {"template": "{subj}'s favorite color is now {val}.", "candidates": COLOR_CANDIDATES},
    "favorite_animal": {"template": "{subj}'s favorite animal is now the {val}.", "candidates": ANIMAL_CANDIDATES},
    "occupation": {"template": "{subj}'s job is now {val}.", "candidates": OCCUPATION_CANDIDATES},
}
NUMERIC_CATEGORIES = {
    "temperature": {"template": "The temperature reading for {subj}'s greenhouse is now {val} degrees.",
                     "pool": [str(n) for n in range(10, 999)]},
    "stock_price":  {"template": "{subj}'s tracked stock price is now ${val}.",
                     "pool": [str(n) for n in range(5, 999)]},
    "page_count":   {"template": "The page count for {subj}'s document is now {val} pages.",
                     "pool": [str(n) for n in range(3, 999)]},
}


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


def build_filtered_vocab(tokenizer):
    filtered = {}
    check_subj = "Sam"
    for name, spec in WORD_CATEGORIES.items():
        kept = [w for w in spec["candidates"]
                if is_single_token_in_context(tokenizer, spec["template"], check_subj, w)]
        filtered[name] = kept
    return filtered


def build_trial(category_key, interference_level, rng, filtered_word_pools):
    if category_key in WORD_CATEGORIES:
        pool = filtered_word_pools[category_key]
    else:
        pool = NUMERIC_CATEGORIES[category_key]["pool"]
    n = interference_level
    if n > len(pool):
        return None
    subj = rng.choice(SUBJECTS)
    values = rng.sample(pool, n)
    return {"subject": subj, "values": values, "gold": values[-1]}


import random

rows = []
for model_key, hf_id in MODEL_REGISTRY.items():
    print(f"Loading tokenizer for {model_key} ({hf_id})...")
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    filtered_word_pools = build_filtered_vocab(tokenizer)
    all_categories = list(WORD_CATEGORIES.keys()) + list(NUMERIC_CATEGORIES.keys())

    for seed in SEEDS:
        rng = random.Random(seed)
        # Replay the EXACT same nested loop order as the original script,
        # with no resume/skip (confirmed clean via the pairing-integrity check).
        for category_key in all_categories:
            for level in INTERFERENCE_LEVELS:
                max_pool = (len(filtered_word_pools[category_key]) if category_key in WORD_CATEGORIES
                            else len(NUMERIC_CATEGORIES[category_key]["pool"]))
                if level > max_pool:
                    continue
                n_trials = TRIALS_PER_LEVEL.get(level, 20)
                for trial_idx in range(n_trials):
                    trial = build_trial(category_key, level, rng, filtered_word_pools)
                    if trial is None:
                        continue
                    rows.append({
                        "model": model_key,
                        "seed": seed,
                        "attribute": category_key,
                        "interference_level": level,
                        "trial_idx": trial_idx,
                        "subject": trial["subject"],
                        "gold": trial["gold"],
                        "distractor_sequence": "|".join(trial["values"]),  # full ordered value list
                    })
    print(f"  {model_key}: {sum(1 for r in rows if r['model']==model_key)} trials reconstructed")

with open("reconstructed_trials.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "seed", "attribute", "interference_level",
                                            "trial_idx", "subject", "gold", "distractor_sequence"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nSaved reconstructed_trials.csv ({len(rows)} rows). Upload this file.")
print("Sanity check tip: 'subject' and 'gold' columns here should exactly match")
print("the same columns in your combined_raw.csv.gz for the same (model, seed,")
print("attribute, interference_level, trial_idx) -- if they don't all match,")
print("something about the vocab lists above differs from your original run.")