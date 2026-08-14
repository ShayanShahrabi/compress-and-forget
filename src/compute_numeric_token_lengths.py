"""
Run this ON THE CLUSTER (needs tokenizer access, no GPU needed).
Produces numeric_token_lengths.csv -- upload that file back.

For every numeric value used in the experiment (3-998), computes how many
tokens it occupies AS IT ACTUALLY APPEARS in each attribute's template
sentence, per model tokenizer -- using the same offset-mapping method as
is_single_token_in_context() in the original eval script, just without the
single-token filter (we want the raw count here, not a yes/no).
"""
import pandas as pd
from transformers import AutoTokenizer

MODEL_REGISTRY = {
    "qwen2.5-7b":  "Qwen/Qwen2.5-7B-Instruct",
    "mistral-7b":  "mistralai/Mistral-7B-Instruct-v0.3",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}

NUMERIC_CATEGORIES = {
    "temperature": {"template": "The temperature reading for {subj}'s greenhouse is now {val} degrees.",
                     "pool": [str(n) for n in range(10, 999)]},
    "stock_price":  {"template": "{subj}'s tracked stock price is now ${val}.",
                     "pool": [str(n) for n in range(5, 999)]},
    "page_count":   {"template": "The page count for {subj}'s document is now {val} pages.",
                     "pool": [str(n) for n in range(3, 999)]},
}

CHECK_SUBJ = "Sam"

def token_count_in_context(tokenizer, template, subj, val):
    prefix_template, suffix_template = template.split("{val}")
    prefix = prefix_template.format(subj=subj)
    suffix = suffix_template.format(subj=subj)
    text = prefix + val + suffix
    start, end = len(prefix), len(prefix) + len(val)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    overlapping = [i for i, (s, e) in enumerate(offsets) if not (e <= start or s >= end)]
    return len(overlapping)

rows = []
for model_key, hf_id in MODEL_REGISTRY.items():
    print(f"Loading tokenizer for {model_key} ({hf_id})...")
    tok = AutoTokenizer.from_pretrained(hf_id)
    for attr, spec in NUMERIC_CATEGORIES.items():
        for val in spec["pool"]:
            n_tok = token_count_in_context(tok, spec["template"], CHECK_SUBJ, val)
            rows.append({"model": model_key, "attribute": attr, "gold": val, "token_count": n_tok})
    print(f"  done: {sum(1 for r in rows if r['model']==model_key)} values tokenized")

out = pd.DataFrame(rows)
out.to_csv("numeric_token_lengths.csv", index=False)
print(f"\nSaved numeric_token_lengths.csv ({len(out)} rows). Upload this file.")