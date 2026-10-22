"""Lightweight CPU-only checks for follow-up reproducibility helpers."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import hashlib
import random

try:
    from .followup_common import (
        load_or_create_manifest, manifest_trials, stable_generation_seed, token_span, update_values,
    )
except ImportError:
    from followup_common import (
        load_or_create_manifest, manifest_trials, stable_generation_seed, token_span, update_values,
    )


FAKE_PROTOCOL = SimpleNamespace(
    WORD_CATEGORIES={
        "mood": {"template": "{subj}'s mood is now {val}."},
        "favorite_color": {"template": "{subj}'s favorite color is now {val}."},
    },
    NUMERIC_CATEGORIES={"temperature": {"pool": [str(i) for i in range(20)]}},
    INTERFERENCE_LEVELS=[1, 2, 4, 8, 16],
    TRIALS_PER_LEVEL={1: 2, 2: 2, 4: 2, 8: 2, 16: 2},
    MODEL_REGISTRY={"mistral-7b": {"hf_id": "test/model"}},
)


def fake_build_trial(category, level, rng, pools):
    subjects = ["Sam", "Alex", "Jo"]
    subject = rng.choice(subjects)
    pool = pools[category] if category in FAKE_PROTOCOL.WORD_CATEGORIES else FAKE_PROTOCOL.NUMERIC_CATEGORIES[category]["pool"]
    values = rng.sample(pool, level)
    if category in FAKE_PROTOCOL.WORD_CATEGORIES:
        template = FAKE_PROTOCOL.WORD_CATEGORIES[category]["template"]
        prompt = "\\n".join(template.format(subj=subject, val=value) for value in values)
        kind = "word"
    else:
        prompt = f"temperature={','.join(values)}"
        kind = "numeric"
    return {"subject": subject, "prompt": prompt, "gold": values[-1], "kind": kind}


FAKE_PROTOCOL.build_trial = fake_build_trial


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, add_special_tokens=False):
        offsets, tokens = [], []
        for index, part in enumerate(text.split()):
            start = text.index(part, offsets[-1][1] if offsets else 0)
            end = start + len(part)
            tokens.append(index)
            offsets.append((start, end))
        return {"input_ids": [tokens], "offset_mapping": [offsets]}


class FollowupTests(unittest.TestCase):
    def test_manifest_is_stable_and_hashes_prompts(self):
        pools = {
            "mood": ["happy", "sad", "calm", "angry", "glad", "blue", "upset", "fine", "mad", "tired", "proud", "shy", "eager", "quiet", "brave", "kind"],
            "favorite_color": ["red", "blue", "green", "gold", "pink", "black", "white", "gray", "teal", "navy", "tan", "mint", "rose", "lime", "plum", "rust"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            first = load_or_create_manifest(
                path, "mistral-7b", pools, 2, protocol_override=FAKE_PROTOCOL
            )
            second = load_or_create_manifest(
                path, "mistral-7b", pools, 2, protocol_override=FAKE_PROTOCOL
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5 * 2 * len(FAKE_PROTOCOL.WORD_CATEGORIES))
            self.assertTrue(all(len(row["prompt_sha256"]) == 64 for row in first))

    def test_manifest_rejects_changed_tokenizer_pool(self):
        pools = {
            "mood": ["happy", "sad", "calm", "angry", "glad", "blue", "upset", "fine", "mad", "tired", "proud", "shy", "eager", "quiet", "brave", "kind"],
            "favorite_color": ["red", "blue", "green", "gold", "pink", "black", "white", "gray", "teal", "navy", "tan", "mint", "rose", "lime", "plum", "rust"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            load_or_create_manifest(path, "mistral-7b", pools, 2, protocol_override=FAKE_PROTOCOL)
            altered = dict(pools)
            altered["mood"] = pools["mood"][:-1]
            with self.assertRaises(ValueError):
                load_or_create_manifest(path, "mistral-7b", altered, 2, protocol_override=FAKE_PROTOCOL)

    def test_manifest_matches_primary_rng_replay_at_key_trials(self):
        pools = {
            "mood": ["happy", "sad", "calm", "angry", "glad", "blue", "upset", "fine", "mad", "tired", "proud", "shy", "eager", "quiet", "brave", "kind"],
            "favorite_color": ["red", "blue", "green", "gold", "pink", "black", "white", "gray", "teal", "navy", "tan", "mint", "rose", "lime", "plum", "rust"],
        }
        actual = manifest_trials("mistral-7b", pools, 2, seeds=[42], protocol_override=FAKE_PROTOCOL)
        rng = random.Random(42)
        expected = []
        categories = list(FAKE_PROTOCOL.WORD_CATEGORIES) + list(FAKE_PROTOCOL.NUMERIC_CATEGORIES)
        for category in categories:
            pool_size = len(pools[category]) if category in pools else len(FAKE_PROTOCOL.NUMERIC_CATEGORIES[category]["pool"])
            for level in FAKE_PROTOCOL.INTERFERENCE_LEVELS:
                if level > pool_size:
                    continue
                for trial_idx in range(FAKE_PROTOCOL.TRIALS_PER_LEVEL[level]):
                    trial = fake_build_trial(category, level, rng, pools)
                    if category in FAKE_PROTOCOL.WORD_CATEGORIES and level == 16 and trial_idx < 2:
                        expected.append((category, level, trial_idx, trial["prompt"], trial["gold"]))
        observed = [(row["attribute"], row["interference_level"], row["trial_idx"], row["prompt"], row["gold"]) for row in actual]
        self.assertEqual(observed, expected)

    def test_update_values_match_prompt_history(self):
        row = {
            "attribute": "mood", "subject": "Sam", "gold": "happy",
            "prompt": "Sam's mood is now sad.\nSam's mood is now happy.\n\nQuestion?",
        }
        self.assertEqual(update_values(row, FAKE_PROTOCOL), ["sad", "happy"])

    def test_value_token_span_uses_offsets(self):
        prompt = "Sam's mood is now happy.\n\nQuestion?"
        rendered = f"<user>{prompt}</user>"
        offsets = [(i, i + 1) for i in range(len(rendered))]
        positions = token_span(rendered, prompt, "happy", offsets)
        self.assertEqual(len(positions), len("happy"))

    def test_repeat_seeds_are_stable_and_distinct(self):
        row = {"model":"qwen2.5-7b", "seed":42, "attribute":"mood",
               "interference_level":64, "trial_idx":3}
        self.assertEqual(stable_generation_seed(row, 0), stable_generation_seed(row, 0))
        self.assertNotEqual(stable_generation_seed(row, 0), stable_generation_seed(row, 1))


if __name__ == "__main__":
    unittest.main()
