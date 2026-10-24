"""CPU-only input-boundary tests; no released dataset or model fitting required."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from revision_stats import K, main, validate_inputs


class ValidateInputsTests(unittest.TestCase):
    def setUp(self):
        observations, reconstruction = [], []
        for trial, kind, attribute, gold, history in (
            (0, "word", "mood", "happy", "happy|happy"),
            (1, "numeric", "temperature", "42", "11|42"),
        ):
            keys = dict(zip(K, ["qwen2.5-7b", 42, attribute, 2, trial]))
            metadata = dict(subject="Sam", gold=gold)
            reconstruction.append(dict(**keys, **metadata, distractor_sequence=history))
            for quant, correct in (("fp16", 1), ("int8", 1), ("int4", 0)):
                observations.append(dict(**keys, **metadata, kind=kind,
                                         quant_mode=quant, correct=correct))
        self.df = pd.DataFrame(observations)
        self.rt = pd.DataFrame(reconstruction)

    def test_valid_data_preserves_legacy_values_and_order(self):
        df = self.df.sample(frac=1, random_state=7)
        rt = self.rt.iloc[::-1].copy()
        original_df, original_rt = df.copy(deep=True), rt.copy(deep=True)
        legacy_rt = rt.assign(dup=rt.distractor_sequence.str.split("|").map(
            lambda s: len(s) != len(set(s))))
        expected = df.pivot_table(index=K + ["kind"], columns="quant_mode",
                                  values="correct").reset_index().merge(
                                      legacy_rt[K + ["dup"]], on=K)
        result = validate_inputs(df, rt)
        assert_frame_equal(result, expected, check_dtype=False)
        assert_frame_equal(df, original_df)
        assert_frame_equal(rt, original_rt)
        self.assertEqual(len(result), 2)

    def test_repeated_history_values_are_valid(self):
        result = validate_inputs(self.df, self.rt)
        self.assertTrue(result.loc[result.kind == "word", "dup"].item())
        self.assertFalse(result.loc[result.kind == "numeric", "dup"].item())

    def test_numeric_metadata_is_string_safe(self):
        self.rt.loc[1, "gold"] = 42
        self.assertEqual(len(validate_inputs(self.df, self.rt)), 2)
        self.rt.loc[1, "gold"] = "42"
        self.df.loc[self.df.kind == "numeric", "gold"] = 42
        self.assertEqual(len(validate_inputs(self.df, self.rt)), 2)

    def test_all_known_models_are_valid(self):
        for model in ("qwen2.5-7b", "mistral-7b", "phi3.5-mini"):
            with self.subTest(model=model):
                validate_inputs(self.df.assign(model=model), self.rt.assign(model=model))

    def test_arbitrary_positive_integer_level_is_valid(self):
        df, rt = self.df.assign(interference_level=3), self.rt.assign(interference_level=3)
        rt["distractor_sequence"] = ["sad|happy|happy", "10|11|42"]
        validate_inputs(df, rt)

    def test_required_columns(self):
        for name, frame in (("observations", self.df), ("reconstruction", self.rt)):
            for column in frame.columns:
                with self.subTest(frame=name, column=column):
                    df, rt = self.df, self.rt
                    if name == "observations":
                        df = df.drop(columns=column)
                    else:
                        rt = rt.drop(columns=column)
                    with self.assertRaisesRegex(ValueError, "missing required columns"):
                        validate_inputs(df, rt)

    def test_null_required_values(self):
        for name, frame in (("observations", self.df), ("reconstruction", self.rt)):
            for column in frame.columns:
                with self.subTest(frame=name, column=column):
                    df, rt = self.df.copy(), self.rt.copy()
                    target = df if name == "observations" else rt
                    target[column] = target[column].astype(object)
                    target.loc[0, column] = None
                    with self.assertRaisesRegex(ValueError, "null required"):
                        validate_inputs(df, rt)

    def test_empty_inputs(self):
        for df, rt in ((self.df.iloc[:0], self.rt), (self.df, self.rt.iloc[:0])):
            with self.assertRaisesRegex(ValueError, "no trials"):
                validate_inputs(df, rt)

    def test_duplicate_observations(self):
        df = pd.concat([self.df, self.df.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate trial/precision"):
            validate_inputs(df, self.rt)

    def test_duplicate_observations_with_different_metadata(self):
        duplicate = self.df.iloc[[0]].assign(kind="numeric", subject="Adam", gold="other")
        df = pd.concat([self.df, duplicate], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate trial/precision"):
            validate_inputs(df, self.rt)

    def test_duplicate_reconstruction(self):
        rt = pd.concat([self.rt, self.rt.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate trial"):
            validate_inputs(self.df, rt)

    def test_missing_precision(self):
        for precision in ("fp16", "int8", "int4"):
            with self.subTest(precision=precision):
                df = self.df.drop(self.df.index[self.df.quant_mode.eq(precision)][0])
                with self.assertRaisesRegex(ValueError, "requires fp16, int8, and int4"):
                    validate_inputs(df, self.rt)

    def test_missing_reconstruction(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_inputs(self.df, self.rt.iloc[[0]])

    def test_extra_reconstruction(self):
        extra = self.rt.iloc[[0]].assign(trial_idx=99)
        rt = pd.concat([self.rt, extra], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_inputs(self.df, rt)

    def test_unknown_categories(self):
        for column, value, message in (("model", "unknown", "unknown model"),
                                       ("kind", "text", "unknown kind"),
                                       ("quant_mode", "int2", "unknown precision")):
            with self.subTest(column=column):
                df = self.df.copy()
                df.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_inputs(df, self.rt)
        with self.assertRaisesRegex(ValueError, "unknown model"):
            validate_inputs(self.df, self.rt.assign(model="unknown"))

    def test_invalid_levels(self):
        for level in (0, -1, 1.5, np.inf, -np.inf, np.nan, "bad", "2", True):
            for name in ("observations", "reconstruction"):
                with self.subTest(level=level, frame=name):
                    df, rt = self.df, self.rt
                    if name == "observations":
                        df = df.assign(interference_level=level)
                    else:
                        rt = rt.assign(interference_level=level)
                    with self.assertRaises(ValueError):
                        validate_inputs(df, rt)

    def test_invalid_correctness(self):
        for correct in (-1, 2, 0.5, np.inf, np.nan, "1", "bad"):
            with self.subTest(correct=correct):
                with self.assertRaises(ValueError):
                    validate_inputs(self.df.assign(correct=correct), self.rt)

    def test_inconsistent_observation_metadata(self):
        for column, value in (("kind", "numeric"), ("subject", "Adam"), ("gold", "sad")):
            with self.subTest(column=column):
                df = self.df.copy()
                df.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    validate_inputs(df, self.rt)

    def test_reconstruction_metadata_mismatch(self):
        for column, value in (("subject", "Adam"), ("gold", "sad")):
            with self.subTest(column=column):
                rt = self.rt.copy()
                rt.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, "does not match"):
                    validate_inputs(self.df, rt)

    def test_invalid_history(self):
        for history in ("", "happy", "sad|sad", "|happy", "sad||happy", " |happy",
                        "sad|happy|happy", "happy|"):
            with self.subTest(history=history):
                rt = self.rt.copy()
                rt.loc[0, "distractor_sequence"] = history
                with self.assertRaisesRegex(ValueError, "history"):
                    validate_inputs(self.df, rt)

    def run_main(self, out, df=None):
        with patch("sys.argv", ["revision_stats.py", "--data", "unused", "--out", str(out)]), \
             patch("revision_stats.pd.read_csv", side_effect=[self.df if df is None else df,
                                                            self.rt]), \
             patch("revision_stats.smf.gee") as gee:
            try:
                main()
            finally:
                gee.assert_not_called()

    def test_main_validates_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            out = Path(directory) / "new-output"
            with self.assertRaisesRegex(ValueError, "binary"):
                self.run_main(out, self.df.assign(correct=2))
            self.assertFalse(out.exists())

    def test_main_rejects_existing_empty_output(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            with self.assertRaises(FileExistsError):
                self.run_main(directory)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_main_preserves_existing_output(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
            sentinel = Path(directory) / "summary.json"
            sentinel.write_text("do not replace")
            with self.assertRaises(FileExistsError):
                self.run_main(directory)
            self.assertEqual(sentinel.read_text(), "do not replace")
            self.assertEqual(list(Path(directory).iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()
