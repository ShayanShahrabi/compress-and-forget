# Ablation: forced lm_head quantization (V2)

Baseline file used: `./combined_raw.csv.gz`

## Descriptive accuracy at each model's key interference level (word-type attributes)

| model       |   interference_level | quant_mode         |   n_trials |   accuracy |   n_correct |
|:------------|---------------------:|:-------------------|-----------:|-----------:|------------:|
| qwen2.5-7b  |                   64 | fp16               |        300 |     0.81   |         243 |
| qwen2.5-7b  |                   64 | int4               |        300 |     0.6833 |         205 |
| qwen2.5-7b  |                   64 | int4_forced_lmhead |        300 |     0.68   |         204 |
| qwen2.5-7b  |                   64 | int8               |        300 |     0.8    |         240 |
| mistral-7b  |                   16 | fp16               |       1200 |     0.44   |         528 |
| mistral-7b  |                   16 | int4               |       1200 |     0.3742 |         449 |
| mistral-7b  |                   16 | int4_forced_lmhead |       1200 |     0.3733 |         448 |
| mistral-7b  |                   16 | int8               |       1200 |     0.4225 |         507 |
| phi3.5-mini |                    8 | fp16               |        500 |     0.618  |         309 |
| phi3.5-mini |                    8 | int4               |        500 |     0.54   |         270 |
| phi3.5-mini |                    8 | int4_forced_lmhead |        500 |     0.54   |         270 |
| phi3.5-mini |                    8 | int8               |        500 |     0.576  |         288 |


## Paired McNemar's tests


### qwen2.5-7b (level=64)

**int4_forced_lmhead_vs_fp16**

```
{
  "n_paired": 300,
  "acc_fp16": 0.81,
  "acc_int4_forced_lmhead": 0.68,
  "both_correct": 201,
  "both_wrong": 54,
  "fp16_only_correct": 42,
  "int4_forced_lmhead_only_correct": 3,
  "statistic": 3.0,
  "pvalue": 8.654978955746628e-10,
  "exact_used": true
}
```

**int4_forced_lmhead_vs_int4**

```
{
  "n_paired": 300,
  "acc_int4": 0.6833,
  "acc_int4_forced_lmhead": 0.68,
  "both_correct": 201,
  "both_wrong": 92,
  "int4_only_correct": 4,
  "int4_forced_lmhead_only_correct": 3,
  "statistic": 3.0,
  "pvalue": 1.0,
  "exact_used": true
}
```

**int8_vs_fp16**

```
{
  "n_paired": 300,
  "acc_fp16": 0.81,
  "acc_int8": 0.8,
  "both_correct": 233,
  "both_wrong": 50,
  "fp16_only_correct": 10,
  "int8_only_correct": 7,
  "statistic": 7.0,
  "pvalue": 0.629058837890625,
  "exact_used": true
}
```

**int8_vs_int4**

```
{
  "n_paired": 300,
  "acc_int4": 0.6833,
  "acc_int8": 0.8,
  "both_correct": 200,
  "both_wrong": 55,
  "int4_only_correct": 5,
  "int8_only_correct": 40,
  "statistic": 5.0,
  "pvalue": 7.878384167270269e-08,
  "exact_used": true
}
```

**Interpretation for qwen2.5-7b:** backbone_driven

```
{
  "backbone_only_gap_vs_fp16": 0.1267,
  "additional_gap_from_quantizing_lmhead": 0.0033,
  "total_gap_fp16_to_full_int4": 0.13,
  "fraction_of_total_penalty_from_lmhead": 0.025
}
```


### mistral-7b (level=16)

**int4_forced_lmhead_vs_fp16**

```
{
  "n_paired": 1200,
  "acc_fp16": 0.44,
  "acc_int4_forced_lmhead": 0.3733,
  "both_correct": 399,
  "both_wrong": 623,
  "fp16_only_correct": 129,
  "int4_forced_lmhead_only_correct": 49,
  "statistic": 35.061797752808985,
  "pvalue": 3.194059162570393e-09,
  "exact_used": false
}
```

**int4_forced_lmhead_vs_int4**

```
{
  "n_paired": 1200,
  "acc_int4": 0.3742,
  "acc_int4_forced_lmhead": 0.3733,
  "both_correct": 445,
  "both_wrong": 748,
  "int4_only_correct": 4,
  "int4_forced_lmhead_only_correct": 3,
  "statistic": 3.0,
  "pvalue": 1.0,
  "exact_used": true
}
```

**int8_vs_fp16**

```
{
  "n_paired": 1200,
  "acc_fp16": 0.44,
  "acc_int8": 0.4225,
  "both_correct": 482,
  "both_wrong": 647,
  "fp16_only_correct": 46,
  "int8_only_correct": 25,
  "statistic": 5.633802816901408,
  "pvalue": 0.017617372234518848,
  "exact_used": false
}
```

**int8_vs_int4**

```
{
  "n_paired": 1200,
  "acc_int4": 0.3742,
  "acc_int8": 0.4225,
  "both_correct": 386,
  "both_wrong": 630,
  "int4_only_correct": 63,
  "int8_only_correct": 121,
  "statistic": 17.657608695652176,
  "pvalue": 2.644552244027362e-05,
  "exact_used": false
}
```

**Interpretation for mistral-7b:** backbone_driven

```
{
  "backbone_only_gap_vs_fp16": 0.0658,
  "additional_gap_from_quantizing_lmhead": 0.0009,
  "total_gap_fp16_to_full_int4": 0.0667,
  "fraction_of_total_penalty_from_lmhead": 0.013
}
```


### phi3.5-mini (level=8)

**int4_forced_lmhead_vs_fp16**

```
{
  "n_paired": 500,
  "acc_fp16": 0.618,
  "acc_int4_forced_lmhead": 0.54,
  "both_correct": 254,
  "both_wrong": 175,
  "fp16_only_correct": 55,
  "int4_forced_lmhead_only_correct": 16,
  "statistic": 16.0,
  "pvalue": 3.753237688593051e-06,
  "exact_used": true
}
```

**int4_forced_lmhead_vs_int4**

```
{
  "n_paired": 500,
  "acc_int4": 0.54,
  "acc_int4_forced_lmhead": 0.54,
  "both_correct": 267,
  "both_wrong": 227,
  "int4_only_correct": 3,
  "int4_forced_lmhead_only_correct": 3,
  "statistic": 3.0,
  "pvalue": 1.0,
  "exact_used": true
}
```

**int8_vs_fp16**

```
{
  "n_paired": 500,
  "acc_fp16": 0.618,
  "acc_int8": 0.576,
  "both_correct": 274,
  "both_wrong": 177,
  "fp16_only_correct": 35,
  "int8_only_correct": 14,
  "statistic": 14.0,
  "pvalue": 0.003801654409748778,
  "exact_used": true
}
```

**int8_vs_int4**

```
{
  "n_paired": 500,
  "acc_int4": 0.54,
  "acc_int8": 0.576,
  "both_correct": 243,
  "both_wrong": 185,
  "int4_only_correct": 27,
  "int8_only_correct": 45,
  "statistic": 4.013888888888889,
  "pvalue": 0.04512694888867611,
  "exact_used": false
}
```

**Interpretation for phi3.5-mini:** backbone_driven

```
{
  "backbone_only_gap_vs_fp16": 0.078,
  "additional_gap_from_quantizing_lmhead": 0.0,
  "total_gap_fp16_to_full_int4": 0.078,
  "fraction_of_total_penalty_from_lmhead": 0.0
}
```


## Suggested Limitations-section paragraph (draft -- edit before pasting)

> We conducted a follow-up ablation directly quantizing the output projection (`lm_head`) to 4-bit precision, having found that our primary `int4` condition left `lm_head` at full precision by default. Comparing FP16, backbone-only INT4, and INT4 with `lm_head` also quantized isolates the output projection's contribution to the interference penalty. Results were qwen2.5-7b: backbone_driven; mistral-7b: backbone_driven; phi3.5-mini: backbone_driven. [EDIT: convert this into 1-2 narrative sentences once you've reviewed the numbers above; do not paste the bracketed placeholder as-is.]
