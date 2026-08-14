# Interference Endurance Score (IES): FP16 vs INT8 vs INT4

IES follows the PI-LLM definition (Wang & Sun, arXiv:2506.08184): the area under the accuracy-vs-interference-level curve, with the x-axis log-scaled. We report both the raw AUC (log2-x space, directly matching PI-LLM's quantity) and a normalized AUC (raw AUC divided by the log2 x-range, giving a value in [0,1] on the same scale as accuracy -- use this version when comparing arms tested over different interference-level ranges).

## IES by model, quant mode, and attribute kind

| model       | quant_mode   | kind    |   n_levels | levels_tested       |   IES_raw_auc |   IES_normalized |   log2_x_range |
|:------------|:-------------|:--------|-----------:|:--------------------|--------------:|-----------------:|---------------:|
| mistral-7b  | fp16         | numeric |          8 | 1,2,4,8,16,32,64,96 |        5.9112 |           0.8977 |          6.585 |
| mistral-7b  | int8         | numeric |          8 | 1,2,4,8,16,32,64,96 |        5.8666 |           0.8909 |          6.585 |
| mistral-7b  | int4         | numeric |          8 | 1,2,4,8,16,32,64,96 |        5.7576 |           0.8744 |          6.585 |
| mistral-7b  | fp16         | word    |          6 | 1,2,4,8,16,32       |        3.4873 |           0.6975 |          5     |
| mistral-7b  | int8         | word    |          6 | 1,2,4,8,16,32       |        3.3982 |           0.6796 |          5     |
| mistral-7b  | int4         | word    |          6 | 1,2,4,8,16,32       |        3.3098 |           0.662  |          5     |
| phi3.5-mini | fp16         | numeric |          8 | 1,2,4,8,16,32,64,96 |        4.6382 |           0.7044 |          6.585 |
| phi3.5-mini | int8         | numeric |          8 | 1,2,4,8,16,32,64,96 |        4.3838 |           0.6657 |          6.585 |
| phi3.5-mini | int4         | numeric |          8 | 1,2,4,8,16,32,64,96 |        4.8231 |           0.7324 |          6.585 |
| phi3.5-mini | fp16         | word    |          5 | 1,2,4,8,16          |        3.2786 |           0.8196 |          4     |
| phi3.5-mini | int8         | word    |          5 | 1,2,4,8,16          |        3.1627 |           0.7907 |          4     |
| phi3.5-mini | int4         | word    |          5 | 1,2,4,8,16          |        3.1092 |           0.7773 |          4     |
| qwen2.5-7b  | fp16         | numeric |          8 | 1,2,4,8,16,32,64,96 |        6.4297 |           0.9764 |          6.585 |
| qwen2.5-7b  | int8         | numeric |          8 | 1,2,4,8,16,32,64,96 |        6.4344 |           0.9771 |          6.585 |
| qwen2.5-7b  | int4         | numeric |          8 | 1,2,4,8,16,32,64,96 |        6.4262 |           0.9759 |          6.585 |
| qwen2.5-7b  | fp16         | word    |          7 | 1,2,4,8,16,32,64    |        5.8508 |           0.9751 |          6     |
| qwen2.5-7b  | int8         | word    |          7 | 1,2,4,8,16,32,64    |        5.8313 |           0.9719 |          6     |
| qwen2.5-7b  | int4         | word    |          7 | 1,2,4,8,16,32,64    |        5.7442 |           0.9574 |          6     |


## Quantization penalty (delta_IES relative to FP16)

| model       | kind    | quant_mode   |   IES_fp16 |   IES_quant |   delta_IES |   pct_IES_lost |
|:------------|:--------|:-------------|-----------:|------------:|------------:|---------------:|
| mistral-7b  | numeric | int8         |     0.8977 |      0.8909 |      0.0068 |           0.76 |
| mistral-7b  | numeric | int4         |     0.8977 |      0.8744 |      0.0233 |           2.6  |
| mistral-7b  | word    | int8         |     0.6975 |      0.6796 |      0.0179 |           2.57 |
| mistral-7b  | word    | int4         |     0.6975 |      0.662  |      0.0355 |           5.09 |
| phi3.5-mini | numeric | int8         |     0.7044 |      0.6657 |      0.0387 |           5.49 |
| phi3.5-mini | numeric | int4         |     0.7044 |      0.7324 |     -0.028  |          -3.98 |
| phi3.5-mini | word    | int8         |     0.8196 |      0.7907 |      0.0289 |           3.53 |
| phi3.5-mini | word    | int4         |     0.8196 |      0.7773 |      0.0423 |           5.16 |
| qwen2.5-7b  | numeric | int8         |     0.9764 |      0.9771 |     -0.0007 |          -0.07 |
| qwen2.5-7b  | numeric | int4         |     0.9764 |      0.9759 |      0.0005 |           0.05 |
| qwen2.5-7b  | word    | int8         |     0.9751 |      0.9719 |      0.0032 |           0.33 |
| qwen2.5-7b  | word    | int4         |     0.9751 |      0.9574 |      0.0177 |           1.82 |


## Suggested paper text (draft -- edit before pasting)

> To allow standardized comparison with PI-LLM (Wang & Sun, 2025) and to quantify the quantization penalty in a single metric, we computed the Interference Endurance Score (IES) -- the area under the accuracy-vs-interference-level curve on a log-scaled x-axis -- for each model under FP16, INT8, and INT4. [EDIT: insert the specific delta_IES / pct_IES_lost numbers from the table above once reviewed; do not paste this placeholder as-is.]
