#!/usr/bin/env bash
set -euo pipefail

# Run one condition:
# ./scripts/run_main.sh qwen2.5-7b int4 42

MODEL="${1:?model required}"
QUANT="${2:?quantization required: fp16|int8|int4}"
SEED="${3:?seed required}"

python src/pi_llm_eval_v2.py   --model "$MODEL"   --quant "$QUANT"   --seed "$SEED"   --out "results_v2_${MODEL}_${QUANT}_seed${SEED}.csv"
