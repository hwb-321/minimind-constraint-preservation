#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$ROOT/result_all_attention_fix/block_keep4"

cd "$ROOT"
mkdir -p "$OUT_DIR"

run_eval() {
  local gpu="$1"
  local tag="$2"
  local data_path="$3"

  echo "[GPU ${gpu}] Running ${tag}"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_calc_dataset.py \
    --weight full_sft_calc \
    --data_path "$data_path" \
    --match_mode answer_only \
    --device cuda:0 \
    --targeted_attention_mode block \
    --attention_keep_prefix 4 \
    --results_path "$OUT_DIR/${tag}.jsonl" \
    | tee "$OUT_DIR/${tag}.log"
}

run_group_gpu0() {
  run_eval 0 clean ./dataset/sft_calc_addition_test.jsonl
  run_eval 0 alpha_len16 ./dataset/sft_calc_addition_test_alpha_noise_len16_seed42.jsonl
  run_eval 0 digit_len16 ./dataset/sft_calc_addition_test_digit_noise_len16_seed42.jsonl
}

run_group_gpu1() {
  run_eval 1 alpha_len64 ./dataset/sft_calc_addition_test_alpha_noise_len64_seed42.jsonl
  run_eval 1 digit_len64 ./dataset/sft_calc_addition_test_digit_noise_len64_seed42.jsonl
  run_eval 1 alpha_len256 ./dataset/sft_calc_addition_test_alpha_noise_len256_seed42.jsonl
}

run_group_gpu2() {
  run_eval 2 digit_len256 ./dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl
  run_eval 2 alpha_len1024 ./dataset/sft_calc_addition_test_alpha_noise_len1024_seed42.jsonl
  run_eval 2 digit_len1024 ./dataset/sft_calc_addition_test_digit_noise_len1024_seed42.jsonl
}

run_group_gpu0 &
run_group_gpu1 &
run_group_gpu2 &

wait
echo "All attention-block evaluations completed."
