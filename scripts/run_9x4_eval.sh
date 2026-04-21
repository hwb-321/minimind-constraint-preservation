#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

run_method() {
  local gpu="$1"
  local method="$2"
  shift 2
  local method_args=("$@")

  mkdir -p "$ROOT/result_all/$method"

  while IFS='|' read -r tag data extra_args; do
    echo "[$method] Running $tag on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/eval_calc_dataset.py \
      --weight full_sft_calc \
      --data_path "$data" \
      --match_mode answer_only \
      --device cuda:0 \
      --results_path "$ROOT/result_all/$method/${tag}.jsonl" \
      "${method_args[@]}" \
      | tee "$ROOT/result_all/$method/${tag}.log"
  done <<'EOF'
clean|./dataset/sft_calc_addition_test.jsonl|
alpha_len16|./dataset/sft_calc_addition_test_alpha_noise_len16_seed42.jsonl|
digit_len16|./dataset/sft_calc_addition_test_digit_noise_len16_seed42.jsonl|
alpha_len64|./dataset/sft_calc_addition_test_alpha_noise_len64_seed42.jsonl|
digit_len64|./dataset/sft_calc_addition_test_digit_noise_len64_seed42.jsonl|
alpha_len256|./dataset/sft_calc_addition_test_alpha_noise_len256_seed42.jsonl|
digit_len256|./dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl|
alpha_len1024|./dataset/sft_calc_addition_test_alpha_noise_len1024_seed42.jsonl|
digit_len1024|./dataset/sft_calc_addition_test_digit_noise_len1024_seed42.jsonl|
EOF
}

run_method 0 baseline &
run_method 1 yarn_orig768 --inference_rope_scaling --yarn_original_max_position_embeddings 768 &
run_method 2 block_keep4 --targeted_attention_mode block --attention_keep_prefix 4 &

wait
echo "All evaluations completed."
