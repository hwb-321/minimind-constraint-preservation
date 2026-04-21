# Calc Prefix Noise Experiments

归档说明：本文件是 prefix-noise 实验早期命令备忘录，仅供实验溯源。当前主入口请优先参考项目根目录的 `README_PROJECT.md`。

本文件整理了 `clean` vs `prefix noise` 实验所需的命令。

目标模型：

- `out/full_sft_calc_768.pth`

基础测试集：

- `dataset/sft_calc_addition_test.jsonl`

## 1. 生成前缀噪声测试集

脚本：

- `scripts/generate_calc_prefix_noise_dataset.py`

支持两类前缀噪声：

- `alpha_noise`：随机字母，近似任务无关噪声
- `digit_noise`：随机数字和空格，近似任务表面相关噪声

长度按**字符数**控制。

### 单个长度示例：长度 1024，种子 42

```bash
python scripts/generate_calc_prefix_noise_dataset.py \
  --length 1024 \
  --seed 42
```

默认会生成两个文件：

- `dataset/sft_calc_addition_test_alpha_noise_len1024_seed42.jsonl`
- `dataset/sft_calc_addition_test_digit_noise_len1024_seed42.jsonl`

### 一键生成长度 Sweep

脚本：

- `scripts/generate_calc_prefix_noise_sweep.py`

默认会生成以下长度：

- `16`
- `32`
- `64`
- `128`
- `256`
- `512`
- `1024`

命令：

```bash
python scripts/generate_calc_prefix_noise_sweep.py --seed 42
```

## 2. 一键全量评估 Sweep

脚本：

- `scripts/eval_calc_prefix_noise_sweep.py`

这个脚本会自动评估：

- `clean`
- `alpha_noise` 的 `16/32/64/128/256/512/1024`
- `digit_noise` 的 `16/32/64/128/256/512/1024`

默认是**全量评估**，不截断测试集。

命令：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_prefix_noise_sweep.py \
  --weight full_sft_calc \
  --seed 42 \
  --device cuda:0
```

会生成：

- 逐条结果文件到 `results/`
- 汇总文件：
  - `results/prefix_noise_sweep_summary.jsonl`

### 打开 YaRN 的 Sweep

如果想测试通用 test-time RoPE 外推是否能缓解前缀噪声问题，可以直接在 sweep 中打开 `YaRN`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_prefix_noise_sweep.py \
  --weight full_sft_calc \
  --seed 42 \
  --device cuda:0 \
  --inference_rope_scaling
```

这会额外生成带 `_yarn` 后缀的逐条结果文件，例如：

- `results/full_sft_calc_clean_seed42_yarn.jsonl`
- `results/full_sft_calc_alpha_len1024_seed42_yarn.jsonl`
- `results/full_sft_calc_digit_len1024_seed42_yarn.jsonl`

## 3. 评估 Alpha Noise 测试集

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_alpha_noise_len1024_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --results_path ./results/full_sft_calc_alpha_len1024_seed42.jsonl
```

打开 YaRN：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_alpha_noise_len1024_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --inference_rope_scaling \
  --results_path ./results/full_sft_calc_alpha_len1024_seed42_yarn.jsonl
```

## 4. 评估 Digit Noise 测试集

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_digit_noise_len1024_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --results_path ./results/full_sft_calc_digit_len1024_seed42.jsonl
```

打开 YaRN：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_digit_noise_len1024_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --inference_rope_scaling \
  --results_path ./results/full_sft_calc_digit_len1024_seed42_yarn.jsonl
```

## 5. 评估 Clean 测试集

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --results_path ./results/full_sft_calc_clean.jsonl
```

打开 YaRN：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --inference_rope_scaling \
  --results_path ./results/full_sft_calc_clean_yarn.jsonl
```

## 6. 结果文件

建议保留三类结果文件，便于后续对比：

- `results/full_sft_calc_clean.jsonl`
- `results/full_sft_calc_alpha_len1024_seed42.jsonl`
- `results/full_sft_calc_digit_len1024_seed42.jsonl`

每条结果会记录：

- `prompt`
- `target`
- `prediction`
- `target_answer`
- `pred_answer`
- `is_correct`

## 7. 后续建议

建议先完成当前这一组长度：

- `16`
- `32`
- `64`
- `128`
- `256`
- `512`
- `1024`

这样可以画出：

- `clean`
- `alpha_noise`
- `digit_noise`

三条 accuracy-length 曲线，用来分析长前缀噪声对 `<calc>` 任务约束保持能力的影响。
