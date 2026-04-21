# Group Project: Constraint Preservation in Long Noisy Contexts

本仓库是在 MiniMind 上扩展出的课程小组项目，主要研究基于位置编码和注意力机制设计的方法，如何缓解长噪声上下文中的约束丢失问题。

当前实验以 `<calc>` 加法任务作为主要验证场景：当模型已经能在短上下文中完成局部计算约束时，我们考察长噪声上下文是否会破坏这种约束保持能力，并测试 RoPE/YaRN、attention blocking 等方法的影响。

本项目基于 MiniMind：

- Upstream: https://github.com/jingyaogong/minimind
- Original README: [README_MINIMIND.md](README_MINIMIND.md)
- Original English README: [README_en.md](README_en.md)
- License: [Apache License 2.0](LICENSE)

本文档说明当前工作目录里和约束保持实验直接相关的数据、训练、评测和 test-time attention 干预。所有命令默认从项目根目录执行，除非特别说明。

---

## 1. 项目目标

核心问题：

> 一个已经能在短上下文中完成简单加法的模型，如果在题目前加入长前缀噪声，是否还能只关注局部 `<calc>` 任务并输出正确答案？

干净样本示例：

```text
<calc>
1 7 + 2 5 =
</calc>
```

期望输出：

```text
<calc>
1 7 + 2 5 = 4 2
</calc>
```

前缀噪声样本会在 `<calc>` 前插入一段字符，例如：

```text
832 17 4092 55 8...
<calc>
1 7 + 2 5 =
</calc>
```

当前使用两类噪声：

- `alpha noise`：随机大小写字母，和加法任务表面形式差异较大。
- `digit noise`：随机数字和空格，和加法任务表面形式相似，更容易干扰注意力。

当前比较的方法：

- `baseline`：不做额外处理。
- `YaRN`：推理阶段启用 RoPE 外推。
- `attention block`：在 test time 屏蔽 `<calc>` 后续 token 对中间噪声前缀的注意力。

---

## 2. 目录结构

```text
.
├── dataset/
│   ├── lm_dataset.py
│   ├── sft_calc_addition_train.jsonl
│   ├── sft_calc_addition_test.jsonl
│   └── sft_calc_addition_test_*_noise_len*_seed*.jsonl
├── docs/
│   └── archive/
│       └── calc_prefix_noise_experiments.md
├── methods/
│   ├── README.md
│   ├── targeted_interventions.py
│   └── demo_targeted_interventions.py
├── model/
│   ├── model_minimind.py
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── out/
│   └── *.pth
├── result_all/
├── result_all_attention_fix/
├── scripts/
│   ├── generate_calc_sft_dataset.py
│   ├── generate_calc_prefix_noise_dataset.py
│   ├── generate_calc_prefix_noise_sweep.py
│   ├── eval_calc_dataset.py
│   ├── eval_calc_prefix_noise_sweep.py
│   ├── run_attention_block_keep4_9evals.sh
│   ├── read_jsonl_acc_dir.py
│   └── read_result_all_acc.py
├── trainer/
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   └── trainer_utils.py
└── README_PROJECT.md
```

关键文件说明：

- [model/model_minimind.py](model/model_minimind.py)：MiniMind 模型实现，包含 RoPE/YaRN 和自定义 attention mask 接口。
- [methods/targeted_interventions.py](methods/targeted_interventions.py)：定位 `<calc>` span，并构造 attention-block mask。
- [scripts/eval_calc_dataset.py](scripts/eval_calc_dataset.py)：单个数据集评测脚本，支持 baseline、YaRN 和 attention block。
- [scripts/run_attention_block_keep4_9evals.sh](scripts/run_attention_block_keep4_9evals.sh)：并行跑 9 组 attention-block 评测。

---

## 3. 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

依赖文件中没有固定安装 `torch`，如环境里没有 PyTorch，需要按机器 CUDA 版本单独安装。

检查项目能否导入：

```bash
python -c "from model.model_minimind import MiniMindConfig; print('ok')"
```

---

## 4. 模型权重

默认权重目录是 [out/](out)。评测脚本按下面的规则拼接权重路径：

```text
--save_dir ./out
--weight full_sft_calc
--hidden_size 768

=> ./out/full_sft_calc_768.pth
```

当前主实验模型：

```text
out/full_sft_calc_768.pth
```

文件名里的 `768` 表示 `hidden_size`，不等于训练上下文长度。

---

## 5. 数据格式

Calc SFT 数据采用 ChatML 风格 JSONL。每行一个样本：

```json
{
  "conversations": [
    {
      "role": "user",
      "content": "<calc>\n1 0 + 1 6 =\n</calc>"
    },
    {
      "role": "assistant",
      "content": "<calc>\n1 0 + 1 6 = 2 6\n</calc>"
    }
  ]
}
```

主训练集：

```text
dataset/sft_calc_addition_train.jsonl
```

主测试集：

```text
dataset/sft_calc_addition_test.jsonl
```

前缀噪声测试集命名规则：

```text
dataset/sft_calc_addition_test_{alpha|digit}_noise_len{L}_seed{S}.jsonl
```

例如：

```text
dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl
```

---

## 6. 生成数据

### 6.1 生成加法 SFT 数据

脚本：[scripts/generate_calc_sft_dataset.py](scripts/generate_calc_sft_dataset.py)

示例：生成两位数加法全集，并保留 1000 道唯一题目做测试。

```bash
python scripts/generate_calc_sft_dataset.py \
  --a-min 10 \
  --a-max 99 \
  --b-min 10 \
  --b-max 99 \
  --mode full \
  --test-count 1000 \
  --output sft_calc_addition_train.jsonl \
  --test-output sft_calc_addition_test.jsonl
```

相对输出路径会自动解析到 `dataset/` 下。脚本会保证 train/test 在具体 `(a,b)` 题目上不重叠。

### 6.2 生成单个长度的前缀噪声测试集

脚本：[scripts/generate_calc_prefix_noise_dataset.py](scripts/generate_calc_prefix_noise_dataset.py)

```bash
python scripts/generate_calc_prefix_noise_dataset.py \
  --input dataset/sft_calc_addition_test.jsonl \
  --length 256 \
  --seed 42
```

该命令会同时生成：

```text
dataset/sft_calc_addition_test_alpha_noise_len256_seed42.jsonl
dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl
```

### 6.3 批量生成多个噪声长度

脚本：[scripts/generate_calc_prefix_noise_sweep.py](scripts/generate_calc_prefix_noise_sweep.py)

```bash
python scripts/generate_calc_prefix_noise_sweep.py \
  --seed 42 \
  --lengths 16 32 64 128 256 512 1024
```

---

## 7. 训练

训练脚本位于 [trainer/](trainer)。脚本默认保存 `.pth` 到 `../out`，保存训练断点到 `../checkpoints`。

### 7.1 预训练

脚本：[trainer/train_pretrain.py](trainer/train_pretrain.py)

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node 4 \
  --master_port 29501 \
  train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain
```

默认关键参数：

- `--hidden_size 768`
- `--num_hidden_layers 8`
- `--max_seq_len 340`
- `--batch_size 32`
- `--accumulation_steps 8`

### 7.2 通用 Full SFT

脚本：[trainer/train_full_sft.py](trainer/train_full_sft.py)

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node 4 \
  --master_port 29501 \
  train_full_sft.py \
  --from_weight pretrain \
  --save_weight full_sft \
  --data_path ../dataset/sft_t2t_mini.jsonl
```

默认关键参数：

- `--hidden_size 768`
- `--num_hidden_layers 8`
- `--max_seq_len 768`
- `--batch_size 16`
- `--learning_rate 1e-5`

### 7.3 Calc SFT

从 `full_sft_768.pth` 继续训练 `<calc>` 加法模型：

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node 4 \
  --master_port 29501 \
  train_full_sft.py \
  --from_weight full_sft \
  --save_weight full_sft_calc \
  --data_path ../dataset/sft_calc_addition_train.jsonl \
  --learning_rate 5e-6 \
  --epochs 5 \
  --batch_size 16
```

如需每个 epoch 后立即在 calc 测试集上评测：

```bash
cd trainer

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node 4 \
  --master_port 29501 \
  train_full_sft.py \
  --from_weight full_sft \
  --save_weight full_sft_calc \
  --data_path ../dataset/sft_calc_addition_train.jsonl \
  --eval_data_path ../dataset/sft_calc_addition_test.jsonl \
  --eval_results_dir ../results/train_eval \
  --eval_match_mode answer_only
```

---

## 8. 评测

核心脚本：[scripts/eval_calc_dataset.py](scripts/eval_calc_dataset.py)

评测指标默认使用 `answer_only` exact match：只比较 `<calc>` 中等号右侧答案，而不是强制完整输出字符串完全一致。

### 8.1 Clean 测试

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --results_path ./results/full_sft_calc_clean.jsonl
```

### 8.2 前缀噪声测试

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --results_path ./results/full_sft_calc_digit_len256.jsonl
```

常用调试参数：

- `--limit 20`：只评测前 20 条样本。
- `--show_examples 5`：打印最多 5 个错误样例。
- `--max_new_tokens 64`：控制生成长度。

### 8.3 启用 YaRN

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --inference_rope_scaling \
  --yarn_original_max_position_embeddings 768 \
  --results_path ./results/full_sft_calc_digit_len256_yarn_orig768.jsonl
```

`--yarn_original_max_position_embeddings 768` 对应当前 Calc SFT 阶段常用的训练截断长度。注意这和模型实现里的结构上限不是同一个概念。

### 8.4 启用 Attention Block

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --targeted_attention_mode block \
  --attention_keep_prefix 4 \
  --results_path ./results/full_sft_calc_digit_len256_block_keep4.jsonl
```

参数含义：

- `--targeted_attention_mode block`：对 `<calc>` 任务段及其后续生成 token 施加局部可见性约束。
- `--attention_keep_prefix 4`：保留最前面的 4 个 chat-template token 可见。

在当前 tokenizer/chat template 下，clean 样本中 `<calc>` 前的固定模板前缀是：

```text
<|im_start|>user\n
```

因此 `keep_prefix=4` 表示保留必要模板信息，同时屏蔽中间噪声前缀。

---

## 9. 批量评测

### 9.1 Baseline/YaRN 前缀噪声 sweep

脚本：[scripts/eval_calc_prefix_noise_sweep.py](scripts/eval_calc_prefix_noise_sweep.py)

Baseline：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_prefix_noise_sweep.py \
  --weight full_sft_calc \
  --seed 42 \
  --lengths 16 64 256 1024 \
  --device cuda:0 \
  --results_dir results/baseline \
  --summary_path results/baseline_summary.jsonl
```

YaRN：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_prefix_noise_sweep.py \
  --weight full_sft_calc \
  --seed 42 \
  --lengths 16 64 256 1024 \
  --device cuda:0 \
  --inference_rope_scaling \
  --yarn_original_max_position_embeddings 768 \
  --results_dir results/yarn_orig768 \
  --summary_path results/yarn_orig768_summary.jsonl
```

### 9.2 Attention Block 9 组评测

脚本：[scripts/run_attention_block_keep4_9evals.sh](scripts/run_attention_block_keep4_9evals.sh)

```bash
bash scripts/run_attention_block_keep4_9evals.sh
```

该脚本默认使用 GPU 0、1、2 并行运行：

```text
clean
alpha_len16
digit_len16
alpha_len64
digit_len64
alpha_len256
digit_len256
alpha_len1024
digit_len1024
```

输出目录：

```text
result_all_attention_fix/block_keep4/
```

---

## 10. 结果汇总

### 10.1 统计任意结果目录中的 JSONL accuracy

脚本：[scripts/read_jsonl_acc_dir.py](scripts/read_jsonl_acc_dir.py)

```bash
python scripts/read_jsonl_acc_dir.py \
  --root ./result_all_attention_fix
```

保存汇总：

```bash
python scripts/read_jsonl_acc_dir.py \
  --root ./result_all_attention_fix \
  --save_path ./result_all_attention_fix/summary.jsonl
```

### 10.2 当前 attention block keep4 结果

当前 `result_all_attention_fix/block_keep4/` 中保存的结果为：

```text
clean          | acc=848/1000=84.8000%
alpha_len16    | acc=784/1000=78.4000%
digit_len16    | acc=793/1000=79.3000%
alpha_len64    | acc=693/1000=69.3000%
digit_len64    | acc=708/1000=70.8000%
alpha_len256   | acc=556/1000=55.6000%
digit_len256   | acc=573/1000=57.3000%
alpha_len1024  | acc=399/1000=39.9000%
digit_len1024  | acc=454/1000=45.4000%
```

### 10.3 读取旧版 result_all 结构

脚本：[scripts/read_result_all_acc.py](scripts/read_result_all_acc.py)

```bash
python scripts/read_result_all_acc.py \
  --root ./result_all
```

---

## 11. 概念说明

### 11.1 `max_seq_len`

`max_seq_len` 是训练数据被 tokenizer 编码后的最大 token 截断长度，不是字符数。

当前训练脚本默认值：

- `trainer/train_pretrain.py`：`--max_seq_len 340`
- `trainer/train_full_sft.py`：`--max_seq_len 768`

### 11.2 `max_position_embeddings`

[model/model_minimind.py](model/model_minimind.py) 中模型结构默认会构造更长的 RoPE 表：

```python
self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
```

这只是结构上限，不代表模型训练过 32768 token 的上下文。

### 11.3 `768`

`full_sft_calc_768.pth` 中的 `768` 是隐藏层维度 `hidden_size`，不是上下文长度。

### 11.4 `answer_only`

`--match_mode answer_only` 会从 `<calc>...</calc>` 内提取等号右侧内容做匹配。例如下面两个输出在 `answer_only` 下都算正确：

```text
<calc>
1 7 + 2 5 = 4 2
</calc>
```

```text
4 2
```

---

## 12. 推荐复现实验流程

1. 确认主模型存在：

```bash
ls out/full_sft_calc_768.pth
```

2. 生成需要的前缀噪声数据：

```bash
python scripts/generate_calc_prefix_noise_sweep.py \
  --seed 42 \
  --lengths 16 64 256 1024
```

3. 跑一个小样本 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --limit 20 \
  --show_examples 3 \
  --results_path ./results/smoke_digit_len256.jsonl
```

4. 跑 baseline 或 attention block：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_calc_dataset.py \
  --weight full_sft_calc \
  --data_path ./dataset/sft_calc_addition_test_digit_noise_len256_seed42.jsonl \
  --match_mode answer_only \
  --device cuda:0 \
  --targeted_attention_mode block \
  --attention_keep_prefix 4 \
  --results_path ./results/block_keep4_digit_len256.jsonl
```

5. 汇总结果：

```bash
python scripts/read_jsonl_acc_dir.py \
  --root ./results
```

---

## 13. 常见问题

### `--from_weight` 必须带值

错误写法：

```bash
python train_full_sft.py --from_weight
```

正确写法：

```bash
python train_full_sft.py --from_weight full_sft
```

### 从根目录和从 `trainer/` 目录运行时路径不同

从根目录运行：

```bash
torchrun trainer/train_full_sft.py \
  --data_path dataset/sft_calc_addition_train.jsonl
```

从 `trainer/` 目录运行：

```bash
cd trainer

torchrun train_full_sft.py \
  --data_path ../dataset/sft_calc_addition_train.jsonl
```

### 找不到权重

评测脚本默认寻找：

```text
./out/{weight}_{hidden_size}.pth
```

如果权重不在 `./out`，需要显式传：

```bash
--save_dir /path/to/out
```

如果 hidden size 不是 768，需要显式传：

```bash
--hidden_size <size>
```

### 多 GPU 脚本里的 `--device cuda:0` 是正常的

例如 `run_attention_block_keep4_9evals.sh` 会先设置：

```bash
CUDA_VISIBLE_DEVICES="$gpu"
```

此时 Python 进程内部看到的第一张卡就是 `cuda:0`。

---

## 14. 当前状态

当前实验分支已经具备：

- `<calc>` 加法数据生成。
- alpha/digit 前缀噪声数据生成。
- baseline、YaRN、attention block 三类评测入口。
- JSONL 逐样本结果保存。
- 目录级 accuracy 汇总脚本。

建议后续继续做：

- 不同 `attention_keep_prefix` 的系统对比。
- 错误样本聚类和可视化。
- 不同训练窗口、不同噪声长度的 sweep。
- 将 attention block 验证到非加法的局部约束任务上。
