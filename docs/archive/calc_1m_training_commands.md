# Calc 1M Training Commands

归档说明：本文件是 100w calc 数据训练阶段的历史命令备忘录，仅供实验溯源。当前主入口请优先参考项目根目录的 `README_PROJECT.md`。

以下命令都基于当前仓库中的 `100w` 条加法训练集：

- 训练集：`dataset/sft_calc_addition_train_1m.jsonl`
- 建议在目录：`trainer/` 下执行

## 1. 从现有 SFT 模型开始训练

基于已有的 `full_sft_768.pth` 继续在 100w 算术数据上做定向 SFT：

```bash
cd trainer
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 train_full_sft.py \
  --from_weight full_sft \
  --save_weight full_sft_calc_from_sft_1m \
  --data_path ../dataset/sft_calc_addition_train_1m.jsonl \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --learning_rate 5e-6 \
  --epochs 5 \
  --batch_size 16 \
  --use_wandb
```

输出权重：

- `out/full_sft_calc_from_sft_1m_768.pth`
- `checkpoints/full_sft_calc_from_sft_1m_768_resume.pth`

## 2. 从预训练模型开始训练

基于已有的 `pretrain_768.pth`，直接在 100w 算术数据上做 SFT：

```bash
cd trainer
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 train_full_sft.py \
  --from_weight pretrain \
  --save_weight calc_from_pretrain_1m \
  --data_path ../dataset/sft_calc_addition_train_1m.jsonl \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --learning_rate 5e-6 \
  --epochs 5 \
  --batch_size 16 \
  --use_wandb
```

输出权重：

- `out/calc_from_pretrain_1m_768.pth`
- `checkpoints/calc_from_pretrain_1m_768_resume.pth`

## 3. 从头开始训练

不加载已有权重，直接在 100w 算术数据上从头开始训练：

```bash
cd trainer
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 train_full_sft.py \
  --from_weight none \
  --save_weight calc_from_scratch_1m \
  --data_path ../dataset/sft_calc_addition_train_1m.jsonl \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --learning_rate 5e-6 \
  --epochs 5 \
  --batch_size 16 \
  --use_wandb
```

输出权重：

- `out/calc_from_scratch_1m_768.pth`
- `checkpoints/calc_from_scratch_1m_768_resume.pth`

## 建议

为了让三组实验更可比，建议：

- 保持相同的 `epochs`
- 保持相同的 `learning_rate`
- 保持相同的 `batch_size`
- 统一使用同一个测试集评估：`dataset/sft_calc_addition_test_10k.jsonl`

对应评估命令示例：

```bash
python scripts/eval_calc_dataset.py \
  --weight full_sft_calc_from_sft_1m \
  --data_path ./dataset/sft_calc_addition_test_10k.jsonl \
  --match_mode answer_only \
  --device cuda
```

## 4. 3 Epoch + 每个 Epoch 保存 + 自动评估

下面这组命令使用已经修改过的 `train_full_sft.py`，支持：

- 固定训练 `3` 个 epoch
- 每个 epoch 结束后额外保存一份 `epoch` 后缀权重
- 每个 epoch 结束后自动在测试集上评估准确率
- 每个 epoch 的逐条评测结果保存到 `results/`

### 4.1 从现有 SFT 模型开始

```bash
cd trainer
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 train_full_sft.py \
  --from_weight full_sft \
  --save_weight full_sft_calc_from_sft_1m \
  --data_path ../dataset/sft_calc_addition_train_1m.jsonl \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --learning_rate 5e-6 \
  --epochs 3 \
  --batch_size 16 \
  --use_wandb \
  --save_each_epoch 1 \
  --eval_data_path ../dataset/sft_calc_addition_test_10k.jsonl \
  --eval_match_mode answer_only \
  --eval_results_dir ../results
```

### 4.2 从预训练模型开始

```bash
cd trainer
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 train_full_sft.py \
  --from_weight pretrain \
  --save_weight calc_from_pretrain_1m \
  --data_path ../dataset/sft_calc_addition_train_1m.jsonl \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --learning_rate 5e-6 \
  --epochs 1 \
  --batch_size 16 \
  --use_wandb \
  --save_each_epoch 1 \
  --eval_data_path ../dataset/sft_calc_addition_test_10k.jsonl \
  --eval_match_mode answer_only \
  --eval_results_dir ../results
```

### 4.3 从头开始训练

```bash
cd trainer
CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 --master_port=29501 train_full_sft.py \
  --from_weight none \
  --save_weight calc_from_scratch_1m \
  --data_path ../dataset/sft_calc_addition_train_1m.jsonl \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --learning_rate 5e-6 \
  --epochs 3 \
  --batch_size 16 \
  --use_wandb \
  --save_each_epoch 1 \
  --eval_data_path ../dataset/sft_calc_addition_test_10k.jsonl \
  --eval_match_mode answer_only \
  --eval_results_dir ../results
```

### 4.4 产物说明

以 `--save_weight full_sft_calc_from_sft_1m` 为例，训练过程中会得到：

- 最新主权重：`out/full_sft_calc_from_sft_1m_768.pth`
- 每个 epoch 的权重：
  - `out/full_sft_calc_from_sft_1m_epoch1_768.pth`
  - `out/full_sft_calc_from_sft_1m_epoch2_768.pth`
  - `out/full_sft_calc_from_sft_1m_epoch3_768.pth`
- 每个 epoch 的逐条评测结果：
  - `results/full_sft_calc_from_sft_1m_epoch1.jsonl`
  - `results/full_sft_calc_from_sft_1m_epoch2.jsonl`
  - `results/full_sft_calc_from_sft_1m_epoch3.jsonl`
