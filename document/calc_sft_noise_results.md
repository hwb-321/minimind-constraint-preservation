# Calc SFT 前缀噪声实验结果

## 实验设置

- 基础任务：在 `<calc>` 标记内完成两数加法。
- 模型规模：MiniMind-3 dense 64M。
- 训练路线：MiniMind-3-Zero / 官方 `sft_zero_768.pth` -> calc SFT -> calc eval。
- 匹配方式：`answer_only`，只比较 `<calc>` 中等号右侧答案。
- 评测数据集：
  - `clean`: `dataset/sft_calc_addition_test.jsonl`
  - `alpha256`: `dataset/sft_calc_addition_test_alpha_noise_len256_seed42.jsonl`
  - `alpha512`: `dataset/sft_calc_addition_test_alpha_noise_len512_seed42.jsonl`

## 本地路线结果

注意：本组结果来自本地训练路线，其中包含门控注意力相关改动，因此不能和官方 `sft_zero_768.pth` 起点结果直接视为只差基座权重的对照。

| 模型 / Calc SFT 数据 | Clean | Alpha256 | Alpha512 |
|---|---:|---:|---:|
| 干净 calc SFT | 248/1000 = 24.8000% | 103/1000 = 10.3000% | 62/1000 = 6.2000% |
| Alpha256 噪声 calc SFT | 508/1000 = 50.8000% | 848/1000 = 84.8000% | 625/1000 = 62.5000% |

## 官方 `sft_zero_768.pth` 起点结果

| 模型 / Calc SFT 数据 | Clean | Alpha256 | Alpha512 |
|---|---:|---:|---:|
| 干净 calc SFT | 185/1000 = 18.5000% | 0/1000 = 0.0000% | 0/1000 = 0.0000% |
| Alpha256 噪声 calc SFT | 36/1000 = 3.6000% | 171/1000 = 17.1000% | 115/1000 = 11.5000% |

## 初步观察

在 calc SFT 训练数据中加入 alpha 前缀噪声后，模型在带噪声 calc 输入上的鲁棒性明显提升。提升在匹配训练噪声长度的 `alpha256` 评测集上最明显，同时也能迁移到更长的 `alpha512` 噪声场景。


## 为什么门控注意力可能有效

标准 softmax attention 会对所有可见 token 分配注意力权重。当前任务中，输入由两部分组成：

```text
随机字母噪声前缀 + <calc> 加法表达式
```

对于加法任务来说，真正有用的信息主要集中在 `<calc>...</calc>` 区间内，而前面的随机字母是任务无关噪声。普通 attention 虽然理论上可以学会忽略噪声，但在小模型和短训练条件下，噪声 token 仍可能分散注意力，导致模型在生成答案时受到无关上下文干扰。

门控注意力的核心思想是在 attention 输出之后加入一个 query-dependent / head-specific gate：

```text
o_t = g_t ⊙ softmax(q_t K^T)V
```

这里的 `g_t` 可以理解为每个位置、每个 head 对当前 attention 输出的“通过强度”。它并不是直接删除噪声 token，而是让模型多一个机制来调节 attention 输出对残差流的影响。

在前缀噪声 calc 场景中，门控注意力可能带来三类好处：

1. 对无关上下文更不敏感：当某些 head 被随机前缀干扰时，gate 可以降低这些 attention 输出进入后续层的强度。
2. 保留专门处理 `<calc>` 区间的 head：不同 head 可以形成分工，一部分 head 关注局部算式结构，一部分 head 处理格式边界，gate 可以让有用 head 在关键位置更强。
3. 提升噪声长度外推：训练时见过 `alpha256` 后，模型可能学到“前缀部分整体可信度较低，calc 区间更重要”的策略，因此在 `alpha512` 上也有一定迁移。

因此，门控注意力的预期贡献不是“把噪声去掉”，而是让模型在内部表示流中更灵活地控制 attention 信息的通过强度。这比测试时手工 block attention 更接近结构性改进，因为它在训练中由数据驱动学习，而不是人为指定哪些 token 可见。