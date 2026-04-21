# Targeted Interventions

这个目录用于放置与当前 `<calc>` 长前缀噪声研究相关的 test-time 方法原型实现。

当前保留一类基础组件：

1. `Mask Blocking`
   对 `<calc>` 任务段之后的 attention 可见性做局部阻断，只允许其看见：
   - 保留的前缀 token
   - 从 `<calc>` 开始到当前 token 的任务后缀

## 文件说明

- [targeted_interventions.py](/data/chenjingdong/ala/minimind/methods/targeted_interventions.py)
  提供通用的 span 检测和 attention blocking mask 构造函数。

- [demo_targeted_interventions.py](/data/chenjingdong/ala/minimind/methods/demo_targeted_interventions.py)
  一个最小示例，用于展示如何识别 `<calc>` span，并构造 blocking mask。

## 当前状态

当前保留的 attention block 已经接入评测脚本：

- 已经可以在 token 级别识别 `<calc>` span
- 已经可以构造 attention mask
- `scripts/eval_calc_dataset.py` 可通过 `--targeted_attention_mode block` 调用
