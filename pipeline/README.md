# Pipeline Scaffold

This folder contains orchestration only. It reuses the existing training and eval code:

- `trainer/train_pretrain.py`
- `trainer/train_full_sft.py`
- `scripts/eval_calc_dataset.py`

Each stage has its own entry file and its own YAML config:

- `run_pretrain.py` -> `stages/pretrain.yaml`
- `run_full_sft.py` -> `stages/full_sft.yaml`
- `run_calc_sft.py` -> `stages/calc_sft.yaml`
- `run_eval.py` -> `stages/eval.yaml`

## DDP support

The existing trainer code already supports DDP through `init_distributed_mode()` and `DistributedDataParallel`.
This pipeline enables multi-GPU runs by launching stages with:

```bash
python -m torch.distributed.run --standalone --nproc_per_node N ...
```

Set `runtime.nproc_per_node` in the specific stage YAML, for example `pipeline/stages/pretrain.yaml`.

## SwanLab

The training scripts use SwanLab through the existing `--use_wandb` flag:

```python
import swanlab as wandb
```

To enable it, set this in a training stage YAML:

```yaml
args:
  use_wandb: true
  wandb_project: MiniMind-3-Zero-Pretrain
```

## Run

Print commands only:

```powershell
.\.venv\Scripts\python.exe pipeline\run_pretrain.py --dry-run
.\.venv\Scripts\python.exe pipeline\run_full_sft.py --dry-run
.\.venv\Scripts\python.exe pipeline\run_calc_sft.py --dry-run
.\.venv\Scripts\python.exe pipeline\run_eval.py --dry-run
```

Run the full experiment:

```powershell
.\.venv\Scripts\python.exe pipeline\run_pretrain.py
.\.venv\Scripts\python.exe pipeline\run_full_sft.py
.\.venv\Scripts\python.exe pipeline\run_calc_sft.py
.\.venv\Scripts\python.exe pipeline\run_eval.py
```

The default data flow is:

- pretrain: `dataset/pretrain_t2t_mini.jsonl`
- full SFT: `dataset/sft_t2t_mini.jsonl`
- calc SFT: `dataset/sft_calc_addition_train.jsonl`
- eval: clean calc plus `alpha256` and `alpha512` noisy calc test sets

## Swap model structure

Change the root `config.yaml` model section:

```yaml
model:
  module: model.model_minimind
  config_class: MiniMindConfig
  model_class: MiniMindForCausalLM
```

For a new attention mechanism, create a new file such as:

```text
model/model_gated_attention.py
```

Then set:

```yaml
model:
  module: model.model_gated_attention
```

For attention experiments that need the same structure during pretrain, full SFT, calc SFT, and eval, set the same `model.module` and related switches before running all four stage files.
