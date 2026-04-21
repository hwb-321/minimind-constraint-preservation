import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
import re
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def normalize_text(text):
    return re.sub(r"\s+", " ", text.strip())


def extract_calc_answer(text):
    match = re.search(r"<calc>\s*(.*?)\s*</calc>", text, flags=re.DOTALL)
    body = match.group(1) if match else text
    if "=" in body:
        body = body.split("=", 1)[1]
    return normalize_text(body)


def compare_prediction(prediction, target, match_mode):
    if match_mode == "full":
        return normalize_text(prediction) == normalize_text(target)
    return extract_calc_answer(prediction) == extract_calc_answer(target)


def save_current_model(weight_name):
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{weight_name}_{lm_config.hidden_size}{moe_suffix}.pth'
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model = getattr(raw_model, '_orig_mod', raw_model)
    state_dict = raw_model.state_dict()
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    del state_dict
    return ckp


def evaluate_calc_dataset(weight_name, epoch_idx):
    if not args.eval_data_path or not is_main_process():
        return None

    model.eval()
    total = 0
    correct = 0
    shown = 0
    results_file = None
    results_path = None

    if args.eval_results_dir:
        os.makedirs(args.eval_results_dir, exist_ok=True)
        results_path = os.path.join(args.eval_results_dir, f"{weight_name}_epoch{epoch_idx + 1}.jsonl")
        results_file = open(results_path, "w", encoding="utf-8")

    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if args.eval_limit and total >= args.eval_limit:
                break

            sample = json.loads(line)
            prompt = sample["conversations"][0]["content"]
            target = sample["conversations"][1]["content"]
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(args.device)

            with torch.no_grad():
                generated = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=args.eval_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prediction = tokenizer.decode(
                generated[0][len(inputs["input_ids"][0]):],
                skip_special_tokens=True,
            )

            target_answer = extract_calc_answer(target)
            pred_answer = extract_calc_answer(prediction)
            is_correct = compare_prediction(prediction, target, args.eval_match_mode)
            correct += int(is_correct)
            total += 1

            if results_file is not None:
                record = {
                    "index": total,
                    "prompt": prompt,
                    "target": target,
                    "prediction": prediction,
                    "target_answer": target_answer,
                    "pred_answer": pred_answer,
                    "is_correct": is_correct,
                    "match_mode": args.eval_match_mode,
                    "weight": weight_name,
                    "epoch": epoch_idx + 1,
                    "data_path": args.eval_data_path,
                }
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if not is_correct and shown < args.eval_show_examples:
                Logger('-' * 100)
                Logger(f'[Eval Mismatch] epoch={epoch_idx + 1}, sample={total}')
                Logger(f'Prompt : {prompt}')
                Logger(f'Target : {target}')
                Logger(f'Pred   : {prediction}')
                if args.eval_match_mode == "answer_only":
                    Logger(f'Target answer: {target_answer}')
                    Logger(f'Pred answer  : {pred_answer}')
                shown += 1

    if results_file is not None:
        results_file.close()

    accuracy = correct / total if total else 0.0
    Logger(f'[Eval] Epoch {epoch_idx + 1}: accuracy={correct}/{total}={accuracy:.4%}')
    if results_path:
        Logger(f'[Eval] Saved per-example results to {results_path}')
    model.train()
    return accuracy


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    last_step = start_step
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            res = model(input_ids, labels=labels)
            loss = res.loss + res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            save_current_model(args.save_weight)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()

        del input_ids, labels, res, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--save_each_epoch", default=1, type=int, choices=[0, 1], help="是否在每个epoch结束时额外保存一个带epoch后缀的权重")
    parser.add_argument("--eval_data_path", type=str, default="", help="可选：每个epoch结束后评测的calc测试集路径")
    parser.add_argument("--eval_match_mode", type=str, default="answer_only", choices=["full", "answer_only"], help="calc评测匹配方式")
    parser.add_argument("--eval_max_new_tokens", type=int, default=64, help="calc评测最大生成token数")
    parser.add_argument("--eval_limit", type=int, default=0, help="可选：限制每轮评测样本数，0表示不限制")
    parser.add_argument("--eval_show_examples", type=int, default=5, help="每轮评测打印多少个错误样例")
    parser.add_argument("--eval_results_dir", type=str, default="../results", help="每轮评测逐条结果jsonl的输出目录")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型、数据、优化器 ==========
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)

        if is_main_process():
            epoch_weight = args.save_weight
            if args.save_each_epoch == 1:
                epoch_weight = f"{args.save_weight}_epoch{epoch + 1}"
                save_current_model(epoch_weight)
                Logger(f'Saved epoch checkpoint to {args.save_dir}/{epoch_weight}_{lm_config.hidden_size}{"_moe" if lm_config.use_moe else ""}.pth')

            eval_acc = evaluate_calc_dataset(epoch_weight, epoch)
            if wandb and eval_acc is not None:
                wandb.log({"eval_accuracy": eval_acc, "epoch": epoch + 1})

        if dist.is_initialized():
            dist.barrier()
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
