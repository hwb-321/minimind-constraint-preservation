import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config_utils import config_bool, config_get, load_model_classes, load_project_config, project_path
from methods.targeted_interventions import (
    build_blocking_attention_mask,
    find_calc_span,
)


def parse_args() -> argparse.Namespace:
    project_config = load_project_config()
    default_eval_device = config_get(project_config, "eval.device", "auto")
    if default_eval_device == "auto":
        default_eval_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="Evaluate calc-style SFT datasets with exact-match accuracy.")
    parser.add_argument("--weight", type=str, default=config_get(project_config, "eval.weight", config_get(project_config, "model.weight", "full_sft_calc")), help="Weight prefix, e.g. full_sft_calc")
    parser.add_argument("--hidden_size", type=int, default=config_get(project_config, "model.hidden_size", 768), help="Model hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=config_get(project_config, "model.num_hidden_layers", 8), help="Number of transformer layers")
    parser.add_argument("--num_attention_heads", type=int, default=config_get(project_config, "model.num_attention_heads", 8), help="Number of attention heads")
    parser.add_argument("--num_key_value_heads", type=int, default=config_get(project_config, "model.num_key_value_heads", 4), help="Number of key/value heads")
    parser.add_argument("--intermediate_size", type=int, default=config_get(project_config, "model.intermediate_size", None), help="MLP intermediate size")
    parser.add_argument("--rms_norm_eps", type=float, default=config_get(project_config, "model.rms_norm_eps", 1e-6), help="RMSNorm epsilon")
    parser.add_argument("--use_qk_norm", action="store_true", default=config_bool(project_config, "model.use_qk_norm", True), help="Enable Q/K RMSNorm in attention")
    parser.add_argument("--use_moe", type=int, default=int(config_bool(project_config, "model.use_moe", False)), choices=[0, 1], help="Whether to load MoE weights")
    parser.add_argument("--use_attention_gate", action="store_true", default=config_bool(project_config, "attention_gate.enabled", False), help="Enable gated softmax attention from config.yaml")
    parser.add_argument("--attention_gate_scale", type=float, default=config_get(project_config, "attention_gate.scale", 2.0), help="Scale applied to sigmoid attention gate")
    parser.add_argument(
        "--inference_rope_scaling",
        action="store_true",
        default=config_bool(project_config, "yarn.inference_rope_scaling", False),
        help="Enable YaRN-style RoPE scaling at inference time",
    )
    parser.add_argument("--yarn_original_max_position_embeddings", type=int, default=config_get(project_config, "yarn.original_max_position_embeddings", 2048), help="YaRN original max position embeddings")
    parser.add_argument("--yarn_factor", type=float, default=config_get(project_config, "yarn.factor", 16.0), help="YaRN scaling factor")
    parser.add_argument("--yarn_beta_fast", type=float, default=config_get(project_config, "yarn.beta_fast", 32.0), help="YaRN beta_fast")
    parser.add_argument("--yarn_beta_slow", type=float, default=config_get(project_config, "yarn.beta_slow", 1.0), help="YaRN beta_slow")
    parser.add_argument("--yarn_attention_factor", type=float, default=config_get(project_config, "yarn.attention_factor", 1.0), help="YaRN attention factor")
    parser.add_argument("--tokenizer_path", type=str, default=project_path(config_get(project_config, "model.tokenizer_path", "./model")), help="Tokenizer directory")
    parser.add_argument("--save_dir", type=str, default=project_path(config_get(project_config, "model.save_dir", "./out")), help="Directory containing native torch checkpoints")
    parser.add_argument(
        "--data_path",
        type=str,
        default=project_path(config_get(project_config, "eval.data_path", "./dataset/sft_calc_addition_test.jsonl")),
        help="Path to the calc test jsonl file",
    )
    parser.add_argument("--device", type=str, default=default_eval_device, help="Inference device")
    parser.add_argument("--max_new_tokens", type=int, default=config_get(project_config, "eval.max_new_tokens", 64), help="Maximum generated tokens per sample")
    parser.add_argument("--batch_size", type=int, default=config_get(project_config, "eval.batch_size", 16), help="Batch size for evaluation generation")
    parser.add_argument("--progress_interval", type=int, default=config_get(project_config, "eval.progress_interval", 10), help="Print progress every N evaluated samples")
    parser.add_argument("--limit", type=int, default=config_get(project_config, "eval.limit", 0), help="Optional cap on the number of evaluated examples")
    parser.add_argument(
        "--targeted_attention_mode",
        type=str,
        default=config_get(project_config, "eval.targeted_attention_mode", "none"),
        choices=["none", "block"],
        help="Optional test-time attention intervention for the <calc> span",
    )
    parser.add_argument(
        "--attention_keep_prefix",
        type=int,
        default=config_get(project_config, "eval.attention_keep_prefix", 0),
        help="When targeted_attention_mode=block, keep the first N prefix tokens visible to <calc> and generated tokens",
    )
    parser.add_argument(
        "--match_mode",
        type=str,
        default=config_get(project_config, "eval.match_mode", "answer_only"),
        choices=["full", "answer_only"],
        help="Compare either the full assistant output or only the RHS after '=' inside <calc>",
    )
    parser.add_argument("--show_examples", type=int, default=config_get(project_config, "eval.show_examples", 5), help="How many mismatched examples to print")
    parser.add_argument(
        "--results_path",
        type=str,
        default=config_get(project_config, "eval.results_path", ""),
        help="Optional jsonl path for saving per-example evaluation results",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_calc_answer(text: str) -> str:
    match = re.search(r"<calc>\s*(.*?)\s*</calc>", text, flags=re.DOTALL)
    body = match.group(1) if match else text
    if "=" in body:
        body = body.split("=", 1)[1]
    return normalize_text(body)


def build_model(args: argparse.Namespace):
    project_config = load_project_config()
    ConfigClass, ModelClass = load_model_classes(project_config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = ConfigClass(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        rms_norm_eps=args.rms_norm_eps,
        use_qk_norm=args.use_qk_norm,
        use_moe=bool(args.use_moe),
        use_attention_gate=args.use_attention_gate,
        attention_gate_scale=args.attention_gate_scale,
        inference_rope_scaling=args.inference_rope_scaling,
    )
    if args.inference_rope_scaling and config.rope_scaling is not None:
        config.rope_scaling.update({
            "original_max_position_embeddings": args.yarn_original_max_position_embeddings,
            "factor": args.yarn_factor,
            "beta_fast": args.yarn_beta_fast,
            "beta_slow": args.yarn_beta_slow,
            "attention_factor": args.yarn_attention_factor,
        })
    model = ModelClass(config)
    moe_suffix = "_moe" if args.use_moe else ""
    ckpt_path = Path(args.save_dir) / f"{args.weight}_{args.hidden_size}{moe_suffix}.pth"
    state_dict = torch.load(ckpt_path, map_location=args.device)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Newly initialized params: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Unused checkpoint params: {load_result.unexpected_keys}")
    model = model.half().eval().to(args.device)
    return model, tokenizer, ckpt_path


def compare_prediction(prediction: str, target: str, match_mode: str) -> bool:
    if match_mode == "full":
        return normalize_text(prediction) == normalize_text(target)
    return extract_calc_answer(prediction) == extract_calc_answer(target)


def build_targeted_attention_mask(tokenizer, input_ids, args):
    if args.targeted_attention_mode == "none":
        return None, None

    calc_start_ids = tokenizer("<calc>", add_special_tokens=False).input_ids
    calc_end_ids = tokenizer("</calc>", add_special_tokens=False).input_ids
    calc_span = find_calc_span(input_ids, calc_start_ids, calc_end_ids)
    attention_mask = build_blocking_attention_mask(
        seq_len=len(input_ids),
        calc_span=calc_span,
        keep_prefix=args.attention_keep_prefix,
    )
    return attention_mask, calc_span


def load_eval_samples(data_path: Path, limit: int) -> list[dict]:
    samples = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit and len(samples) >= limit:
                break
            sample = json.loads(line)
            samples.append({
                "prompt": sample["conversations"][0]["content"],
                "target": sample["conversations"][1]["content"],
            })
    return samples


def iter_batches(samples: list[dict], batch_size: int):
    batch_size = max(1, batch_size)
    for start in range(0, len(samples), batch_size):
        yield start, samples[start:start + batch_size]


def generate_batch_predictions(model, tokenizer, texts: list[str], args: argparse.Namespace) -> list[str]:
    encoded_rows = [tokenizer(text, add_special_tokens=True).input_ids for text in texts]
    predictions = [""] * len(texts)
    length_groups: dict[int, list[int]] = {}
    for row_idx, input_ids in enumerate(encoded_rows):
        length_groups.setdefault(len(input_ids), []).append(row_idx)

    for prompt_length, row_indices in length_groups.items():
        input_ids = torch.tensor(
            [encoded_rows[row_idx] for row_idx in row_indices],
            dtype=torch.long,
            device=args.device,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            generated = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
        )
        for row_idx, prediction in zip(row_indices, decoded):
            predictions[row_idx] = prediction

    return predictions


def main() -> None:
    args = parse_args()
    model, tokenizer, ckpt_path = build_model(args)
    data_path = Path(args.data_path)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Evaluating dataset: {data_path}")
    print(f"Match mode: {args.match_mode}")
    print(f"YaRN enabled: {args.inference_rope_scaling}")
    print(f"Attention gate enabled: {args.use_attention_gate}")
    if args.use_attention_gate:
        print(f"Attention gate scale: {args.attention_gate_scale}")
    print(f"Targeted attention mode: {args.targeted_attention_mode}")
    if args.targeted_attention_mode == "block":
        print(f"Attention keep_prefix: {args.attention_keep_prefix}")
    if args.inference_rope_scaling:
        print(
            "YaRN params: "
            f"orig={args.yarn_original_max_position_embeddings}, "
            f"factor={args.yarn_factor}, "
            f"beta_fast={args.yarn_beta_fast}, "
            f"beta_slow={args.yarn_beta_slow}, "
            f"attn_factor={args.yarn_attention_factor}"
        )
    print(f"Eval batch size: {args.batch_size}")
    print(f"Progress interval: {args.progress_interval}")

    total = 0
    correct = 0
    shown = 0
    results_file = None

    if args.results_path:
        results_path = Path(args.results_path)
        if not results_path.is_absolute():
            results_path = Path.cwd() / results_path
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_file = results_path.open("w", encoding="utf-8")
        print(f"Saving per-example results to: {results_path}")

    try:
        samples = load_eval_samples(data_path, args.limit)
        print(f"Total eval samples: {len(samples)}")
        for batch_start, batch in iter_batches(samples, args.batch_size):
            prompts = [sample["prompt"] for sample in batch]
            targets = [sample["target"] for sample in batch]
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
            predictions = generate_batch_predictions(model, tokenizer, texts, args)

            for offset, (prompt, target, prediction) in enumerate(zip(prompts, targets, predictions), start=1):
                target_answer = extract_calc_answer(target)
                pred_answer = extract_calc_answer(prediction)
                is_correct = compare_prediction(prediction, target, args.match_mode)
                correct += int(is_correct)
                total += 1

                if results_file is not None:
                    record = {
                        "index": batch_start + offset,
                        "prompt": prompt,
                        "target": target,
                        "prediction": prediction,
                        "target_answer": target_answer,
                        "pred_answer": pred_answer,
                        "is_correct": is_correct,
                        "match_mode": args.match_mode,
                        "weight": args.weight,
                        "data_path": str(data_path),
                        "targeted_attention_mode": args.targeted_attention_mode,
                        "calc_span": None,
                    }
                    results_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if not is_correct and shown < args.show_examples:
                    print("-" * 80)
                    print(f"Example {total} mismatch")
                    print(f"Prompt : {prompt}")
                    print(f"Target : {target}")
                    print(f"Pred   : {prediction}")
                    if args.match_mode == "answer_only":
                        print(f"Target answer: {target_answer}")
                        print(f"Pred answer  : {pred_answer}")
                    shown += 1

            if total == len(samples) or (args.progress_interval and total % args.progress_interval == 0):
                running_acc = correct / total if total else 0.0
                print(f"[Eval Progress] {total}/{len(samples)} accuracy={correct}/{total}={running_acc:.4%}", flush=True)
    finally:
        if results_file is not None:
            results_file.close()

    accuracy = correct / total if total else 0.0
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {accuracy:.4%}")


if __name__ == "__main__":
    main()
