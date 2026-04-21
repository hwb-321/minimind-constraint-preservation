import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from methods.targeted_interventions import (
    build_blocking_attention_mask,
    find_calc_span,
)
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calc-style SFT datasets with exact-match accuracy.")
    parser.add_argument("--weight", type=str, default="full_sft_calc", help="Weight prefix under out/, e.g. full_sft_calc")
    parser.add_argument("--hidden_size", type=int, default=768, help="Model hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="Whether to load MoE weights")
    parser.add_argument(
        "--inference_rope_scaling",
        action="store_true",
        help="Enable YaRN-style RoPE scaling at inference time",
    )
    parser.add_argument("--yarn_original_max_position_embeddings", type=int, default=2048, help="YaRN original max position embeddings")
    parser.add_argument("--yarn_factor", type=float, default=16.0, help="YaRN scaling factor")
    parser.add_argument("--yarn_beta_fast", type=float, default=32.0, help="YaRN beta_fast")
    parser.add_argument("--yarn_beta_slow", type=float, default=1.0, help="YaRN beta_slow")
    parser.add_argument("--yarn_attention_factor", type=float, default=1.0, help="YaRN attention factor")
    parser.add_argument("--tokenizer_path", type=str, default="./model", help="Tokenizer directory")
    parser.add_argument("--save_dir", type=str, default="./out", help="Directory containing native torch checkpoints")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/sft_calc_addition_test.jsonl",
        help="Path to the calc test jsonl file",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum generated tokens per sample")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on the number of evaluated examples")
    parser.add_argument(
        "--targeted_attention_mode",
        type=str,
        default="none",
        choices=["none", "block"],
        help="Optional test-time attention intervention for the <calc> span",
    )
    parser.add_argument(
        "--attention_keep_prefix",
        type=int,
        default=0,
        help="When targeted_attention_mode=block, keep the first N prefix tokens visible to <calc> and generated tokens",
    )
    parser.add_argument(
        "--match_mode",
        type=str,
        default="answer_only",
        choices=["full", "answer_only"],
        help="Compare either the full assistant output or only the RHS after '=' inside <calc>",
    )
    parser.add_argument("--show_examples", type=int, default=5, help="How many mismatched examples to print")
    parser.add_argument(
        "--results_path",
        type=str,
        default="",
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
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe),
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
    model = MiniMindForCausalLM(config)
    moe_suffix = "_moe" if args.use_moe else ""
    ckpt_path = Path(args.save_dir) / f"{args.weight}_{args.hidden_size}{moe_suffix}.pth"
    state_dict = torch.load(ckpt_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
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


def main() -> None:
    args = parse_args()
    model, tokenizer, ckpt_path = build_model(args)
    data_path = Path(args.data_path)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Evaluating dataset: {data_path}")
    print(f"Match mode: {args.match_mode}")
    print(f"YaRN enabled: {args.inference_rope_scaling}")
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
        with data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if args.limit and total >= args.limit:
                    break

                sample = json.loads(line)
                prompt = sample["conversations"][0]["content"]
                target = sample["conversations"][1]["content"]

                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(text, return_tensors="pt").to(args.device)

                calc_span = None
                row_attention_mask = None
                if args.targeted_attention_mode != "none":
                    raw_input_ids = inputs["input_ids"][0].tolist()
                    row_attention_mask, attn_calc_span = build_targeted_attention_mask(tokenizer, raw_input_ids, args)
                    if calc_span is None:
                        calc_span = attn_calc_span
                    if row_attention_mask is not None:
                        row_attention_mask = row_attention_mask.unsqueeze(0).to(args.device)

                with torch.no_grad():
                    generated = model.generate(
                        inputs=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        custom_attention_mask=row_attention_mask,
                        blocking_calc_start=None if calc_span is None else calc_span.start,
                        blocking_keep_prefix=args.attention_keep_prefix,
                        max_new_tokens=args.max_new_tokens,
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
                is_correct = compare_prediction(prediction, target, args.match_mode)
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
                        "match_mode": args.match_mode,
                        "weight": args.weight,
                        "data_path": str(data_path),
                        "targeted_attention_mode": args.targeted_attention_mode,
                        "calc_span": None if calc_span is None else {"start": calc_span.start, "end": calc_span.end},
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
    finally:
        if results_file is not None:
            results_file.close()

    accuracy = correct / total if total else 0.0
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {accuracy:.4%}")


if __name__ == "__main__":
    main()
