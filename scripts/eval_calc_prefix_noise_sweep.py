import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_LENGTHS = [16, 32, 64, 128, 256, 512, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate calc prefix-noise datasets and save a summary jsonl."
    )
    parser.add_argument("--weight", type=str, default="full_sft_calc", help="Weight prefix under out/")
    parser.add_argument("--seed", type=int, required=True, help="Seed used in dataset filenames")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="Prefix noise lengths in characters",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device passed to eval script")
    parser.add_argument(
        "--inference_rope_scaling",
        action="store_true",
        help="Enable YaRN-style RoPE scaling during evaluation",
    )
    parser.add_argument("--yarn_original_max_position_embeddings", type=int, default=2048, help="YaRN original max position embeddings")
    parser.add_argument("--yarn_factor", type=float, default=16.0, help="YaRN scaling factor")
    parser.add_argument("--yarn_beta_fast", type=float, default=32.0, help="YaRN beta_fast")
    parser.add_argument("--yarn_beta_slow", type=float, default=1.0, help="YaRN beta_slow")
    parser.add_argument("--yarn_attention_factor", type=float, default=1.0, help="YaRN attention factor")
    parser.add_argument("--clean-data", type=str, default="dataset/sft_calc_addition_test.jsonl", help="Clean test set path")
    parser.add_argument("--match_mode", type=str, default="answer_only", choices=["full", "answer_only"], help="Matching mode")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens for generation")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on evaluation examples")
    parser.add_argument("--show_examples", type=int, default=0, help="How many mismatches to print per run")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for per-example jsonl outputs")
    parser.add_argument(
        "--summary_path",
        type=str,
        default="results/prefix_noise_sweep_summary.jsonl",
        help="Path to save one-line-per-run summary records",
    )
    return parser.parse_args()


def run_eval(script_path: Path, repo_root: Path, args: argparse.Namespace, data_path: str, results_path: str):
    cmd = [
        sys.executable,
        str(script_path),
        "--weight",
        args.weight,
        "--data_path",
        data_path,
        "--match_mode",
        args.match_mode,
        "--device",
        args.device,
        "--inference_rope_scaling" if args.inference_rope_scaling else "",
        "--yarn_original_max_position_embeddings",
        str(args.yarn_original_max_position_embeddings),
        "--yarn_factor",
        str(args.yarn_factor),
        "--yarn_beta_fast",
        str(args.yarn_beta_fast),
        "--yarn_beta_slow",
        str(args.yarn_beta_slow),
        "--yarn_attention_factor",
        str(args.yarn_attention_factor),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--show_examples",
        str(args.show_examples),
        "--results_path",
        results_path,
    ]
    cmd = [item for item in cmd if item != ""]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=True)
    stdout = proc.stdout
    accuracy_line = ""
    for line in stdout.splitlines():
        if line.startswith("Accuracy:"):
            accuracy_line = line
    if not accuracy_line:
        raise RuntimeError(f"Failed to parse accuracy from output:\n{stdout}")

    prefix = "Accuracy:"
    payload = accuracy_line[len(prefix):].strip()
    counts, ratio = [part.strip() for part in payload.split("=", 1)]
    correct_str, total_str = counts.split("/", 1)
    return {
        "correct": int(correct_str),
        "total": int(total_str),
        "accuracy": float(ratio.rstrip("%")) / 100.0,
        "stdout": stdout,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    eval_script = Path(__file__).resolve().parent / "eval_calc_dataset.py"

    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = repo_root / summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    clean_results = results_dir / f"{args.weight}_clean_seed{args.seed}.jsonl"
    if args.inference_rope_scaling:
        clean_results = results_dir / f"{args.weight}_clean_seed{args.seed}_yarn.jsonl"
    clean_eval = run_eval(
        eval_script,
        repo_root,
        args,
        args.clean_data,
        str(clean_results),
    )

    records = [{
        "weight": args.weight,
        "noise_type": "clean",
        "length": 0,
        "seed": args.seed,
        "inference_rope_scaling": args.inference_rope_scaling,
        "yarn_original_max_position_embeddings": args.yarn_original_max_position_embeddings,
        "yarn_factor": args.yarn_factor,
        "yarn_beta_fast": args.yarn_beta_fast,
        "yarn_beta_slow": args.yarn_beta_slow,
        "yarn_attention_factor": args.yarn_attention_factor,
        "data_path": args.clean_data,
        "results_path": str(clean_results),
        "correct": clean_eval["correct"],
        "total": clean_eval["total"],
        "accuracy": clean_eval["accuracy"],
    }]

    for length in args.lengths:
        for noise_type in ("alpha", "digit"):
            data_path = f"dataset/sft_calc_addition_test_{noise_type}_noise_len{length}_seed{args.seed}.jsonl"
            results_path = results_dir / f"{args.weight}_{noise_type}_len{length}_seed{args.seed}.jsonl"
            if args.inference_rope_scaling:
                results_path = results_dir / f"{args.weight}_{noise_type}_len{length}_seed{args.seed}_yarn.jsonl"
            print(f"Evaluating {noise_type} noise, length={length}")
            eval_res = run_eval(
                eval_script,
                repo_root,
                args,
                data_path,
                str(results_path),
            )
            records.append({
                "weight": args.weight,
                "noise_type": noise_type,
                "length": length,
                "seed": args.seed,
                "inference_rope_scaling": args.inference_rope_scaling,
                "yarn_original_max_position_embeddings": args.yarn_original_max_position_embeddings,
                "yarn_factor": args.yarn_factor,
                "yarn_beta_fast": args.yarn_beta_fast,
                "yarn_beta_slow": args.yarn_beta_slow,
                "yarn_attention_factor": args.yarn_attention_factor,
                "data_path": data_path,
                "results_path": str(results_path),
                "correct": eval_res["correct"],
                "total": eval_res["total"],
                "accuracy": eval_res["accuracy"],
            })

    with summary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved summary to {summary_path}")
    for record in records:
        print(
            f"{record['noise_type']:>5} | len={record['length']:>4} | "
            f"acc={record['correct']}/{record['total']}={record['accuracy']:.4%}"
        )


if __name__ == "__main__":
    main()
