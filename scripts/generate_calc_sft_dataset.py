import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple


def spaced_digits(number: int) -> str:
    return " ".join(str(number))


def build_sample(a: int, b: int) -> dict:
    result = a + b
    prompt = f"<calc>\n{spaced_digits(a)} + {spaced_digits(b)} =\n</calc>"
    answer = f"<calc>\n{spaced_digits(a)} + {spaced_digits(b)} = {spaced_digits(result)}\n</calc>"
    return {
        "conversations": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an SFT dataset for positive integer addition in <calc> format."
    )
    parser.add_argument("--a-min", type=int, required=True, help="Inclusive minimum for the first addend.")
    parser.add_argument("--a-max", type=int, required=True, help="Inclusive maximum for the first addend.")
    parser.add_argument("--b-min", type=int, required=True, help="Inclusive minimum for the second addend.")
    parser.add_argument("--b-max", type=int, required=True, help="Inclusive maximum for the second addend.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples to generate when not enumerating the full Cartesian product.",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
        help="`sample` draws random pairs with replacement; `full` enumerates all pairs in range order.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in sample mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sft_calc_addition.jsonl",
        help="Train output filename or path. Relative paths are resolved under dataset/.",
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default="sft_calc_addition_test.jsonl",
        help="Test output filename or path. Relative paths are resolved under dataset/.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of unique problems reserved for the test split. Ignored when --test-count is set.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="Number of unique problems reserved for the test split.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.a_min <= 0 or args.b_min <= 0:
        raise ValueError("Only positive integers are supported, so both minima must be > 0.")
    if args.a_min > args.a_max:
        raise ValueError("--a-min must be <= --a-max.")
    if args.b_min > args.b_max:
        raise ValueError("--b-min must be <= --b-max.")
    if args.mode == "sample" and args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0 in sample mode.")
    if args.test_count is not None and args.test_count < 0:
        raise ValueError("--test-count must be >= 0.")
    if not 0.0 <= args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be in [0.0, 1.0).")


def resolve_output_path(output_arg: str) -> Path:
    output_path = Path(output_arg)
    if not output_path.is_absolute():
        repo_root = Path(__file__).resolve().parent.parent
        output_path = repo_root / "dataset" / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def collect_problem_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    if args.mode == "full":
        pairs: List[Tuple[int, int]] = []
        for a in range(args.a_min, args.a_max + 1):
            for b in range(args.b_min, args.b_max + 1):
                pairs.append((a, b))
        return pairs

    rng = random.Random(args.seed)
    pairs = []
    for _ in range(args.num_samples):
        a = rng.randint(args.a_min, args.a_max)
        b = rng.randint(args.b_min, args.b_max)
        pairs.append((a, b))
    return pairs


def split_pairs(args: argparse.Namespace, pairs: List[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    unique_pairs = sorted(set(pairs))
    if not unique_pairs:
        return [], []

    rng = random.Random(args.seed)
    rng.shuffle(unique_pairs)

    if args.test_count is not None:
        test_size = min(args.test_count, len(unique_pairs))
    else:
        test_size = int(len(unique_pairs) * args.test_ratio)

    test_pairs = set(unique_pairs[:test_size])
    train_pairs = [pair for pair in pairs if pair not in test_pairs]
    test_pairs_list = [pair for pair in pairs if pair in test_pairs]
    return train_pairs, test_pairs_list


def main() -> None:
    args = parse_args()
    validate_args(args)
    train_output_path = resolve_output_path(args.output)
    test_output_path = resolve_output_path(args.test_output)

    pairs = collect_problem_pairs(args)
    train_pairs, test_pairs = split_pairs(args, pairs)

    train_count = 0
    with train_output_path.open("w", encoding="utf-8") as f:
        for a, b in train_pairs:
            sample = build_sample(a, b)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            train_count += 1

    test_count = 0
    with test_output_path.open("w", encoding="utf-8") as f:
        for a, b in test_pairs:
            sample = build_sample(a, b)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            test_count += 1

    print(f"Wrote {train_count} training samples to {train_output_path}")
    print(f"Wrote {test_count} test samples to {test_output_path}")
    print(f"Unique train problems: {len(set(train_pairs))}")
    print(f"Unique test problems: {len(set(test_pairs))}")


if __name__ == "__main__":
    main()
