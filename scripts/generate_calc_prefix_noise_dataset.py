import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate prefix-noise calc test sets with fixed character length noise."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/sft_calc_addition_test.jsonl",
        help="Input calc test jsonl path",
    )
    parser.add_argument(
        "--length",
        type=int,
        required=True,
        help="Prefix noise length in characters",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used to generate noise",
    )
    parser.add_argument(
        "--alpha-output",
        type=str,
        default="",
        help="Output path for alphabetic noise dataset. Relative paths are resolved under dataset/.",
    )
    parser.add_argument(
        "--digit-output",
        type=str,
        default="",
        help="Output path for digit-and-space noise dataset. Relative paths are resolved under dataset/.",
    )
    return parser.parse_args()


def resolve_dataset_path(path_arg: str, repo_root: Path) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = repo_root / path
    return path


def resolve_output_path(path_arg: str, fallback_name: str, repo_root: Path) -> Path:
    path = Path(path_arg) if path_arg else Path("dataset") / fallback_name
    if not path.is_absolute():
        path = repo_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_alpha_noise(length: int, rng: random.Random) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(alphabet) for _ in range(length))


def make_digit_noise(length: int, rng: random.Random) -> str:
    alphabet = "0123456789 "
    return "".join(rng.choice(alphabet) for _ in range(length))


def prepend_noise(prompt: str, noise: str) -> str:
    return f"{noise}\n{prompt}"


def generate_dataset(input_path: Path, output_path: Path, length: int, seed: int, noise_kind: str) -> int:
    rng = random.Random(seed)
    count = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            sample = json.loads(line)
            prompt = sample["conversations"][0]["content"]
            if noise_kind == "alpha":
                noise = make_alpha_noise(length, rng)
            else:
                noise = make_digit_noise(length, rng)

            sample["conversations"][0]["content"] = prepend_noise(prompt, noise)
            dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    args = parse_args()
    if args.length < 0:
        raise ValueError("--length must be >= 0")

    repo_root = Path(__file__).resolve().parent.parent
    input_path = resolve_dataset_path(args.input, repo_root)

    alpha_name = f"sft_calc_addition_test_alpha_noise_len{args.length}_seed{args.seed}.jsonl"
    digit_name = f"sft_calc_addition_test_digit_noise_len{args.length}_seed{args.seed}.jsonl"
    alpha_output = resolve_output_path(args.alpha_output, alpha_name, repo_root)
    digit_output = resolve_output_path(args.digit_output, digit_name, repo_root)

    alpha_count = generate_dataset(input_path, alpha_output, args.length, args.seed, "alpha")
    digit_count = generate_dataset(input_path, digit_output, args.length, args.seed, "digit")

    print(f"Wrote {alpha_count} samples to {alpha_output}")
    print(f"Wrote {digit_count} samples to {digit_output}")


if __name__ == "__main__":
    main()
