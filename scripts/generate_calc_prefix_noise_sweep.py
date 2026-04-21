import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_LENGTHS = [16, 32, 64, 128, 256, 512, 1024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate multiple calc prefix-noise datasets for a sweep of prefix lengths."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed used for all generated datasets",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/sft_calc_addition_test.jsonl",
        help="Input calc test jsonl path",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=DEFAULT_LENGTHS,
        help="Prefix noise lengths in characters",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent / "generate_calc_prefix_noise_dataset.py"

    for length in args.lengths:
        cmd = [
            sys.executable,
            str(script_path),
            "--input",
            args.input,
            "--length",
            str(length),
            "--seed",
            str(args.seed),
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
