import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Read clean.log for the four current methods")
    parser.add_argument(
        "--root",
        type=str,
        default="result_all",
        help="Root directory containing per-method result folders",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=12,
        help="Number of leading lines to print from each clean.log",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    methods = ["baseline", "yarn_orig768", "anchor128", "reset0"]

    for method in methods:
        log_path = root / method / "clean.log"
        print(f"=== {method} ===")
        print(f"log: {log_path}")
        if not log_path.exists():
            print("missing\n")
            continue

        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in lines[: args.lines]:
            print(line)
        print()


if __name__ == "__main__":
    main()
