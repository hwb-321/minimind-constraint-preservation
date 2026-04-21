import argparse
import json
import re
from pathlib import Path


ACC_RE = re.compile(r"Accuracy:\s+(\d+)/(\d+)\s+=\s+([0-9.]+)%")


def parse_args():
    parser = argparse.ArgumentParser(description="Read accuracies from result_all experiment folders")
    parser.add_argument(
        "--root",
        type=str,
        default="result_all",
        help="Root directory containing per-method subdirectories",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional path to save the parsed summary as JSONL",
    )
    return parser.parse_args()


def read_acc_from_log(log_path: Path):
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = ACC_RE.search(line)
        if match:
            correct = int(match.group(1))
            total = int(match.group(2))
            acc_percent = float(match.group(3))
            return {
                "correct": correct,
                "total": total,
                "acc_percent": acc_percent,
                "source": "log",
            }
    return None


def read_acc_from_jsonl(jsonl_path: Path):
    if not jsonl_path.exists():
        return None

    total = 0
    correct = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            if row.get("is_correct", False):
                correct += 1

    if total == 0:
        return None

    return {
        "correct": correct,
        "total": total,
        "acc_percent": 100.0 * correct / total,
        "source": "jsonl",
    }


def parse_tag(tag: str):
    if tag == "clean":
        return "clean", 0

    parts = tag.split("_len")
    if len(parts) != 2:
        return tag, None

    noise_type = parts[0]
    try:
        noise_len = int(parts[1])
    except ValueError:
        noise_len = None
    return noise_type, noise_len


def main():
    args = parse_args()
    root = Path(args.root)

    rows = []
    for method_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        method = method_dir.name
        for jsonl_path in sorted(method_dir.glob("*.jsonl")):
            tag = jsonl_path.stem
            log_path = method_dir / f"{tag}.log"

            metrics = read_acc_from_log(log_path)
            if metrics is None:
                metrics = read_acc_from_jsonl(jsonl_path)
            if metrics is None:
                continue

            noise_type, noise_len = parse_tag(tag)
            rows.append(
                {
                    "method": method,
                    "tag": tag,
                    "noise_type": noise_type,
                    "noise_len": noise_len,
                    "correct": metrics["correct"],
                    "total": metrics["total"],
                    "acc_percent": metrics["acc_percent"],
                    "source": metrics["source"],
                    "log_path": str(log_path) if log_path.exists() else "",
                    "jsonl_path": str(jsonl_path),
                }
            )

    rows.sort(
        key=lambda x: (
            x["method"],
            -1 if x["noise_len"] is None else x["noise_len"],
            x["noise_type"],
            x["tag"],
        )
    )

    current_method = None
    for row in rows:
        if row["method"] != current_method:
            current_method = row["method"]
            print(f"\n[{current_method}]")
        print(
            f"{row['tag']:>15} | "
            f"acc={row['correct']}/{row['total']}={row['acc_percent']:.4f}% "
            f"({row['source']})"
        )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved summary to {save_path}")


if __name__ == "__main__":
    main()
