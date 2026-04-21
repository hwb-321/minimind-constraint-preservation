import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Read accuracy from a directory of evaluation jsonl files")
    parser.add_argument(
        "--root",
        type=str,
        default="result_all_attention_fix",
        help="Root directory to scan recursively for jsonl result files",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="Optional JSONL path to save the parsed summary",
    )
    return parser.parse_args()


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


def read_jsonl_acc(path: Path):
    total = 0
    correct = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            correct += bool(row.get("is_correct", False))
    if total == 0:
        return None
    return correct, total, 100.0 * correct / total


def main():
    args = parse_args()
    root = Path(args.root)

    rows = []
    for path in sorted(root.rglob("*.jsonl")):
        metrics = read_jsonl_acc(path)
        if metrics is None:
            continue
        correct, total, acc_percent = metrics
        method = path.parent.relative_to(root).as_posix() or "."
        tag = path.stem
        noise_type, noise_len = parse_tag(tag)
        rows.append(
            {
                "method": method,
                "tag": tag,
                "noise_type": noise_type,
                "noise_len": noise_len,
                "correct": correct,
                "total": total,
                "acc_percent": acc_percent,
                "jsonl_path": str(path),
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
        print(f"{row['tag']:>15} | acc={row['correct']}/{row['total']}={row['acc_percent']:.4f}%")

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nSaved summary to {save_path}")


if __name__ == "__main__":
    main()
