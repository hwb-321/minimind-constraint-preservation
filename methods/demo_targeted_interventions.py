import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parent.parent))

from methods.targeted_interventions import (
    build_blocking_attention_mask,
    find_calc_span,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo calc-targeted attention block intervention.")
    parser.add_argument("--tokenizer_path", type=str, default="./model", help="Tokenizer path")
    parser.add_argument(
        "--text",
        type=str,
        default="noise noise noise\n<calc>\n1 7 + 2 5 =\n</calc>",
        help="Input text containing a calc span",
    )
    parser.add_argument("--keep_prefix", type=int, default=0, help="How many prefix tokens remain visible in blocking mask")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    input_ids = tokenizer(args.text, add_special_tokens=False).input_ids
    calc_start_ids = tokenizer("<calc>", add_special_tokens=False).input_ids
    calc_end_ids = tokenizer("</calc>", add_special_tokens=False).input_ids

    calc_span = find_calc_span(input_ids, calc_start_ids, calc_end_ids)
    print("input_ids:", input_ids)
    print("tokens:", tokenizer.convert_ids_to_tokens(input_ids))
    print("calc_span:", calc_span)

    mask = build_blocking_attention_mask(len(input_ids), calc_span, keep_prefix=args.keep_prefix)

    print("blocking_mask_shape:", tuple(mask.shape))
    print("blocking_mask_last_rows:")
    for row in mask[-min(5, mask.size(0)):]:
        print(row.int().tolist())


if __name__ == "__main__":
    main()
