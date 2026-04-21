from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class CalcSpan:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def find_subsequence(sequence: Sequence[int], pattern: Sequence[int], start: int = 0) -> Optional[int]:
    if not pattern:
        return None
    last = len(sequence) - len(pattern) + 1
    for i in range(start, max(last, 0)):
        if list(sequence[i:i + len(pattern)]) == list(pattern):
            return i
    return None


def find_calc_span(
    input_ids: Sequence[int],
    calc_start_ids: Sequence[int],
    calc_end_ids: Sequence[int],
) -> Optional[CalcSpan]:
    start = find_subsequence(input_ids, calc_start_ids)
    if start is None:
        return None
    end_start = find_subsequence(input_ids, calc_end_ids, start + len(calc_start_ids))
    if end_start is None:
        return None
    return CalcSpan(start=start, end=end_start + len(calc_end_ids))


def build_blocking_attention_mask(
    seq_len: int,
    calc_span: Optional[CalcSpan],
    keep_prefix: int = 0,
    device: Optional[torch.device] = None,
) -> torch.BoolTensor:
    mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    if calc_span is None:
        return mask

    for row in range(calc_span.start, seq_len):
        row_allowed = torch.zeros(seq_len, dtype=torch.bool, device=device)
        if keep_prefix > 0:
            row_allowed[:keep_prefix] = True
        # Keep the whole suffix from <calc> start up to the current row visible.
        # This preserves the assistant/generation prompt tokens that appear after </calc>
        # in the chat template, instead of accidentally masking them out.
        row_allowed[calc_span.start:row + 1] = True
        row_allowed[row + 1:] = False
        mask[row] = row_allowed
    return mask


def extend_blocking_attention_mask(
    mask: torch.BoolTensor,
    calc_span: Optional[CalcSpan],
    keep_prefix: int = 0,
) -> torch.BoolTensor:
    if mask.dim() != 2 or mask.size(0) != mask.size(1):
        raise ValueError("Blocking attention mask must be a square 2D tensor.")

    old_len = mask.size(0)
    new_len = old_len + 1
    new_mask = torch.zeros(new_len, new_len, dtype=torch.bool, device=mask.device)
    new_mask[:old_len, :old_len] = mask

    if calc_span is None:
        new_mask[:, :] = torch.ones(new_len, new_len, dtype=torch.bool, device=mask.device).tril()
        return new_mask

    row_allowed = torch.zeros(new_len, dtype=torch.bool, device=mask.device)
    if keep_prefix > 0:
        row_allowed[: min(keep_prefix, new_len)] = True
    row_allowed[calc_span.start:new_len] = True
    new_mask[new_len - 1] = row_allowed
    return new_mask
