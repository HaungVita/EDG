"""Shape adapter that accompanied the historical EDG experiments.

EDG feeds batch-first tensors to ActionFormer, whose backbone expects
channel-first tensors with a fixed temporal extent of 128.
"""
import os

import torch
import torch.nn.functional as F


def fit_model_input(features, max_len=128):
    features = features.transpose(1, 2)
    if features.size(-1) < max_len:
        features = F.pad(features, (0, max_len - features.size(-1)))
    return features[..., :max_len]


def fit_model_mask(mask, max_len=128):
    mask = mask.to(dtype=bool).unsqueeze(1)
    mode = os.environ.get("EDG_MASK_MODE", "valid_zero")
    if mask.size(-1) < max_len:
        pad_value = mode in {"padding_true", "invert_valid_zero"}
        mask = F.pad(mask, (0, max_len - mask.size(-1)), value=pad_value)
    mask = mask[..., :max_len]
    if mode in {"invert", "invert_valid_zero"}:
        mask = torch.logical_not(mask)
    if mode not in {"valid_zero", "padding_true", "invert", "invert_valid_zero"}:
        raise ValueError(f"Unknown EDG_MASK_MODE={mode!r}")
    return mask
