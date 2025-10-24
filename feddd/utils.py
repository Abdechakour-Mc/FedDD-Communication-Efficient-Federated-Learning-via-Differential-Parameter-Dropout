from __future__ import annotations
from typing import Dict
import torch

State = Dict[str, torch.Tensor]

def clone_state(sd: State) -> State:
    return {k: v.detach().clone() for k, v in sd.items()}

def zeros_like_state(sd: State) -> State:
    return {k: torch.zeros_like(v) for k, v in sd.items()}

def elementwise_div(numer: State, denom: State, prev: State, eps: float = 1e-12) -> State:
    out = {}
    for k in numer:
        mask_has = (denom[k] > 0)
        safe_div = numer[k] / (denom[k] + eps)
        out[k] = torch.where(mask_has, safe_div, prev[k])
    return out

def apply_mask(sd: State, mask: State) -> State:
    return {k: sd[k] * mask[k] for k in sd}

def count_parameters(sd: State) -> int:
    return int(sum(v.numel() for v in sd.values()))