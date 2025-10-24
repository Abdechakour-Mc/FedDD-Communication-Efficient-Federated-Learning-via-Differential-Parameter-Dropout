from __future__ import annotations
from typing import Dict
import torch
from torch import nn
from .utils import State

@torch.no_grad()
def _imp_conv(W: torch.Tensor, dW: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # per-out-channel score
    reduce_dims = (1, 2, 3)
    num = dW.abs().sum(dim=reduce_dims)
    ratio = (W.add(dW).abs().sum(dim=reduce_dims)) / (W.abs().sum(dim=reduce_dims) + eps)
    return num * ratio

@torch.no_grad()
def _imp_linear(W: torch.Tensor, dW: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    num = dW.abs().sum(dim=1)  # across in_features
    ratio = (W.add(dW).abs().sum(dim=1)) / (W.abs().sum(dim=1) + eps)
    return num * ratio

@torch.no_grad()
def build_channel_masks(model: nn.Module, W_old: State, W_new: State, dropout_rate: float) -> State:
    D = float(dropout_rate)
    masks: State = {}
    modules = dict(model.named_modules())
    for name, param in W_new.items():
        base = name.rsplit('.', 1)[0] if '.' in name else name
        m = modules.get(base, None)
        if m is None:
            masks[name] = torch.ones_like(param)
            continue
        if isinstance(m, nn.Conv2d) and name.endswith('weight'):
            scores = _imp_conv(W_old[name], W_new[name] - W_old[name])
            keep = max(1, int(round(scores.numel() * (1.0 - D))))
            idx = torch.topk(scores, k=keep).indices
            mask_o = torch.zeros_like(scores, dtype=param.dtype); mask_o[idx] = 1.0
            masks[name] = mask_o.view(-1,1,1,1).expand_as(param)
            bkey = base + '.bias'
            if bkey in W_new:
                masks[bkey] = mask_o.view(-1)
        elif isinstance(m, nn.Linear) and name.endswith('weight'):
            scores = _imp_linear(W_old[name], W_new[name] - W_old[name])
            keep = max(1, int(round(scores.numel() * (1.0 - D))))
            idx = torch.topk(scores, k=keep).indices
            mask_o = torch.zeros_like(scores, dtype=param.dtype); mask_o[idx] = 1.0
            masks[name] = mask_o.view(-1,1).expand_as(param)
            bkey = base + '.bias'
            if bkey in W_new:
                masks[bkey] = mask_o.view(-1)
        else:
            masks[name] = torch.ones_like(param)
    return masks