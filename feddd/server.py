from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import torch
from .config import ServerConfig
from .utils import State, zeros_like_state, elementwise_div, apply_mask, count_parameters
from .allocator import ClientStats, allocate_cvx, allocate_heuristic

class ParameterServer:
    def __init__(self, model_fn, clients, cfg: ServerConfig, device: str = "cpu"):
        self.model = model_fn().to(device)
        self.device = device
        self.clients = clients
        self.cfg = cfg
        self.global_weights: State = {k: v.detach().clone() for k,v in self.model.state_dict().items()}
        self.U = count_parameters(self.global_weights)
        self.round = 0 # Current training round  
        # store last masks and dropout per client for sparse download
        self.client_masks: Dict[int, State] = {}
        self.client_dropout: Dict[int, float] = {}

    @torch.no_grad()
    def aggregate(self, reports: List[Tuple[int, State, State, int]]):
        # Each report: (cid, masked_params, mask, m_samples)
        numer = zeros_like_state(self.global_weights)
        denom = zeros_like_state(self.global_weights)
        for cid, masked_params, mask, m in reports:
            w = float(m)
            for k in self.global_weights.keys():
                numer[k] += w * masked_params[k].to(self.device)
                denom[k] += w * mask[k].to(self.device)
        # new_global = elementwise_div(numer, denom, eps=1e-12)
        new_global = elementwise_div(numer, denom, prev=self.global_weights, eps=1e-12)
        self.global_weights = new_global
        self.model.load_state_dict(new_global)

    def allocate_dropout_rates(self, stats: List[ClientStats], use_cvx: bool = True) -> Dict[int, float]:
        if use_cvx:
            return allocate_cvx(stats, self.cfg.A_server, self.cfg.D_max, delta=self.cfg.delta)
        return allocate_heuristic(stats, self.cfg.A_server, self.cfg.D_max)

    
    def broadcast(self, cid: int):
        # Implements Step 6: full model every h rounds; otherwise sparse W^t ⊙ M_n^t
        if self.round % self.cfg.h_full_broadcast == 0 or cid not in self.client_masks:
            return self.global_weights, None, True
        M = self.client_masks[cid]
        return apply_mask(self.global_weights, M), M, False