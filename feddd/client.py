from __future__ import annotations
from typing import Tuple, Dict
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .utils import State, clone_state, apply_mask, count_parameters
from .importance import build_channel_masks

class Client:
    def __init__(self, cid: int, model_fn, dataset, batch_size: int = 64, device: str = 'cpu'):
        self.cid = cid
        self.model: nn.Module = model_fn().to(device)
        self.device = device
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.m_samples = len(dataset)
        self.loss_last = None
        self.Un = count_parameters(self.model.state_dict())
        # simple link rates (mock); you can instrument real ones
        self.run = 1e7   # uplink bytes/sec
        self.rdn = 1e7   # downlink bytes/sec
        self.tcmp = max(1.0, self.m_samples / 1000.0)

        # # ---- simple energy model (tune these) ----
        # self.e_up  = 2.0e-9   # J per parameter uploaded (placeholder)
        # self.e_dn  = 1.5e-9   # J per parameter downloaded (placeholder)
        # self.e_cmp = 1.0      # J per second compute (placeholder power)
        # self.battery_rem = 50.0  # J available for THIS round (you can make it cumulative)


    @torch.no_grad()
    def set_weights(self, W: State):
        self.model.load_state_dict(W, strict=True)

    def local_update(self, epochs: int = 1, lr: float = 0.01) -> Tuple[State, float]:
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        total, cnt = 0.0, 0
        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad(); logits = self.model(x); loss = F.cross_entropy(logits, y)
                loss.backward(); opt.step()
                total += float(loss.detach().cpu()); cnt += 1
        avg = total / max(1, cnt)
        self.loss_last = avg
        return clone_state(self.model.state_dict()), avg

    @torch.no_grad()
    def build_report(self, global_before: State, local_after: State, dropout_rate: float) -> Tuple[State, State, float, int, int, int]:
        mask = build_channel_masks(self.model, global_before, local_after, dropout_rate)
        masked_params = apply_mask(local_after, mask)
        upload = sum(int(mask[k].sum().item()) for k in mask.keys())
        return masked_params, mask, float(self.loss_last or 0.0), self.m_samples, upload
