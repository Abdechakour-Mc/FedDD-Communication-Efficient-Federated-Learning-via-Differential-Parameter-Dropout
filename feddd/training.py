# ===== FILE: feddd/training.py =====
from __future__ import annotations
from typing import List, Tuple
from time import perf_counter

import random
import time
import numpy as np
import torch

from torch.utils.data import random_split
from torchvision import datasets, transforms

from .config import ServerConfig, TrainConfig
from .models import TinyCNN
from .server import ParameterServer
from .client import Client
from .allocator import ClientStats
from .utils import apply_mask, clone_state
from .network import NetworkProfile


def _split_dataset(dataset, num_clients: int, seed: int, noniid: bool = False, alpha: float = 0.5) -> List:
    """IID or Dirichlet non-IID split."""
    lengths = [len(dataset) // num_clients] * num_clients
    lengths[0] += len(dataset) - sum(lengths)

    if not noniid:
        g = torch.Generator().manual_seed(seed)
        return list(random_split(dataset, lengths, generator=g))

    # label-wise Dirichlet split for non-IID
    y = np.array(dataset.targets)
    classes = np.unique(y)
    idxs = [np.where(y == c)[0] for c in classes]
    rng = np.random.default_rng(seed)
    parts = [[] for _ in range(num_clients)]
    for cls_idxs in idxs:
        rng.shuffle(cls_idxs)
        portions = rng.dirichlet(alpha=[alpha] * num_clients)
        portions = (portions / portions.sum() * len(cls_idxs)).astype(int)
        # adjust rounding
        while portions.sum() < len(cls_idxs):
            portions[rng.integers(0, num_clients)] += 1
        offset = 0
        for i, p in enumerate(portions):
            parts[i].extend(cls_idxs[offset:offset + p])
            offset += p
    shards = []
    for ids in parts:
        subset = torch.utils.data.Subset(dataset, ids)
        shards.append(subset)
    return shards


def _make_clients(model_fn, shards, cfg: TrainConfig) -> List[Client]:
    return [Client(i, model_fn, shards[i], batch_size=cfg.batch_size, device=cfg.device) for i in range(len(shards))]


def _make_test_loader(name: str):
    T = transforms.Compose([transforms.ToTensor()])
    if name.upper() == 'MNIST':
        test = datasets.MNIST(root='./data', train=False, download=True, transform=T)
    elif name.upper() == 'FMNIST':
        test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=T)
    else:
        raise ValueError('Only MNIST or FMNIST in this demo.')
    return torch.utils.data.DataLoader(test, batch_size=512, shuffle=False, num_workers=0)


@torch.no_grad()
def _evaluate(model, loader, device='cpu'):
    import torch.nn.functional as F
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def run_feddd(train_cfg: TrainConfig, server_cfg: ServerConfig):
    # ---- data ----
    T = transforms.Compose([transforms.ToTensor()])
    if train_cfg.dataset_name.upper() == 'MNIST':
        base = datasets.MNIST(root='./data', train=True, download=True, transform=T)
    elif train_cfg.dataset_name.upper() == 'FMNIST':
        base = datasets.FashionMNIST(root='./data', train=True, download=True, transform=T)
    else:
        raise ValueError('Only MNIST or FMNIST in this demo.')

    shards = _split_dataset(
        base, train_cfg.num_clients, train_cfg.seed,
        noniid=getattr(train_cfg, "noniid", False),
        alpha=getattr(train_cfg, "dirichlet_alpha", 0.5)
    )
    model_fn = lambda: TinyCNN(c1=32, c2=64, fc=256)
    clients = _make_clients(model_fn, shards, train_cfg)
    server = ParameterServer(model_fn, clients, server_cfg, device=train_cfg.device)

    test_loader = _make_test_loader(train_cfg.dataset_name)
    tic = perf_counter()

    # initialize client models with full broadcast (round 0)
    for c in clients:
        c.set_weights(server.global_weights)

    net = NetworkProfile(
        seed=getattr(train_cfg, "net_seed", train_cfg.seed),
        base=getattr(train_cfg, "net_base", 1e7)
    )

    for rnd in range(1, train_cfg.num_rounds + 1):
        # update link rates for all clients (changes every two rounds)
        for c in clients:
            c.run = net.uplink(c.cid, rnd)
            c.rdn = net.downlink(c.cid, rnd)

        server.round = rnd
        reports: List[Tuple[int, dict, dict, int]] = []

        # ---- download per client (Step 6) with barrier sleep ----
        t_downs: List[float] = []
        # (client, W_payload, M, is_full, kept_down_params)
        payloads: List[Tuple[Client, dict, dict | None, bool, int]] = []
        for c in clients:
            W_payload, M, is_full = server.broadcast(c.cid)
            # count PARAMETERS sent (not bytes)
            if is_full or M is None:
                kept_down_params = sum(int(t.numel()) for t in server.global_weights.values())
            else:
                kept_down_params = int(sum(v.sum().item() for v in M.values()))
            payloads.append((c, W_payload, M, is_full, kept_down_params))
            # convert params -> bytes (float32)
            bytes_down = kept_down_params * 4
            t_downs.append(bytes_down / max(1.0, c.rdn))

        # simulate network barrier: wait for slowest downlink
        time.sleep(max(t_downs) if len(t_downs) > 0 else 0.0)

        # apply received weights after barrier
        for (c, W_payload, M, is_full, _kept_down_params) in payloads:
            if is_full:
                # Eq. (6): W_n^{t+1} = W^t (full model)
                c.set_weights(W_payload)
            else:
                # Eq. (5): W_n^{t+1} = W^t ⊙ M_n^t + Ŵ_n^t ⊙ (1 − M_n^t)
                local_prev = c.model.state_dict()
                invM = {k: torch.ones_like(M[k]) - M[k] for k in M}
                part_local = apply_mask(local_prev, invM)
                combined = {k: W_payload[k] + part_local[k] for k in W_payload}
                c.set_weights(combined)

        # ---- local train & build report ----
        stats_for_alloc: List[ClientStats] = []
        per_client_log = []

        # (client, masked_params, mask, m, loss, kept_down_params)
        upload_bundles: List[Tuple[Client, dict, dict, int, float, int]] = []
        t_ups: List[float] = []

        for (c, _W_payload, M, is_full, kept_down_params) in payloads:
            W_before = clone_state(c.model.state_dict())
            W_after, loss = c.local_update(epochs=train_cfg.local_epochs, lr=train_cfg.lr)

            # use last assigned dropout if available; round 1 equal to A_server
            default_D = max(0.0, min(server_cfg.D_max, 1.0 - server_cfg.A_server))
            Dn = default_D if rnd == 1 else server.client_dropout.get(c.cid, default_D)

            out = c.build_report(W_before, W_after, dropout_rate=Dn)
            masked_params, mask, lossv, m = out[0], out[1], out[2], out[3]

            # kept entries (PARAMS) for upload
            kept_up_params = int(sum(v.sum().item() for v in mask.values()))
            # convert to bytes for latency
            bytes_up = kept_up_params * 4
            t_ups.append(bytes_up / max(1.0, c.run))

            upload_bundles.append((c, masked_params, mask, m, lossv, kept_down_params))

            # compute actual keep ratio this round from mask density (for logging)
            total_params = sum(int(v.numel()) for v in mask.values())
            keep_ratio = kept_up_params / max(1, total_params)
            per_client_log.append({
                "cid": c.cid,
                "m": m,
                "loss": lossv,
                "Un": c.Un,
                "keep_ratio_this_round": keep_ratio,
                "run": c.run,
                "rdn": c.rdn,
            })

        # simulate uplink barrier
        time.sleep(max(t_ups) if len(t_ups) > 0 else 0.0)

        # deliver uploads to server + prepare allocator stats (with energy)
        for (c, masked_params, mask, m, lossv, kept_down_params) in upload_bundles:
            server.client_masks[c.cid] = mask
            reports.append((c.cid, masked_params, mask, m))

            # ---- pass battery/energy parameters into allocator (enables constraint) ----
            stats_for_alloc.append(ClientStats(
                cid=c.cid, m_samples=m, loss=lossv, Un=c.Un, tcmp=c.tcmp, run=c.run, rdn=c.rdn,
                # e_up=c.e_up, e_dn=c.e_dn, e_cmp=c.e_cmp, B_rem=c.battery_rem
            ))

            # # ---- OPTIONAL: deduct this round's energy (comm + compute) ----
            # kept_up_params = int(sum(v.sum().item()) for v in mask.values())
            # E_comm = c.e_up * kept_up_params + c.e_dn * kept_down_params
            # E_cmp = c.e_cmp * c.tcmp
            # c.battery_rem = max(0.0, c.battery_rem - (E_comm + E_cmp))

        # ---- aggregate (Eq. 4) ----
        server.aggregate(reports)

        # ---- evaluate global accuracy ----
        acc = _evaluate(server.model, test_loader, device=train_cfg.device)
        elapsed = perf_counter() - tic

        # ---- allocate per-client dropout for next round (Eq. 14–17) ----
        D_map = server.allocate_dropout_rates(stats_for_alloc, use_cvx=True)
        server.client_dropout = D_map

        # ---- logging ----
        avg_loss = sum(s.loss for s in stats_for_alloc) / len(stats_for_alloc)
        avg_D = sum(D_map.values()) / max(1, len(D_map))
        print(f"[Round {rnd}] acc={acc*100:5.2f}% | time={elapsed:7.2f}s | avg loss={avg_loss:.4f} | avg dropout={avg_D:.3f}")
        print("    per-client D:", {k: round(v, 3) for k, v in sorted(D_map.items())})

        # per-client details
        for row in sorted(per_client_log, key=lambda r: r["cid"]):
            cid = row["cid"]
            Dn_next = D_map.get(cid, float('nan'))
            keep_next = (1.0 - Dn_next) if isinstance(Dn_next, float) else float('nan')
            up_MBps = row['run'] / 1e6
            dn_MBps = row['rdn'] / 1e6
            # crude time estimates (units consistent with Un and rates):
            est_up = (row['Un'] * keep_next / row['run']) if isinstance(keep_next, float) else float('nan')
            est_dn = (row['Un'] * keep_next / row['rdn']) if isinstance(keep_next, float) else float('nan')
            print(
                f"  - client {cid:>2} | m={row['m']:<5d} | loss={row['loss']:.4f} | Un={row['Un']:<7d} "
                f"| kept_now~{row['keep_ratio_this_round']:.3f} | next_D={Dn_next:.3f} (keep~{keep_next:.3f}) "
                f"| net up≈{up_MBps:.2f}MB/s dn≈{dn_MBps:.2f}MB/s | est_next_up≈{est_up:.3f}s dn≈{est_dn:.3f}s"
            )

    print("Training done.")