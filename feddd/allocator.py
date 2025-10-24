from __future__ import annotations
from typing import Dict, List, Tuple
import math


import cvxpy as cp


class ClientStats:
    def __init__(
        self, cid: int, m_samples: int, loss: float, Un: int, tcmp: float, run: float, rdn: float,
        # e_up: float | None = None, e_dn: float | None = None, e_cmp: float | None = None, B_rem: float | None = None
    ):
        self.cid = cid                     # unique client ID (identifier for this client)
        self.m = m_samples                 # number of local training samples (data size)
        self.loss = max(1e-6, float(loss)) # latest average training loss (data quality indicator)
        self.Un = Un                       # number of trainable parameters (model size, upload cost proxy)
        self.tcmp = max(1e-6, float(tcmp)) # local computation latency (time to finish one local update)
        self.run = max(1e-6, float(run))   # uplink bandwidth (bytes/sec, affects upload time)
        self.rdn = max(1e-6, float(rdn))   # downlink bandwidth (bytes/sec, affects download time)

        # # energy model (optional, all in consistent units, e.g., Joules)
        # self.e_up  = float(e_up)   # J per uploaded parameter
        # self.e_dn  = float(e_dn)   # J per downloaded parameter
        # self.e_cmp = float(e_cmp)  # J per second of compute
        # self.B_rem = float(B_rem)  # J remaining this round


    def upload_time(self, frac_keep: float) -> float:
        return self.Un * frac_keep / self.run

# Heuristic fallback inspired by Eq. (13)-(15)
def allocate_heuristic(stats: List[ClientStats], A_server: float, D_max: float) -> Dict[int, float]:
    U_sum = sum(s.Un for s in stats)
    m_sum = sum(s.m for s in stats)
    # weight ~ contribution / latency
    weights = []
    for s in stats:
        ren = (s.m / max(1, m_sum)) * (s.Un / max(1, U_sum)) * s.loss
        latency = s.tcmp + s.Un / s.run
        w = ren / max(1e-9, latency)
        weights.append((s.cid, max(1e-12, w)))
    Z = sum(w for _, w in weights)
    target_fraction = A_server  # average fraction to keep across clients
    per_client_frac = {cid: target_fraction * (w / Z) * len(stats) for cid, w in weights}
    D = {}
    for cid, frac in per_client_frac.items():
        frac = float(max(0.0, min(1.0, frac)))
        D[cid] = float(max(0.0, min(D_max, 1.0 - frac)))
    return D

# Exact convex program for Eq. (14)-(17)
def allocate_cvx(stats: List[ClientStats], A_server: float, D_max: float, delta: float = 3.0) -> Dict[int, float]:
    if cp is None:
        return allocate_heuristic(stats, A_server, D_max)
    N = len(stats)
    # Variables
    D = cp.Variable(N)         # dropout rates per client
    tserver = cp.Variable()    # round time upper bound
    # Constants
    Un = [s.Un for s in stats]
    mn = [s.m for s in stats]
    losst = [s.loss for s in stats]
    run = [s.run for s in stats]
    rdn = [s.rdn for s in stats]
    tcmp = [s.tcmp for s in stats]
    U = sum(Un)

    # # Energy constants (optional)
    # e_up  = [s.e_up  for s in stats]
    # e_dn  = [s.e_dn  for s in stats]
    # e_cmp = [s.e_cmp for s in stats]
    # Brem  = [s.B_rem for s in stats]

    # Objective: tserver + delta * sum( mn/m * (Un/U) * losst * D_n^t )
    m_total = max(1, sum(mn))
    coeff = [ (mn[i]/m_total) * (Un[i]/U) * losst[i] for i in range(N) ]
    objective = cp.Minimize(tserver + delta * cp.sum(cp.multiply(coeff, D)))
    # Constraints
    cons = []
    # bounds
    cons += [D >= 0, D <= D_max]
    # server parameter budget: average keep fraction equals A_server
    cons += [ A_server * U - cp.sum([ Un[i]*(1 - D[i]) for i in range(N) ]) == 0 ]
    # per-client total round time <= tserver: tcmp + upload + download <= tserver
    # upload time = Un*(1-D)/run ; download time = Un*(1-D)/rdn (symmetric for simplicity)
    for i in range(N):
        up = Un[i] * (1 - D[i]) / run[i]
        down = Un[i] * (1 - D[i]) / rdn[i]
        cons += [ tcmp[i] + up + down <= tserver ]

        # # ---- Battery / energy per round (optional) ----
        # # You can add any type of constraints (based on use case)
        # # E_comm = e_up * kept_upload + e_dn * kept_download
        # # kept_upload/download are both Un[i]*(1 - D[i]) under the simple symmetric model
        # # E_cmp  = e_cmp * tcmp (compute energy)
        # # Total  <= B_rem (remaining battery energy for this round)
        # if Brem[i] is not None:
        #     E_comm = (e_up[i] + e_dn[i]) * Un[i] * (1 - D[i])
        #     E_cmp  = e_cmp[i] * tcmp[i]
        #     cons += [ E_comm + E_cmp <= Brem[i] ]

    prob = cp.Problem(objective, cons)
    try:
        prob.solve(solver=cp.ECOS, warm_start=True)
        vals = D.value
        if vals is None:
            return allocate_heuristic(stats, A_server, D_max)
        out = {}
        for i, s in enumerate(stats):
            di = float(max(0.0, min(D_max, vals[i])))
            out[s.cid] = di
        return out
    except Exception:
        return allocate_heuristic(stats, A_server, D_max)