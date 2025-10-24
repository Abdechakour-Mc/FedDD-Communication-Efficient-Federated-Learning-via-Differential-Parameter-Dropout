# FedDD: Toward Communication-Efficient Federated Learning With Differential Parameter Dropout (unofficial)
This repository provides an independent, research-oriented implementation of a communication-efficient Federated Learning (FL) method inspired by the “FedDD” approach (differential parameter dropout with importance-based sparsification). The goal is to make the core ideas reproducible for coursework, benchmarking, and extensions.

## Highlights

* **Differential dropout allocation** per client via a convex program (CVXPY), enforcing a global upload budget while adapting to client heterogeneity.
* **Importance-based sparsification** at the client (per-channel scoring in Conv/Linear layers) to upload only the most impactful parameters.
* **Masked aggregation** at the server with correct fallback: if no client updates a parameter, the previous global value is retained.
* **Sparse broadcasts** with periodic full refresh (`h_full_broadcast`) to bound drift.
* **Latency simulation** using a configurable network model with lognormal jitter and straggler throttling.
* **Built-in evaluation** (global test accuracy per round) and two canonical experiments:

  * Time-to-Accuracy (T2A) vs. FedAvg under IID.
  * Robustness under shrinking upload budget in non-IID settings.

---

## Repository Layout

```
feddd/
  allocator.py      # Dropout allocation (CVXPY program + heuristic fallback)
  client.py         # Local training, importance masks, sparse update reporting
  server.py         # Masked aggregation, sparse/full broadcast policy
  training.py       # Training loop, network simulation, evaluation, logging
  importance.py     # Per-layer importance scoring and mask construction
  models.py         # TinyCNN for MNIST / FMNIST
  network.py        # Uplink/downlink profile (heterogeneous, straggler support)
  config.py         # Experiment configuration dataclasses
  utils.py          # State management utilities
  __init__.py
run_demo.py         # Entry point with example runs (Exp-A / Exp-B)
```

---

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.0
* torchvision
* numpy
* cvxpy (recommended; falls back to heuristic if unavailable)

Install:

```bash
pip install torch torchvision numpy cvxpy
```

> If installing cvxpy is problematic, you can temporarily omit it; the allocator will use a heuristic. For faithful behavior, enable cvxpy (ECOS or another supported solver).

---

## Quick Start

The dataset (MNIST/Fashion-MNIST) is downloaded automatically by `torchvision`.

### Run the demo

```bash
python run_demo.py
```

The repo ships with two experiment patterns inside `run_demo.py`. Uncomment the block you want to run.

---

## Experiments

### A) Time-to-Accuracy (T2A) vs. FedAvg (IID FMNIST)

Purpose: Compare baseline FedAvg (full uploads) with the differential dropout approach at a constrained upload budget on IID data.

In `run_demo.py`, use a configuration like:

```python
# Common train config
train = TrainConfig(
    num_rounds=12, num_clients=6,
    local_epochs=2, batch_size=64, lr=0.01,
    device="cuda", dataset_name="FMNIST", seed=42
)

# Baseline: FedAvg (full communication)
fedavg = ServerConfig(A_server=1.0, D_max=0.0, h_full_broadcast=1)
run_feddd(train, fedavg)

# FedDD: constrained upload with periodic full broadcasts
feddd = ServerConfig(A_server=0.6, D_max=0.8, h_full_broadcast=3)
run_feddd(train, feddd)
```

Interpretation:

* The script prints test accuracy (`acc`) and accumulated simulated time (`time`) each round.
* To report T2A, pick a target accuracy and compare the times at which each method first reaches it.

### B) Robustness under Shrinking Budget (Non-IID FMNIST)

Purpose: Observe accuracy degradation as the global upload budget is reduced, under a non-IID client partition.

Example loop in `run_demo.py`:

```python
train = TrainConfig(
    num_rounds=12, num_clients=10,
    local_epochs=2, batch_size=64, lr=0.01,
    device="cuda", dataset_name="FMNIST", seed=7, net_base=1e7
)
train.noniid = True
train.dirichlet_alpha = 0.3  # lower -> more skew

for A in [0.8, 0.4, 0.2]:
    print(f"\n=== FedDD non-IID (A_server={A}) ===")
    server = ServerConfig(A_server=A, D_max=0.9, h_full_broadcast=3)
    run_feddd(train, server)
```

Interpretation:

* Compare final round accuracies across `A_server` values.
* Expect graceful degradation; sparsity should reduce communication while maintaining reasonable accuracy.

---

## Configuration

### `TrainConfig` (in `feddd/config.py`)

* `num_rounds`: number of federated rounds
* `num_clients`: number of clients
* `local_epochs`, `batch_size`, `lr`
* `device`: `"cpu"` or `"cuda"`
* `dataset_name`: `"MNIST"` or `"FMNIST"`
* `seed`: base RNG seed
* `net_seed`: network RNG seed (default: `seed`)
* `net_base`: average link bandwidth in bytes/sec (affects simulated latency)
* Optional runtime attributes:

  * `noniid: bool` — non-IID partition (default False)
  * `dirichlet_alpha: float` — skew for Dirichlet partition

### `ServerConfig`

* `A_server`: target average keep fraction across clients (e.g., `0.6` means 60% of parameters kept/uploaded on average)
* `D_max`: per-client maximum dropout
* `h_full_broadcast`: frequency of full-model broadcasts (e.g., `3` → full every 3 rounds)
* `delta`: trade-off weight in the allocator objective

---

## Implementation Notes

* **Masked aggregation with fallback:** The server aggregates per-parameter using masks. If no client contributes to a position in a round, the previous global weight is preserved (prevents drift toward zero).
* **Importance-based masks:** Clients compute per-channel importance (Conv/Linear) based on parameter change magnitude and relative ratio; only the top fraction (set by dropout rate) are uploaded.
* **Differential dropout allocation:** The server solves a small convex program to assign next-round dropout rates, minimizing round time under a global budget. If the solver fails or is absent, a heuristic allocation is used.
* **Network simulation:** Each round simulates download/upload times based on per-client rates (lognormal jitter) and an optional throttled straggler. The loop uses barrier sleeps to model synchronous rounds.

---

## Output and Logging

Per round, the script prints:

```
[Round t] acc=XX.XX% | time=YYY.YYs | avg loss=... | avg dropout=...
    per-client D: {0: ..., 1: ..., ...}
  - client 00 | m=... | loss=... | Un=... | kept_now~... | next_D=... (keep~...) | net up≈...MB/s dn≈...MB/s | est_next_up≈...s dn≈...s
```

* `acc`: global test accuracy
* `time`: cumulative simulated time including download/upload barriers
* `avg dropout`: mean dropout across clients (lower keep → sparser communication)
* Per-client line shows current keep ratio, next assigned dropout, and estimated latencies.

---

## Reproducibility Tips

* Fix `seed` and `net_seed` to stabilize both data partitioning and network profiles.
* For debugging/tracing, temporarily set `h_full_broadcast=1` (always full broadcasts), then re-enable sparse downloads.
* Ensure `cvxpy` is installed (e.g., `pip install cvxpy ecos`) to use the convex allocator; otherwise, the heuristic may lead to slightly different allocations.

---

## Limitations

* This implementation focuses on clarity and alignment with the algorithmic ideas; it is not optimized for production throughput.
* The network model is a simulation (sleeps) and does not perform real network transfers.
* Only MNIST/Fashion-MNIST and a compact CNN are provided for quick iteration.

---

## Citation

If this code assists your work, please cite the original FedDD paper as appropriate. This repository is an independent re-implementation for educational and research use.

```
@article{feddd2023,
  title={FedDD: Toward Communication-Efficient Federated Learning With Differential Parameter Dropout},
  author={Liu, et al.},
  journal={IEEE Transactions on Mobile Computing},
  year={2023}
}
```

---

## License

Specify your chosen license here (e.g., MIT).
