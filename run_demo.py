# -------------------------------
# file: run_demo.py
# -------------------------------
from feddd import ServerConfig, TrainConfig, run_feddd

# # (Exp-A) T2A: FedDD vs. FedAvg on FMNIST, IID (paper setup)
# if __name__ == "__main__":
#     train = TrainConfig(
#         num_rounds=12,
#         num_clients=10,            # paper commonly uses 10 clients
#         local_epochs=2,
#         batch_size=64,
#         lr=0.01,
#         device="cuda",
#         dataset_name="FMNIST",      # paper’s Fig. 7 uses MNIST/FMNIST (IID)
#         seed=42,
#         net_seed=42,
#         # net_base=1e7               # 10 MB/s base
#         net_base=4e5,   # ≈0.38 MB/s
#     )

#     print("\n=== FedAvg (full) ===")
#     fedavg = ServerConfig(A_server=1.0, D_max=0.0, h_full_broadcast=1, delta=3.0)
#     run_feddd(train, fedavg)

#     print("\n=== FedDD (A_server=0.6, h=3) ===")
#     feddd = ServerConfig(A_server=0.4, D_max=0.9, h_full_broadcast=3, delta=3.0)
#     run_feddd(train, feddd)



# (Exp-B) Robustness under shrinking budget (non-IID), FMNIST


if __name__ == "__main__":
    train = TrainConfig(
        num_rounds=12, num_clients=10,
        local_epochs=2, batch_size=64, lr=0.01,
        device="cuda", dataset_name="FMNIST", seed=7, net_base=1e7
    )
    # enable non-IID in TrainConfig dynamically
    train.noniid = True
    train.dirichlet_alpha = 0.3

    for A in [0.8, 0.4, 0.2]:
        print(f"\n=== FedDD non-IID (A_server={A}) ===")
        server = ServerConfig(A_server=A, D_max=0.9, h_full_broadcast=3)
        run_feddd(train, server)

















# # -------------------------------
# # file: run_demo.py
# # -------------------------------
# from feddd import ServerConfig, TrainConfig, run_feddd

# # if __name__ == "__main__":
# #     train = TrainConfig(num_rounds=12, num_clients=6, local_epochs=2, batch_size=64, lr=0.01, device="cuda", dataset_name="MNIST")
# #     server = ServerConfig(A_server=0.8, D_max=0.98, h_full_broadcast=6)
# #     run_feddd(train, server)


# # (Exp-A) T2A: FedDD vs. FedAvg on FMNIST, IID
# if __name__ == "__main__":
#     # Common train config
#     train = TrainConfig(
#         num_rounds=12, num_clients=6,
#         local_epochs=2, batch_size=64, lr=0.01,
#         device="cuda", dataset_name="FMNIST", seed=42
#     )

#     # ===== Baseline: FedAvg =====
#     print("\n=== FedAvg (full) ===")
#     fedavg_server = ServerConfig(A_server=1.0, D_max=0.0, h_full_broadcast=1)
#     run_feddd(train, fedavg_server)

#     # ===== FedDD: 60% budget =====
#     print("\n=== FedDD (A_server=0.6) ===")
#     feddd_server = ServerConfig(A_server=0.6, D_max=0.8, h_full_broadcast=3)
#     run_feddd(train, feddd_server)