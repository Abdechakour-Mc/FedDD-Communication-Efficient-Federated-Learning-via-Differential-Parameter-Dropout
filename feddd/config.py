from dataclasses import dataclass

@dataclass
class ServerConfig:
    A_server: float = 0.5 # proportion of parameters server wants (constraint)
    D_max: float = 0.9 # max dropout per client
    h_full_broadcast: int = 5 # send full model every h rounds
    delta: float = 3.0          # tradeoff in allocator objective (paper default)

@dataclass
class TrainConfig:
    num_rounds: int = 20
    num_clients: int = 5
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 0.01
    device: str = "cpu"
    dataset_name: str = "MNIST" # or "FMNIST"
    seed: int = 42
    net_seed: int = 42
    # net_base: float = 1e7
    net_base: float = 5e5 # 0.5 Mb/s or 1e7 10 Mb/s

