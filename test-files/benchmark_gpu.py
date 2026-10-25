"""GPU/CPU benchmark for the client-executor backends (light footprint).

Runs FedAvg on FedSynthetic(dimension=512) with the mlp_d3 candidate model
(~1.19M params), comparing sequential vs threads backends on CPU and GPU.
Best-of-3 wall clock per configuration; results appended to a log file.

Run:
    python test-files/benchmark_gpu.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1].resolve()))

import torch  # noqa: E402

import fl_sim.algorithms as algorithms  # noqa: E402
from fl_sim.data_processing.fed_synthetic import FedSynthetic  # noqa: E402

ITERS = 3
CLIENTS = 64
EPOCHS = 5
BATCH = 256
WORKERS = 8
REPEATS = 3


def run_once(executor: str, num_workers: int, use_gpu: bool) -> float:
    reg = algorithms.get_algorithm("FedAvg")
    dataset = FedSynthetic(1, 1, False, max(30, CLIENTS), dimension=512)
    server = reg["server"](
        dataset.candidate_models["mlp_d3"],
        dataset,
        reg["server_config"](
            ITERS,
            CLIENTS,
            clients_sample_ratio=1.0,
            verbose=0,
            visiable_gpus=[0] if use_gpu else [],
        ),
        reg["client_config"](BATCH, EPOCHS),
    )
    server.config.executor = executor
    server.config.num_workers = num_workers
    start = time.perf_counter()
    server.train_federated()
    return time.perf_counter() - start


def bench(tag: str, use_gpu: bool) -> None:
    print(f"=== {tag} ({ITERS} iters x {CLIENTS} clients x {EPOCHS} local epochs, best of {REPEATS}) ===", flush=True)
    base = min(run_once("sequential", 0, use_gpu) for _ in range(REPEATS))
    print(f"  sequential      {base:8.2f} s   1.00x", flush=True)
    t = min(run_once("threads", WORKERS, use_gpu) for _ in range(REPEATS))
    print(f"  threads@{WORKERS}      {t:8.2f} s   {base / t:5.2f}x", flush=True)


if __name__ == "__main__":
    print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    bench("CPU", use_gpu=False)
    if torch.cuda.is_available():
        bench("GPU", use_gpu=True)
    print("BENCH_GPU_DONE", flush=True)
