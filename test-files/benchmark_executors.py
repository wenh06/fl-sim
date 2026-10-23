"""Benchmark the client-executor backends (parallel execution layer).

Usage (from the repo root):

    python test-files/benchmark_executors.py [--model KEY] [--iters N] [--clients N]

Measures the wall-clock time of a fixed-seed federated run under the
``sequential`` and ``threads`` backends across worker counts, and prints
the speedup table. Not part of CI (runner timing noise); results go into
the documentation page of the parallel execution layer.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1].resolve()))

import fl_sim.algorithms as algorithms  # noqa: E402
from fl_sim.data_processing.fed_synthetic import FedSynthetic  # noqa: E402


def run_once(executor: str, num_workers: int, num_iters: int, num_clients: int) -> float:
    reg = algorithms.get_algorithm("FedAvg")
    dataset = FedSynthetic(1, 1, False, 30)
    server = reg["server"](
        dataset.candidate_models["mlp_d1"],
        dataset,
        reg["server_config"](num_iters, num_clients, clients_sample_ratio=1.0, verbose=0),
        reg["client_config"](32, 2),
    )
    server.config.executor = executor
    server.config.num_workers = num_workers
    start = time.perf_counter()
    server.train_federated()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--clients", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    results = {}
    results["sequential"] = min(run_once("sequential", 0, args.iters, args.clients) for _ in range(args.repeats))
    for workers in args.workers:
        results[f"threads@{workers}"] = min(run_once("threads", workers, args.iters, args.clients) for _ in range(args.repeats))

    print(f"\n=== FedAvg / FedSynthetic / {args.iters} iters / {args.clients} clients (best of {args.repeats}) ===")
    baseline = results["sequential"]
    for name, seconds in results.items():
        print(f"{name:16s} {seconds:8.2f} s   speedup {baseline / seconds:5.2f}x")
    with open(Path(__file__).parent / "benchmark_executors_results.json", "w") as f:
        json.dump({"args": vars(args), "seconds": results, "baseline": baseline}, f, indent=1)
    print("results written to test-files/benchmark_executors_results.json")


if __name__ == "__main__":
    main()
