"""
Distributed Flower gRPC Server for Multi-Hospital Federated Learning.
Run this process to coordinate remote or cross-silo hospital nodes.

Usage:
    python ml/distributed_flower_server.py --port 8080 --rounds 10 --min-clients 5
"""

import sys
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in [str(_ROOT), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import flwr as fl
from flwr.common import ndarrays_to_parameters
from ml.model import FederatedHealthNet, get_parameters
from ml.flower_strategy import HospitalFedAvgStrategy


def main():
    parser = argparse.ArgumentParser(description="Flower Federated Server")
    parser.add_argument("--port", type=int, default=8080, help="gRPC Server port")
    parser.add_argument("--rounds", type=int, default=10, help="Federated training rounds")
    parser.add_argument("--min-clients", type=int, default=5, help="Minimum clients required")
    parser.add_argument("--input-dim", type=int, default=5, help="Number of model input features")
    args = parser.parse_args()

    # Initialize model weights
    init_net = FederatedHealthNet(input_dim=args.input_dim)
    initial_params = ndarrays_to_parameters(get_parameters(init_net))

    strategy = HospitalFedAvgStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_params,
    )

    server_address = f"0.0.0.0:{args.port}"
    print(f"[*] Starting Flower Federated Server on {server_address}...")
    print(f"[*] Expecting {args.min_clients} hospital clients across {args.rounds} rounds.")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
