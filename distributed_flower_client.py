"""
Distributed Flower gRPC Client for a Hospital Node.
Connects an individual hospital node (A, B, C, D, or E) to the central Flower aggregator.

Usage:
    python ml/distributed_flower_client.py --hospital A --server 127.0.0.1:8080
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
from ml.data_loader import load_all_hospitals_data, DEFAULT_FEATURES, DEFAULT_TARGET
from ml.flower_client import HospitalNumPyClient
from ml.privacy_engine import FederatedPrivacyManager


def main():
    parser = argparse.ArgumentParser(description="Flower Hospital Client")
    parser.add_argument("--hospital", type=str, required=True, choices=["A", "B", "C", "D", "E"], help="Hospital ID")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Flower server gRPC address")
    parser.add_argument("--epochs", type=int, default=3, help="Local epochs per federated round")
    parser.add_argument("--lr", type=float, default=0.01, help="Local learning rate")
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy (DP-SGD)")
    parser.add_argument("--noise-multiplier", type=float, default=0.8, help="DP noise multiplier")
    parser.add_argument("--clipping-norm", type=float, default=1.0, help="DP per-example gradient clipping norm")
    args = parser.parse_args()

    ml_dir = Path(__file__).resolve().parent
    dataset_dir = ml_dir / "dataset"

    # Load hospital dataset
    hospital_data, scaler_X, scaler_y = load_all_hospitals_data(
        dataset_dir=dataset_dir,
        features=DEFAULT_FEATURES,
        target_col=DEFAULT_TARGET,
    )

    h_data = hospital_data[args.hospital]

    privacy_manager = None
    if args.dp:
        print(f"[*] Differential Privacy ENABLED on Hospital {args.hospital} (Noise={args.noise_multiplier}, Clip={args.clipping_norm})")
        privacy_manager = FederatedPrivacyManager(
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.clipping_norm,
        )

    client = HospitalNumPyClient(
        hospital_id=args.hospital,
        X_train=h_data["X_train"],
        y_train=h_data["y_train"],
        X_test=h_data["X_test"],
        y_test=h_data["y_test"],
        scaler_y=scaler_y,
        input_dim=len(DEFAULT_FEATURES),
        learning_rate=args.lr,
        epochs_per_round=args.epochs,
        privacy_manager=privacy_manager,
    )

    print(f"[*] Hospital {args.hospital} connecting to Flower Server at {args.server}...")
    print(f"[*] Local training set: {h_data['n_train_samples']} samples. Held-out test set: {h_data['n_test_samples']} samples.")
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
