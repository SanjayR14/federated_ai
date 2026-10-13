"""
PyTorch Federated Parameter Aggregator (FedAvg).
Computes sample-weighted parameter averaging across hospital local weight checkpoints.
"""

from pathlib import Path
import torch

def aggregate_fedavg(weight_files, sample_counts=None):
    if sample_counts is not None and len(sample_counts) == len(weight_files):
        total = float(sum(sample_counts))
        weights = [float(s) / total if total > 0 else 1.0 / len(weight_files) for s in sample_counts]
    else:
        weights = [1.0 / len(weight_files)] * len(weight_files)

    state_dicts = [torch.load(f, map_location="cpu", weights_only=True) for f in weight_files]
    global_weights = {}

    for key in state_dicts[0].keys():
        global_weights[key] = torch.zeros_like(state_dicts[0][key]).float()
        for sd, w in zip(state_dicts, weights):
            global_weights[key] += sd[key].float() * w

    return global_weights

if __name__ == "__main__":
    ml_dir = Path(__file__).resolve().parent
    files = [ml_dir / f"weights_{h}.pth" for h in ["A", "B", "C", "D", "E"]]
    existing_files = [f for f in files if f.exists()]

    if len(existing_files) >= 2:
        print(f"Aggregating weights from {len(existing_files)} hospitals...")
        global_model_weights = aggregate_fedavg(existing_files)
        torch.save(global_model_weights, ml_dir / "Global_Model.pth")
        print("Successfully created Global_Model.pth!")
    else:
        print("Run ml/run_flower_federated_system.py to train and aggregate all hospital nodes via Flower.")