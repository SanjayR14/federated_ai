"""
Flower Hospital Client Implementation.
Subclasses flwr.client.NumPyClient to provide genuine federated training on private local datasets.
"""

from typing import Dict, List, Tuple, Any, Optional
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

from ml.model import (
    FederatedHealthNet,
    get_parameters,
    set_parameters,
    quantize_fp16,
    dequantize_fp16,
    serialize_parameters,
)
from ml.metrics import compute_regression_metrics
from ml.privacy_engine import FederatedPrivacyManager


class HospitalNumPyClient(fl.client.NumPyClient):
    """
    Flower Client representing a single hospital node (e.g., Hospital A, B, C, D, E).
    Enforces strict patient record isolation: raw data never leaves the client instance.
    """

    def __init__(
        self,
        hospital_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler_y: Any,
        input_dim: int = 5,
        learning_rate: float = 0.01,
        epochs_per_round: int = 3,
        batch_size: int = 32,
        privacy_manager: Optional[FederatedPrivacyManager] = None,
        use_quantization: bool = False,
    ):
        self.hospital_id = str(hospital_id)
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.epochs_per_round = epochs_per_round
        self.batch_size = batch_size
        self.scaler_y = scaler_y
        self.privacy_manager = privacy_manager
        self.use_quantization = use_quantization

        # Client-private data storage
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Local neural network instance
        self.model = FederatedHealthNet(input_dim=self.input_dim)

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Extract current local model parameters."""
        params = get_parameters(self.model)
        if self.use_quantization:
            params = quantize_fp16(params)
        return params

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """
        Train local model using received global parameters:
        1. Set local model weights to global parameters.
        2. Run local PyTorch optimization (or DP-SGD via Opacus if enabled).
        3. Return updated weights, local sample count, and training telemetry.
        """
        start_time = time.time()

        # Handle dequantization if needed
        if self.use_quantization:
            parameters = dequantize_fp16(parameters)

        # Fresh model instance per round to avoid duplicate Opacus hooks
        local_model = FederatedHealthNet(input_dim=self.input_dim)
        set_parameters(local_model, parameters)
        local_model.train()

        # Dynamic config overrides
        lr = float(config.get("lr", self.learning_rate))
        epochs = int(config.get("epochs", self.epochs_per_round))

        dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32).view(-1, 1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        if self.privacy_manager is not None:
            # DP-SGD via Opacus
            p_model, p_optim, p_loader, p_engine = self.privacy_manager.make_private_client(
                local_model, optimizer, loader
            )
            total_loss = 0.0
            num_batches = 0
            for _ in range(epochs):
                for batch_x, batch_y in p_loader:
                    p_optim.zero_grad()
                    out = p_model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    p_optim.step()
                    total_loss += float(loss.item())
                    num_batches += 1

            # Accumulate privacy steps in accountant
            sample_rate = self.batch_size / len(self.X_train)
            self.privacy_manager.record_round_steps(num_batches, sample_rate)
            avg_loss = total_loss / max(num_batches, 1)
            updated_params = get_parameters(local_model)
        else:
            # Standard SGD / Adam
            total_loss = 0.0
            num_batches = 0
            for _ in range(epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    out = local_model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.item())
                    num_batches += 1
            avg_loss = total_loss / max(num_batches, 1)
            updated_params = get_parameters(local_model)
        payload_bytes = len(serialize_parameters(updated_params))

        if self.use_quantization:
            updated_params = quantize_fp16(updated_params)
            payload_bytes = len(serialize_parameters(updated_params))

        duration = time.time() - start_time

        metrics = {
            "hospital_id": self.hospital_id,
            "train_loss": round(avg_loss, 4),
            "training_duration": round(duration, 3),
            "payload_bytes": payload_bytes,
        }

        return updated_params, len(self.X_train), metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Evaluate global model on this hospital's held-out test split.
        Returns loss, test sample count, and regression metrics (RMSE, MAE, R², MAPE).
        """
        if self.use_quantization:
            parameters = dequantize_fp16(parameters)

        set_parameters(self.model, parameters)
        self.model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(self.X_test, dtype=torch.float32)
            preds_scaled = self.model(x_tensor).numpy().flatten()

        # Invert target scaling to report metrics on true clinical scale (%)
        preds = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        reg_metrics = compute_regression_metrics(self.y_test, preds)
        test_mse = float(reg_metrics["RMSE"] ** 2)

        return (
            test_mse,
            len(self.X_test),
            {
                "hospital_id": self.hospital_id,
                "RMSE": reg_metrics["RMSE"],
                "MAE": reg_metrics["MAE"],
                "R2": reg_metrics["R2"],
                "MAPE": reg_metrics["MAPE"],
            },
        )
