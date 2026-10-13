"""
Custom Flower FedAvg Strategy for Multi-Hospital Federated Healthcare.
Extends flwr.server.strategy.FedAvg with metric aggregation and network byte accounting.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy


class HospitalFedAvgStrategy(fl.server.strategy.FedAvg):
    """
    Hospital Federated Averaging Strategy.
    Performs sample-weighted FedAvg parameter aggregation and tracks communication volume.
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 5,
        min_evaluate_clients: int = 5,
        min_available_clients: int = 5,
        initial_parameters: Optional[Parameters] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
        )
        self.total_uplink_bytes = 0
        self.total_downlink_bytes = 0
        self.round_history: List[Dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights using sample-weighted FedAvg:
        w_{t+1} = sum_{k=1}^K (n_k / n) * w_{t+1}^k
        """
        if not results:
            return None, {}

        # Track uplink communication from clients
        round_uplink = sum(r.metrics.get("payload_bytes", 0) for _, r in results)
        self.total_uplink_bytes += round_uplink

        # Use Flower's built-in FedAvg aggregation logic
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Model broadcast to clients for next round
            downlink_size = len(aggregated_parameters.tensors[0]) if aggregated_parameters.tensors else 0
            round_downlink = downlink_size * len(results)
            self.total_downlink_bytes += round_downlink

        # Collect train losses
        losses = [float(r.metrics.get("train_loss", 0.0)) for _, r in results]
        avg_loss = float(np.mean(losses)) if losses else 0.0

        round_info = {
            "round": server_round,
            "participating_clients": len(results),
            "train_loss": round(avg_loss, 4),
            "uplink_bytes": round_uplink,
        }
        self.round_history.append(round_info)

        metrics["avg_train_loss"] = avg_loss
        return aggregated_parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate evaluation metrics across hospital client test splits:
        Computes sample-weighted RMSE, MAE, R², and MAPE.
        """
        if not results:
            return None, {}

        total_samples = sum(r.num_examples for _, r in results)

        weighted_rmse = sum(r.metrics.get("RMSE", 0.0) * r.num_examples for _, r in results) / total_samples
        weighted_mae = sum(r.metrics.get("MAE", 0.0) * r.num_examples for _, r in results) / total_samples
        weighted_r2 = sum(r.metrics.get("R2", 0.0) * r.num_examples for _, r in results) / total_samples
        weighted_mape = sum(r.metrics.get("MAPE", 0.0) * r.num_examples for _, r in results) / total_samples
        weighted_loss = sum(r.loss * r.num_examples for _, r in results) / total_samples

        aggregated_metrics = {
            "RMSE": round(float(weighted_rmse), 4),
            "MAE": round(float(weighted_mae), 4),
            "R2": round(float(weighted_r2), 4),
            "MAPE": round(float(weighted_mape), 2),
        }

        return float(weighted_loss), aggregated_metrics
