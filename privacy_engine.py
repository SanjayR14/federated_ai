"""
Differential Privacy Engine using Opacus.
Provides per-example gradient clipping, calibrated Gaussian noise injection (DP-SGD),
and formal Rényi Differential Privacy (RDP) accounting for (epsilon, delta)-DP guarantees.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant


class FederatedPrivacyManager:
    """
    Manages differential privacy parameters, DP-SGD training on local client nodes,
    and cumulative privacy budget accounting across federated rounds.
    """

    def __init__(
        self,
        noise_multiplier: float = 0.8,
        max_grad_norm: float = 1.0,
        target_delta: float = 1e-5,
    ):
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.target_delta = float(target_delta)
        self.accountant = RDPAccountant()
        self.total_steps = 0

    def make_private_client(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]:
        """
        Attaches Opacus PrivacyEngine to a client's local training loop:
        Applies per-sample gradient clipping and calibrated Gaussian noise addition.
        """
        engine = PrivacyEngine(secure_mode=False)
        res = engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )
        private_model = res[0]
        private_optimizer = res[1]
        private_loader = res[-1]
        return private_model, private_optimizer, private_loader, engine

    def record_round_steps(self, steps: int, sample_rate: float) -> None:
        """
        Accumulate privacy accounting steps across federated training rounds.
        """
        for _ in range(steps):
            self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=sample_rate)
        self.total_steps += steps

    def get_privacy_budget(self) -> Dict[str, Any]:
        """
        Compute and return the formal (epsilon, delta) privacy budget.
        """
        if self.total_steps == 0:
            epsilon = 0.0
        else:
            epsilon = float(self.accountant.get_epsilon(delta=self.target_delta))

        return {
            "mechanism": "DP-SGD (Per-example gradient clipping + Gaussian noise)",
            "privacy_guarantee": f"({round(epsilon, 3)}, {self.target_delta})-Differential Privacy",
            "epsilon": round(epsilon, 4),
            "delta": self.target_delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
            "total_dp_steps": self.total_steps,
            "accounting_method": "Rényi Differential Privacy (RDP) via Opacus",
            "scope": "Record-level Central/Distributed DP-SGD",
            "note": "Raw patient samples stay on client. Bounded gradient sensitivity ensures minimal information leakage regarding any individual record.",
        }
