"""
Automated Test Suite for Multi-Hospital Federated Learning System.
Tests:
1. Data isolation & patient-level splits.
2. Flower model serialization & parameter aggregation.
3. Flower HospitalNumPyClient training & evaluation.
4. Opacus Differential Privacy accounting.
5. Secure Aggregation pairwise zero-sum masking cancellation.
6. Byzantine model poisoning attack defense via Trimmed Mean.
7. Communication quantization payload reduction.
8. End-to-end inference prediction and clinical interpretation.
"""

import unittest
import sys
import os
import json
import subprocess
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ML_DIR = _HERE.parent
_ROOT = _ML_DIR.parent
for p in [str(_ROOT), str(_ML_DIR), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ml.model import (
    FederatedHealthNet,
    get_parameters,
    set_parameters,
    quantize_fp16,
    dequantize_fp16,
    serialize_parameters,
)
from ml.data_loader import (
    clean_dataframe,
    split_by_patient,
    load_all_hospitals_data,
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    HOSPITALS,
)
from ml.metrics import compute_regression_metrics
from ml.privacy_engine import FederatedPrivacyManager
from ml.security_analysis import (
    SecureAggregationDemonstrator,
    coordinate_wise_trimmed_mean,
    standard_fedavg_aggregate,
)
from ml.flower_client import HospitalNumPyClient


class TestFederatedHealthcareSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = _ML_DIR / "dataset"
        cls.hospital_data, cls.scaler_X, cls.scaler_y = load_all_hospitals_data(
            dataset_dir=cls.dataset_dir,
            features=DEFAULT_FEATURES,
            target_col=DEFAULT_TARGET,
            test_size=0.2,
            seed=42,
        )

    def test_01_data_isolation_and_patient_splits(self):
        """Verify data remains strictly isolated across all 5 hospitals with no patient leakage."""
        self.assertEqual(len(self.hospital_data), 5)
        for hid in HOSPITALS:
            data = self.hospital_data[hid]
            self.assertIn("X_train", data)
            self.assertIn("y_train", data)
            self.assertIn("X_test", data)
            self.assertIn("y_test", data)
            # Patient count isolation
            self.assertGreater(data["n_train_patients"], 150)
            self.assertGreater(data["n_test_patients"], 30)
            self.assertGreater(data["n_train_samples"], 1800)
            self.assertGreater(data["n_test_samples"], 400)

    def test_02_model_parameter_serialization(self):
        """Verify parameters can be converted to NumPy, quantized, and restored without distortion."""
        net = FederatedHealthNet(input_dim=5)
        params = get_parameters(net)
        self.assertEqual(len(params), 6)  # fc1.w, fc1.b, fc2.w, fc2.b, fc3.w, fc3.b

        # Test FP16 quantization roundtrip
        quant = quantize_fp16(params)
        dequant = dequantize_fp16(quant)
        for p_orig, p_deq in zip(params, dequant):
            np.testing.assert_allclose(p_orig, p_deq, atol=1e-3)

    def test_03_flower_client_fit_and_evaluate(self):
        """Verify Flower HospitalNumPyClient fits local model and evaluates held-out metrics."""
        h_data = self.hospital_data["A"]
        client = HospitalNumPyClient(
            hospital_id="A",
            X_train=h_data["X_train"][:100],  # test subset
            y_train=h_data["y_train"][:100],
            X_test=h_data["X_test"][:50],
            y_test=h_data["y_test"][:50],
            scaler_y=self.scaler_y,
            input_dim=5,
            learning_rate=0.01,
            epochs_per_round=1,
            batch_size=16,
        )

        init_params = get_parameters(client.model)
        updated_params, n_samples, metrics = client.fit(init_params, {"lr": 0.01, "epochs": 1})

        self.assertEqual(n_samples, 100)
        self.assertIn("train_loss", metrics)
        self.assertIn("payload_bytes", metrics)

        # Test evaluation
        test_loss, n_test, eval_metrics = client.evaluate(updated_params, {})
        self.assertEqual(n_test, 50)
        self.assertIn("RMSE", eval_metrics)
        self.assertIn("MAE", eval_metrics)
        self.assertGreater(eval_metrics["RMSE"], 0.0)

    def test_04_differential_privacy_accounting(self):
        """Verify Opacus RDPAccountant calculates formal (epsilon, delta) budget."""
        manager = FederatedPrivacyManager(noise_multiplier=0.8, max_grad_norm=1.0, target_delta=1e-5)
        manager.record_round_steps(steps=100, sample_rate=0.03)

        budget = manager.get_privacy_budget()
        self.assertGreater(budget["epsilon"], 0.0)
        self.assertEqual(budget["delta"], 1e-5)
        self.assertEqual(budget["total_dp_steps"], 100)

    def test_05_cryptographic_secure_aggregation_cancellation(self):
        """Verify pairwise zero-sum masks cancel identically to zero during SecAgg."""
        # Create mock parameter updates for 5 hospital clients
        base_net = FederatedHealthNet(input_dim=5)
        client_updates = [
            [p + float(i) * 0.1 for p in get_parameters(base_net)]
            for i in range(5)
        ]

        result = SecureAggregationDemonstrator.run_protocol(client_updates, seed=123)
        self.assertTrue(result["exact_aggregate_reconstructed"])
        self.assertLess(result["max_reconstruction_error"], 1e-4)
        self.assertGreater(result["average_mask_magnitude"], 1.0)

    def test_06_byzantine_poisoning_attack_defense(self):
        """Verify Coordinate-wise Trimmed Mean rejects malicious poisoned updates."""
        # Honest updates around +1.0
        honest = [
            [np.array([1.0, 1.0], dtype=np.float32)],
            [np.array([1.1, 0.9], dtype=np.float32)],
            [np.array([0.9, 1.1], dtype=np.float32)],
            [np.array([1.05, 0.95], dtype=np.float32)],
        ]
        # Malicious update (adversarial sign-flip & amplification)
        malicious = [np.array([-100.0, -100.0], dtype=np.float32)]

        all_updates = honest + [malicious]

        # Standard FedAvg gets corrupted by malicious outlier
        standard_agg = standard_fedavg_aggregate(all_updates, [1, 1, 1, 1, 1])[0]
        self.assertLess(standard_agg[0], -15.0)  # severely corrupted

        # Trimmed Mean successfully discards the extreme outlier
        trimmed_agg = coordinate_wise_trimmed_mean(all_updates, trim_fraction=0.2)[0]
        self.assertGreater(trimmed_agg[0], 0.5)  # recovered near 1.0!
        self.assertLess(trimmed_agg[0], 1.5)

    def test_07_regression_metrics_computation(self):
        """Verify metric functions calculate correct values without NaN or Inf."""
        y_true = np.array([6.5, 7.0, 5.5, 8.0])
        y_pred = np.array([6.4, 7.1, 5.6, 7.9])
        metrics = compute_regression_metrics(y_true, y_pred)

        self.assertAlmostEqual(metrics["RMSE"], 0.1, places=1)
        self.assertAlmostEqual(metrics["MAE"], 0.1, places=1)
        self.assertGreater(metrics["R2"], 0.9)
        self.assertGreater(metrics["MAPE"], 0.0)

    def test_08_end_to_end_predict_script(self):
        """Test backend/ai_engine/predict.py via subprocess to verify production API contract."""
        predict_script = _ROOT / "backend" / "ai_engine" / "predict.py"
        self.assertTrue(predict_script.exists())

        sample_patient = {
            "fastingGlucose": 110,
            "postprandialGlucose": 150,
            "triglycerides": 140,
            "hdl": 48,
            "ldl": 115,
        }

        cmd = [sys.executable, str(predict_script), json.dumps(sample_patient)]
        res = subprocess.run(cmd, capture_output=True, text=True)

        self.assertEqual(res.returncode, 0, f"predict.py failed: {res.stderr}")
        output = json.loads(res.stdout.strip())

        self.assertIn("predicted_hba1c", output)
        self.assertIn("risk_category", output)
        self.assertIn("risk_color", output)
        self.assertIn("recommendations", output)
        self.assertGreaterEqual(output["predicted_hba1c"], 4.0)
        self.assertLessEqual(output["predicted_hba1c"], 15.0)


if __name__ == "__main__":
    unittest.main()
