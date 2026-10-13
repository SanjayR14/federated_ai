"""
Standardized Regression Evaluation Metrics.
Calculates RMSE, MAE, R², and MAPE.
"""

from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics on original unscaled units:
    - RMSE: Root Mean Squared Error
    - MAE: Mean Absolute Error
    - R²: Coefficient of Determination
    - MAPE: Mean Absolute Percentage Error (%)
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    # Guard against NaN/Inf
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "MAPE": 0.0}

    y_t = y_true[mask]
    y_p = y_pred[mask]

    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_t, y_p))

    # Variance of true target
    var_y = float(np.var(y_t))
    if var_y > 1e-9:
        r2 = float(r2_score(y_t, y_p))
    else:
        r2 = 0.0

    mape = float(np.mean(np.abs((y_t - y_p) / (np.abs(y_t) + 1e-8))) * 100.0)

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }
