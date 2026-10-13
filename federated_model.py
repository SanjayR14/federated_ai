"""Shared federated ensemble used by training and inference."""

import numpy as np
from sklearn.metrics import r2_score


class FederatedEnsemble:
    """Weighted average of local hospital regressors."""

    def __init__(self, models, weights):
        self.models = models
        self.weights = np.asarray(weights, dtype=float)
        total = self.weights.sum()
        self.weights = self.weights / total if total > 0 else self.weights

    def fit(self, X, y):
        return self

    def predict(self, X):
        preds = np.zeros(X.shape[0], dtype=float)
        for model, weight in zip(self.models, self.weights):
            preds += weight * model.predict(X)
        return preds

    def score(self, X, y):
        return r2_score(y, self.predict(X))
