"""
Federated HbA1c Regression Pipeline — tree-based models (XGBoost / RandomForest).
Regression metrics only: RMSE, MAE, R², MAPE.
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from federated_model import FederatedEnsemble
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
ML_DIR = Path(__file__).resolve().parent
HOSPITALS = ["A", "B", "C", "D", "E"]
TARGET_COL = "HbA1c_%"
FEATURE_COLUMNS = [
    "Fasting_Glucose",
    "Postprandial_Glucose",
    "Triglycerides",
    "HDL",
    "LDL",
]
LOG1P_COLUMNS = ["Triglycerides", "Postprandial_Glucose"]
DISTRIBUTION_COLUMNS = [
    "HbA1c_%",
    "Fasting_Glucose",
    "Postprandial_Glucose",
    "Triglycerides",
]
HBA1C_MIN, HBA1C_MAX = 4.0, 15.0
RANDOM_STATE = 42
INVALID_TOKENS = ["?", "NA", "N/A", "", "nan", "NaN"]


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def apply_log1p(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = np.log1p(np.maximum(out[col].astype(float), 0.0))
    return out


def soft_clean_dataframe(df):
    df = df.copy()
    df = df.replace(INVALID_TOKENS, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in FEATURE_COLUMNS + [TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in FEATURE_COLUMNS:
        if col in df.columns:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
        df = df.dropna(subset=[TARGET_COL])

    return df


def load_hospital_dataframe(hospital_id):
    hospital_file = DATASET_DIR / f"Hospital_{hospital_id}_Master_Records.csv"
    df = pd.read_csv(hospital_file)
    print(f"\n{'=' * 70}")
    print(f"HOSPITAL {hospital_id}: Loading and Cleaning Data")
    print(f"{'=' * 70}")
    print(f"Initial records: {len(df)}")

    df = soft_clean_dataframe(df)

    if TARGET_COL not in df.columns:
        alt = [c for c in df.columns if c.startswith("HbA1c")]
        if not alt:
            print(f"ERROR: No HbA1c column found for Hospital {hospital_id}")
            return None
        target_col = alt[0]
    else:
        target_col = TARGET_COL

    print(f"\n{target_col} distribution (before range filter):")
    print(df[target_col].describe())

    invalid_mask = (df[target_col] < HBA1C_MIN) | (df[target_col] > HBA1C_MAX)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(
            f"Removing {invalid_count} rows with medically unrealistic "
            f"{target_col} (< {HBA1C_MIN} or > {HBA1C_MAX})"
        )
        df = df.loc[~invalid_mask].copy()

    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns]
    for feat in missing_feats:
        print(f"  Warning: feature '{feat}' not found")

    df = apply_log1p(df, LOG1P_COLUMNS)
    print(f"Records after soft cleaning: {len(df)}")
    print(f"Features used: {available}")
    return df, available, target_col


def plot_feature_distributions(frames):
    combined = pd.concat(frames, ignore_index=True)
    cols = [c for c in DISTRIBUTION_COLUMNS if c in combined.columns]
    if not cols:
        print("No distribution columns available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        series = pd.to_numeric(combined[col], errors="coerce").dropna()
        ax.hist(series, bins=40, color="#2E86AB", edgecolor="white", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
    for ax in axes[len(cols) :]:
        ax.axis("off")

    plt.suptitle("Feature Distributions (All Hospitals)", fontsize=14)
    plt.tight_layout()
    out_path = ML_DIR / "feature_distributions.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved feature distributions: {out_path}")


def create_model_candidates():
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if HAS_XGBOOST:
        candidates["XGBoostRegressor"] = XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
        )
    else:
        print("XGBoost not installed — using RandomForestRegressor as primary tree model.")
    return candidates


def select_best_model(X_train, y_train, X_val, y_val):
    print("\nBaseline model comparison (validation set):")
    print(f"{'Model':<28} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'MAPE':>10}")
    print("-" * 72)

    priority = {
        "XGBoostRegressor": 0,
        "RandomForestRegressor": 1,
        "LinearRegression": 2,
    }
    comparison = {}
    ranked = []

    for name, model in create_model_candidates().items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, val_pred)
        comparison[name] = metrics
        ranked.append((name, model, metrics))
        print(
            f"{name:<28} {metrics['RMSE']:10.4f} {metrics['MAE']:10.4f} "
            f"{metrics['R2']:10.4f} {metrics['MAPE']:9.2f}%"
        )

    best_by_rmse = min(ranked, key=lambda item: item[2]["RMSE"])
    preferred = "XGBoostRegressor" if HAS_XGBOOST else "RandomForestRegressor"
    best_name, best_model, best_metrics = next(
        item for item in ranked if item[0] == preferred
    )

    print(
        f"\nBest validation model (by RMSE): {best_by_rmse[0]} "
        f"(RMSE = {best_by_rmse[2]['RMSE']:.4f}, R² = {best_by_rmse[2]['R2']:.4f})"
    )
    print(
        f"Deployed model (tree-based default): {best_name} "
        f"(RMSE = {best_metrics['RMSE']:.4f}, R² = {best_metrics['R2']:.4f})"
    )
    return best_model, best_name, comparison


def train_hospital_model(hospital_id, df, features, target_col, global_scaler=None):
    print(f"\n{'=' * 70}")
    print(f"HOSPITAL {hospital_id}: Training Tree-Based Model")
    print(f"{'=' * 70}")

    X = df[features].values.astype(float)
    y = df[target_col].values.astype(float)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
    )

    print(
        f"  Train: {len(X_train)} ({len(X_train) / len(X) * 100:.1f}%) | "
        f"Val: {len(X_val)} ({len(X_val) / len(X) * 100:.1f}%) | "
        f"Test: {len(X_test)} ({len(X_test) / len(X) * 100:.1f}%)"
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model, model_name, baseline_comparison = select_best_model(
        X_train_s, y_train, X_val_s, y_val
    )

    X_trainval_s = scaler.transform(np.vstack([X_train, X_val]))
    y_trainval = np.concatenate([y_train, y_val])
    model.fit(X_trainval_s, y_trainval)

    y_pred_test = model.predict(X_test_s)
    test_metrics = compute_metrics(y_test, y_pred_test)

    model_path = ML_DIR / f"model_{hospital_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "model_name": model_name,
                "scaler": scaler,
                "features": features,
                "log1p_columns": LOG1P_COLUMNS,
            },
            f,
        )

    metrics = {
        "Hospital": hospital_id,
        "Model": model_name,
        "Train_Samples": len(X_train),
        "Val_Samples": len(X_val),
        "Test_Samples": len(X_test),
        "Baseline_Comparison": baseline_comparison,
        **{f"Test_{k}": v for k, v in test_metrics.items()},
        "Features": features,
    }

    print(f"\nHospital {hospital_id} test metrics ({model_name}):")
    print(f"  RMSE: {test_metrics['RMSE']:.4f}")
    print(f"  MAE:  {test_metrics['MAE']:.4f}")
    print(f"  R²:   {test_metrics['R2']:.4f}")
    print(f"  MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"  Saved: {model_path.name}")

    bundle = {
        "hospital_id": hospital_id,
        "model": model,
        "model_name": model_name,
        "scaler": scaler,
        "X_test": X_test_s,
        "y_test": y_test,
        "train_samples": len(X_train),
    }
    return metrics, bundle


def fit_global_scaler_from_training_data(hospital_frames, features):
    """Fit scaler on all hospital training rows (log1p + median imputation already applied)."""
    parts = []
    for df, feats, target_col in hospital_frames:
        X = df[feats].values.astype(float)
        y = df[target_col].values.astype(float)
        X_train, _, _, _ = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE
        )
        parts.append(X_train)
    combined = np.vstack(parts)
    scaler = StandardScaler()
    scaler.fit(combined)
    return scaler


def weighted_federated_ensemble(hospital_bundles):
    print(f"\n{'=' * 70}")
    print("WEIGHTED FEDERATED ENSEMBLE (prediction aggregation)")
    print(f"{'=' * 70}")

    models = [b["model"] for b in hospital_bundles]
    train_counts = [b["train_samples"] for b in hospital_bundles]
    total = sum(train_counts)
    weights = [c / total for c in train_counts]

    for hid, w, n in zip(HOSPITALS, weights, train_counts):
        print(f"  Hospital {hid}: weight={w:.3f} ({n} train samples)")

    ensemble = FederatedEnsemble(models, weights)
    global_path = ML_DIR / "Global_Model.pkl"
    with open(global_path, "wb") as f:
        pickle.dump(
            {
                "ensemble": ensemble,
                "models": models,
                "weights": weights,
                "model_names": [b["model_name"] for b in hospital_bundles],
                "features": FEATURE_COLUMNS,
                "log1p_columns": LOG1P_COLUMNS,
            },
            f,
        )
    print(f"Global federated model saved: {global_path.name}")
    return ensemble, weights


def evaluate_local_vs_global(hospital_bundles, global_ensemble):
    print(f"\n{'=' * 70}")
    print("FEDERATED EVALUATION: Local vs Global (side-by-side)")
    print(f"{'=' * 70}")

    rows = []
    all_y_true, all_y_local, all_y_global = [], [], []

    header = (
        f"{'Hospital':<10} {'Model':<8} {'Samples':>8} "
        f"{'Local_RMSE':>11} {'Glob_RMSE':>11} {'Local_R2':>10} {'Glob_R2':>10} "
        f"{'Local_MAPE':>11} {'Glob_MAPE':>11}"
    )
    print(header)
    print("-" * len(header))

    for bundle in hospital_bundles:
        hid = bundle["hospital_id"]
        X_test = bundle["X_test"]
        y_test = bundle["y_test"]
        local_pred = bundle["model"].predict(X_test)
        global_pred = global_ensemble.predict(X_test)

        local_m = compute_metrics(y_test, local_pred)
        global_m = compute_metrics(y_test, global_pred)

        print(
            f"{hid:<10} {bundle['model_name'][:8]:<8} {len(y_test):>8} "
            f"{local_m['RMSE']:11.4f} {global_m['RMSE']:11.4f} "
            f"{local_m['R2']:10.4f} {global_m['R2']:10.4f} "
            f"{local_m['MAPE']:10.2f}% {global_m['MAPE']:10.2f}%"
        )

        rows.append(
            {
                "Hospital": hid,
                "Local_Model": bundle["model_name"],
                "Test_Samples": len(y_test),
                "Local": local_m,
                "Global": global_m,
            }
        )
        all_y_true.extend(y_test)
        all_y_local.extend(local_pred)
        all_y_global.extend(global_pred)

    combined_local = compute_metrics(all_y_true, all_y_local)
    combined_global = compute_metrics(all_y_true, all_y_global)

    print(f"\n{'=' * 70}")
    print("COMBINED TEST METRICS (all hospitals)")
    print(f"{'=' * 70}")
    print(
        f"{'Scope':<12} {'RMSE':>10} {'MAE':>10} {'R2':>10} {'MAPE':>10}"
    )
    print("-" * 54)
    for label, metrics in [
        ("Local", combined_local),
        ("Global", combined_global),
    ]:
        print(
            f"{label:<12} {metrics['RMSE']:10.4f} {metrics['MAE']:10.4f} "
            f"{metrics['R2']:10.4f} {metrics['MAPE']:9.2f}%"
        )

    return rows, combined_local, combined_global


def plot_permutation_importance(model, X, y, feature_names):
    print(f"\n{'=' * 70}")
    print("PERMUTATION FEATURE IMPORTANCE (global model)")
    print(f"{'=' * 70}")

    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=-1,
    )

    order = np.argsort(result.importances_mean)
    sorted_features = [feature_names[i] for i in order]
    sorted_scores = result.importances_mean[order]

    plt.figure(figsize=(9, 5))
    plt.barh(sorted_features, sorted_scores, color="#A23B72")
    plt.xlabel("Permutation importance (mean decrease in R²)")
    plt.title("Global Federated Model — Feature Importance")
    plt.tight_layout()
    out_path = ML_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")

    for feat, score in zip(sorted_features, sorted_scores):
        print(f"  {feat}: {score:.4f}")

    return dict(zip(feature_names, result.importances_mean.tolist()))


def main():
    print("\n" + "=" * 70)
    print("FEDERATED HbA1c REGRESSION — TREE-BASED MODELS")
    print("=" * 70)
    print("Metrics: RMSE, MAE, R², MAPE (regression only — no accuracy)")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Log1p applied to: {LOG1P_COLUMNS}")
    print(f"Primary tree model: {'XGBoostRegressor' if HAS_XGBOOST else 'RandomForestRegressor'}")
    print("=" * 70)

    raw_frames = []
    hospital_frames = []
    hospital_metrics = []
    hospital_bundles = []

    for hospital_id in HOSPITALS:
        result = load_hospital_dataframe(hospital_id)
        if not result:
            continue
        df, features, target_col = result
        raw_frames.append(df[DISTRIBUTION_COLUMNS].copy() if all(c in df.columns for c in DISTRIBUTION_COLUMNS) else df)
        hospital_frames.append((df, features, target_col))

    plot_feature_distributions(raw_frames)

    if hospital_frames:
        pooled = pd.concat([df for df, _, _ in hospital_frames], ignore_index=True)
        print(f"\nFeature correlation with {TARGET_COL} (leakage-free features only):")
        corr = pooled[FEATURE_COLUMNS + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
        print(corr.sort_values(ascending=False).to_string())

    global_scaler = fit_global_scaler_from_training_data(hospital_frames, FEATURE_COLUMNS)
    scaler_path = ML_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(global_scaler, f)
    print(f"\nSaved global scaler: {scaler_path.name}")

    feature_cols_path = ML_DIR / "feature_columns.json"
    with open(feature_cols_path, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    print(f"Saved feature order: {feature_cols_path.name}")

    for (df, features, target_col), hospital_id in zip(hospital_frames, HOSPITALS):
        metrics, bundle = train_hospital_model(
            hospital_id, df, features, target_col
        )
        hospital_metrics.append(metrics)
        hospital_bundles.append(bundle)

        test_path = ML_DIR / f"test_data_{hospital_id}.npy"
        np.save(
            test_path,
            {"X_test": bundle["X_test"], "y_test": bundle["y_test"]},
            allow_pickle=True,
        )

    global_ensemble, weights = weighted_federated_ensemble(hospital_bundles)
    comparison_rows, local_combined, global_combined = evaluate_local_vs_global(
        hospital_bundles, global_ensemble
    )

    X_imp = np.vstack([b["X_test"] for b in hospital_bundles])
    y_imp = np.concatenate([b["y_test"] for b in hospital_bundles])
    importance = plot_permutation_importance(
        global_ensemble, X_imp, y_imp, FEATURE_COLUMNS
    )

    results_summary = {
        "Pipeline": "Federated HbA1c Regression — Tree-Based Models",
        "Model_Type": "XGBoost/RandomForest ensemble (weighted federated predictions)",
        "Features": FEATURE_COLUMNS,
        "Log1p_Columns": LOG1P_COLUMNS,
        "Hospitals": HOSPITALS,
        "Metrics_Type": "Regression (RMSE, MAE, R², MAPE)",
        "Local_Models": hospital_metrics,
        "Local_vs_Global_Comparison": comparison_rows,
        "Combined_Local_Metrics": local_combined,
        "Combined_Global_Metrics": global_combined,
        "Permutation_Importance": importance,
        "Federated_Weights": dict(zip(HOSPITALS, weights)),
    }

    results_path = ML_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    pooled = pd.concat([df for df, _, _ in hospital_frames], ignore_index=True)
    training_reference = {
        "hba1c_mean": float(pooled[TARGET_COL].mean()),
        "hba1c_std": float(pooled[TARGET_COL].std()),
        "feature_means": {
            col: float(pooled[col].mean()) for col in FEATURE_COLUMNS if col in pooled.columns
        },
    }
    with open(ML_DIR / "training_reference.json", "w") as f:
        json.dump(training_reference, f, indent=2)

    federated_metrics = {
        "local_vs_global": comparison_rows,
        "combined_local": local_combined,
        "combined_global": global_combined,
        "hospital_metrics": [
            {
                "hospital": m["Hospital"],
                "model": m["Model"],
                "test_rmse": m.get("Test_RMSE"),
                "test_mape": m.get("Test_MAPE"),
                "test_r2": m.get("Test_R2"),
            }
            for m in hospital_metrics
        ],
        "feature_importance": importance,
        "federated_weights": dict(zip(HOSPITALS, [float(w) for w in weights])),
    }
    with open(ML_DIR / "federated_metrics.json", "w") as f:
        json.dump(federated_metrics, f, indent=2)

    clinical_summary = {
        "system": "Federated Intelligent Healthcare Analytics and Clinical Decision Support",
        "risk_categories": [
            {"range": "< 5.7", "label": "Normal", "color": "#22c55e"},
            {"range": "5.7 – 6.4", "label": "Prediabetes", "color": "#eab308"},
            {"range": "6.5 – 8.4", "label": "Diabetes", "color": "#f97316"},
            {"range": ">= 8.5", "label": "High Risk Diabetes", "color": "#ef4444"},
        ],
        "confidence_levels": {
            "High": ">= 70",
            "Moderate": "45 – 69",
            "Low": "< 45",
        },
        "federated_learning": {
            "title": "Privacy-Preserving Federated Learning",
            "points": [
                "Local hospital models train independently on node-specific data.",
                "Only model parameters and aggregated predictions are shared — never raw patient records.",
                "The global model combines hospital contributions using sample-size-weighted ensembling.",
                "Raw patient data never leaves the originating hospital node.",
            ],
        },
        "features_used": FEATURE_COLUMNS,
        "excluded_leakage_features": ["Estimated_Avg_Glucose"],
        "combined_global_metrics": global_combined,
    }
    with open(ML_DIR / "clinical_summary.json", "w") as f:
        json.dump(clinical_summary, f, indent=2)

    sample_predictions = []
    for bundle in hospital_bundles:
        hid = bundle["hospital_id"]
        preds = global_ensemble.predict(bundle["X_test"])
        for i in range(min(5, len(bundle["y_test"]))):
            sample_predictions.append(
                {
                    "hospital": hid,
                    "actual_hba1c": round(float(bundle["y_test"][i]), 2),
                    "predicted_hba1c": round(float(preds[i]), 2),
                }
            )
    prediction_results = {
        "description": "Sample federated predictions from held-out hospital test splits",
        "metrics": global_combined,
        "samples": sample_predictions,
    }
    with open(ML_DIR / "prediction_results.json", "w") as f:
        json.dump(prediction_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"Trained {len(hospital_bundles)} hospital models")
    print(f"Global federated R²:  {global_combined['R2']:.4f}")
    print(f"Global federated RMSE: {global_combined['RMSE']:.4f}")
    print(f"Global federated MAPE: {global_combined['MAPE']:.2f}%")
    target_met = global_combined["R2"] > 0.5 and global_combined["MAPE"] < 20.0
    print(
        f"Target (R² > 0.5, MAPE < 20%): "
        f"{'MET' if target_met else 'NOT MET (dataset may need more quality)'}",
    )
    print(
        "Artifacts: Global_Model.pkl, scaler.pkl, feature_columns.json, results.json, "
        "federated_metrics.json, clinical_summary.json, prediction_results.json"
    )
    print(f"Plots: feature_distributions.png, feature_importance.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
