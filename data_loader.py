"""
Robust Healthcare Dataset Preprocessing and Partitioning Module.
Ensures patient-level split (preventing data leakage) and consistent feature/target scaling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "Fasting_Glucose",
    "Postprandial_Glucose",
    "Triglycerides",
    "HDL",
    "LDL",
]
DEFAULT_TARGET = "HbA1c_%"
HOSPITALS = ["A", "B", "C", "D", "E"]
INVALID_TOKENS = ["?", "NA", "N/A", "", "nan", "NaN", "None", "null"]


def clean_dataframe(df: pd.DataFrame, features: List[str], target_col: str) -> pd.DataFrame:
    """
    Cleans raw hospital dataframe:
    - Replaces invalid tokens with NaN
    - Converts feature and target columns to numeric
    - Drops rows where target is missing or non-positive
    - Imputes missing features with column medians
    """
    df = df.copy()
    df = df.replace(INVALID_TOKENS, np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clean target
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col])
    df = df[df[target_col] > 0]

    # Clean features
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = 0.0

    return df


def split_by_patient(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Splits records by MedID (patient ID) so that no patient records
    appear simultaneously in train and test sets.
    """
    if "MedID" in df.columns:
        patients = df["MedID"].dropna().unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(patients)

        split_idx = int(len(patients) * (1 - test_size))
        train_patients = set(patients[:split_idx])
        test_patients = set(patients[split_idx:])

        train_mask = df["MedID"].isin(train_patients)
        test_mask = df["MedID"].isin(test_patients)

        train_df = df[train_mask]
        test_df = df[test_mask]
        n_train_pts = len(train_patients)
        n_test_pts = len(test_patients)
    else:
        # Fallback to random partition if MedID not present
        rng = np.random.RandomState(seed)
        indices = np.arange(len(df))
        rng.shuffle(indices)
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[indices[:split_idx]]
        test_df = df.iloc[indices[split_idx:]]
        n_train_pts = len(train_df)
        n_test_pts = len(test_df)

    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    return X_train, y_train, X_test, y_test, n_train_pts, n_test_pts


def load_all_hospitals_data(
    dataset_dir: Path,
    features: List[str] = DEFAULT_FEATURES,
    target_col: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, Any]], StandardScaler, StandardScaler]:
    """
    Loads all 5 hospital datasets, partitions them into patient-isolated splits,
    and fits global reference StandardScaler for features and target.
    """
    raw_splits = {}
    all_X_train = []
    all_y_train = []

    for hid in HOSPITALS:
        fpath = dataset_dir / f"Hospital_{hid}_Master_Records.csv"
        if not fpath.exists():
            raise FileNotFoundError(f"Hospital data file not found: {fpath}")

        df = pd.read_csv(fpath)
        cleaned = clean_dataframe(df, features, target_col)
        X_tr, y_tr, X_te, y_te, n_tr_pts, n_te_pts = split_by_patient(
            cleaned, features, target_col, test_size=test_size, seed=seed
        )

        raw_splits[hid] = {
            "X_train_raw": X_tr,
            "y_train_raw": y_tr,
            "X_test_raw": X_te,
            "y_test_raw": y_te,
            "n_train_pts": n_tr_pts,
            "n_test_pts": n_te_pts,
            "total_records": len(cleaned),
        }
        all_X_train.append(X_tr)
        all_y_train.append(y_tr)

    # Fit global reference scalers
    stacked_X = np.vstack(all_X_train)
    stacked_y = np.concatenate(all_y_train).reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_X.fit(stacked_X)

    scaler_y = StandardScaler()
    scaler_y.fit(stacked_y)

    # Scale the splits
    hospital_data = {}
    for hid, split in raw_splits.items():
        hospital_data[hid] = {
            "X_train": scaler_X.transform(split["X_train_raw"]).astype(np.float32),
            "y_train": scaler_y.transform(split["y_train_raw"].reshape(-1, 1)).flatten().astype(np.float32),
            "X_test": scaler_X.transform(split["X_test_raw"]).astype(np.float32),
            "y_test": split["y_test_raw"].astype(np.float32),  # unscaled ground truth for final metric reporting
            "y_test_scaled": scaler_y.transform(split["y_test_raw"].reshape(-1, 1)).flatten().astype(np.float32),
            "n_train_samples": len(split["X_train_raw"]),
            "n_test_samples": len(split["X_test_raw"]),
            "n_train_patients": split["n_train_pts"],
            "n_test_patients": split["n_test_pts"],
            "features": features,
            "target": target_col,
        }

    return hospital_data, scaler_X, scaler_y
