"""Rule-based clinical interpretation for federated HbA1c predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RISK_BANDS = [
    (0.0, 5.7, "Normal", "#22c55e"),
    (5.7, 6.5, "Prediabetes", "#eab308"),
    (6.5, 8.5, "Diabetes", "#f97316"),
    (8.5, 20.0, "High Risk Diabetes", "#ef4444"),
]

FEATURE_KEYS = [
    "Fasting_Glucose",
    "Postprandial_Glucose",
    "Triglycerides",
    "HDL",
    "LDL",
]


def categorize_hba1c(hba1c: float) -> dict[str, str]:
    value = float(hba1c)
    if value < 5.7:
        category, color = "Normal", "#22c55e"
    elif value < 6.5:
        category, color = "Prediabetes", "#eab308"
    elif value < 8.5:
        category, color = "Diabetes", "#f97316"
    else:
        category, color = "High Risk Diabetes", "#ef4444"
    return {
        "risk_category": category,
        "risk_color": color,
        "triage_level": category,
    }


def _feature_values(data: dict) -> dict[str, float]:
    aliases = {
        "Fasting_Glucose": ("fastingGlucose", "fasting_glucose", "Fasting_Glucose", "glucose"),
        "Postprandial_Glucose": (
            "postprandialGlucose",
            "postprandial_glucose",
            "Postprandial_Glucose",
        ),
        "Triglycerides": ("triglycerides", "Triglycerides"),
        "HDL": ("hdl", "HDL"),
        "LDL": ("ldl", "LDL"),
    }
    out = {}
    for feat, keys in aliases.items():
        val = None
        for key in keys:
            if key in data and data[key] not in (None, ""):
                try:
                    val = float(data[key])
                    break
                except (TypeError, ValueError):
                    continue
        out[feat] = val
    return out


def feature_completeness(features: dict[str, float | None]) -> float:
    present = sum(
        1
        for v in features.values()
        if v is not None and np.isfinite(v) and v > 0
    )
    return present / max(len(FEATURE_KEYS), 1)


def load_training_reference(ml_dir: Path | None = None) -> dict:
    base = ml_dir or Path(__file__).resolve().parent
    path = base / "training_reference.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "hba1c_mean": 6.95,
        "hba1c_std": 1.55,
        "feature_means": {},
    }


def compute_confidence(
    predicted_hba1c: float,
    features: dict[str, float | None],
    ml_pred: float | None = None,
    clinical_pred: float | None = None,
    training_ref: dict | None = None,
) -> dict[str, Any]:
    ref = training_ref or load_training_reference()
    mean = float(ref.get("hba1c_mean", 6.95))
    std = max(float(ref.get("hba1c_std", 1.55)), 0.5)

    z_distance = abs(predicted_hba1c - mean) / std
    distance_score = max(0.0, 1.0 - min(z_distance / 3.0, 1.0))

    completeness = feature_completeness(features)

    if ml_pred is not None and clinical_pred is not None:
        spread = abs(ml_pred - clinical_pred)
        stability_score = max(0.0, 1.0 - min(spread / 2.5, 1.0))
    else:
        stability_score = 0.65

    score_100 = round(
        100 * (0.4 * distance_score + 0.35 * completeness + 0.25 * stability_score),
        1,
    )
    if score_100 >= 70:
        label = "High"
    elif score_100 >= 45:
        label = "Moderate"
    else:
        label = "Low"

    return {
        "confidence_score": score_100,
        "confidence_level": label,
        "confidence_factors": {
            "distance_from_training_mean": round(distance_score, 3),
            "feature_completeness": round(completeness, 3),
            "prediction_stability": round(stability_score, 3),
        },
    }


def generate_medical_explanation(features: dict[str, float | None], hba1c: float) -> str:
    parts = []
    fasting = features.get("Fasting_Glucose")
    post = features.get("Postprandial_Glucose")
    tg = features.get("Triglycerides")
    hdl = features.get("HDL")
    ldl = features.get("LDL")

    if fasting is not None and fasting >= 126:
        parts.append("elevated fasting glucose")
    elif fasting is not None and fasting >= 100:
        parts.append("borderline fasting glucose")

    if post is not None and post >= 200:
        parts.append("elevated postprandial glucose")
    elif post is not None and post >= 140:
        parts.append("postprandial glucose above optimal range")

    if tg is not None and tg >= 200:
        parts.append("high triglycerides")
    elif tg is not None and tg >= 150:
        parts.append("elevated triglycerides")

    if hdl is not None and hdl < 40:
        parts.append("low HDL cholesterol")
    elif hdl is not None and hdl < 50:
        parts.append("suboptimal HDL levels")

    if ldl is not None and ldl >= 160:
        parts.append("high LDL cholesterol")
    elif ldl is not None and ldl >= 130:
        parts.append("borderline LDL cholesterol")

    risk = categorize_hba1c(hba1c)["risk_category"]
    if not parts:
        if risk == "Normal":
            return (
                "Current glucose and lipid markers are within commonly monitored ranges. "
                "Continue routine preventive care and periodic HbA1c screening."
            )
        return (
            f"Predicted HbA1c suggests {risk.lower()} glycemic control. "
            "Metabolic markers should be monitored with follow-up laboratory testing."
        )

    joined = ", ".join(parts[:3])
    if len(parts) > 3:
        joined += ", and additional lipid abnormalities"
    return (
        f"{joined.capitalize()} indicate increased metabolic risk. "
        f"The federated model estimates HbA1c consistent with {risk.lower()} status."
    )


def generate_recommendations(hba1c: float, features: dict[str, float | None]) -> list[str]:
    recs = []
    risk = categorize_hba1c(hba1c)["risk_category"]
    fasting = features.get("Fasting_Glucose")
    post = features.get("Postprandial_Glucose")
    tg = features.get("Triglycerides")

    if risk in ("Prediabetes", "Diabetes", "High Risk Diabetes"):
        recs.append("Reduce refined sugar and ultra-processed carbohydrate intake")
        recs.append("Increase physical activity (≥150 minutes/week moderate exercise)")

    if risk in ("Diabetes", "High Risk Diabetes"):
        recs.append("Consult an endocrinologist for individualized glycemic management")
        recs.append("Repeat HbA1c test after 3 months")

    if risk == "High Risk Diabetes":
        recs.append("Consider comprehensive cardiovascular risk assessment")

    if fasting is not None and fasting >= 100:
        recs.append("Monitor fasting glucose and reinforce dietary counseling")

    if post is not None and post >= 140:
        recs.append("Review post-meal glucose patterns with the care team")

    if tg is not None and tg >= 150:
        recs.append("Address hypertriglyceridemia through diet and lifestyle modification")

    if risk == "Normal" and len(recs) == 0:
        recs.append("Maintain healthy lifestyle habits")
        recs.append("Repeat HbA1c screening per preventive care guidelines (typically annually)")

    seen = set()
    unique = []
    for item in recs:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def build_clinical_interpretation(
    data: dict,
    predicted_hba1c: float,
    ml_pred: float | None = None,
    clinical_pred: float | None = None,
    training_ref: dict | None = None,
) -> dict[str, Any]:
    features = _feature_values(data)
    risk = categorize_hba1c(predicted_hba1c)
    confidence = compute_confidence(
        predicted_hba1c,
        features,
        ml_pred=ml_pred,
        clinical_pred=clinical_pred,
        training_ref=training_ref,
    )
    explanation = generate_medical_explanation(features, predicted_hba1c)
    recommendations = generate_recommendations(predicted_hba1c, features)

    return {
        "predicted_hba1c": round(float(predicted_hba1c), 2),
        **risk,
        **confidence,
        "medical_explanation": explanation,
        "recommendations": recommendations,
        "feature_snapshot": {
            k: (None if v is None else round(v, 2)) for k, v in features.items()
        },
    }
