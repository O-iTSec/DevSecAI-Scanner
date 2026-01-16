import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "randomforest_risk_model.pkl")  

FEATURE_COLS = [
    "total_vulns", "critical", "high", "medium", "low",
    "fixable_vulns", "fixable_ratio", "high_risk_ratio", "severity_index"
]


def extract_features_from_trivy(json_path):
    """Parse a Trivy JSON file and extract the same features used for ML training."""
    with open(json_path, "r") as f:
        scan = json.load(f)

    results = scan.get("Results", [])
    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    total_vulns, fixable_vulns = 0, 0

    for res in results:
        vulns = res.get("Vulnerabilities") or []
        total_vulns += len(vulns)
        for v in vulns:
            sev = v.get("Severity", "UNKNOWN").upper()
            if sev not in sev_counts:
                sev = "UNKNOWN"
            sev_counts[sev] += 1
            if v.get("FixedVersion"):
                fixable_vulns += 1

    # Derived metrics
    fixable_ratio = round(fixable_vulns / total_vulns, 4) if total_vulns > 0 else 0
    high_risk_ratio = (
        (sev_counts["CRITICAL"] + sev_counts["HIGH"]) / total_vulns 
        if total_vulns > 0 else 0
    )
    severity_index = (
        4 * sev_counts["CRITICAL"] +
        3 * sev_counts["HIGH"] +
        2 * sev_counts["MEDIUM"] +
        1 * sev_counts["LOW"]
    )

    row = {
        "total_vulns": total_vulns,
        "critical": sev_counts["CRITICAL"],
        "high": sev_counts["HIGH"],
        "medium": sev_counts["MEDIUM"],
        "low": sev_counts["LOW"],
        "fixable_vulns": fixable_vulns,
        "fixable_ratio": fixable_ratio,
        "high_risk_ratio": round(high_risk_ratio, 4),
        "severity_index": severity_index,
    }

    return row


def predict_risk(json_path):
    print(f"[*] Loading Trivy scan: {json_path}")

    # Extract features from the Trivy file
    feats = extract_features_from_trivy(json_path)
    X = pd.DataFrame([feats])[FEATURE_COLS]

    print("\n[+] Extracted Features:")
    print(X)

    # Load artifacts
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    # Detect model type (XGBoost uses encoded labels, others use string labels)
    model_type = type(model).__name__
    is_xgboost = "XGB" in model_type.upper() or "XGBoost" in model_type

    # Scale input
    X_scaled = scaler.transform(X)

    # Predict
    pred_raw = model.predict(X_scaled)
    
    # Decode predictions if XGBoost (it uses encoded labels)
    if is_xgboost:
        # XGBoost returns numeric predictions, need to decode
        pred_label = label_encoder.inverse_transform(pred_raw)
    else:
        # RandomForest/LogisticRegression return string labels directly
        pred_label = pred_raw

    # Probabilities (confidence scores)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        confidence = np.max(probs)
        
        # Map probabilities to class labels
        if is_xgboost:
            # For XGBoost, probabilities are in encoded order, map to labels
            class_labels = label_encoder.classes_
        else:
            # For other models, get class labels from the model
            class_labels = model.classes_
    else:
        probs = None
        confidence = None
        class_labels = None

    # Print results
    print("\n=== RISK PREDICTION ===")
    print(f"Image: {os.path.basename(json_path)}")
    print(f"Predicted Risk Level: {pred_label[0]}")
    if confidence:
        print(f"Model Confidence: {confidence:.4f}")

    if probs is not None and class_labels is not None:
        print("\nClass Probabilities:")
        for label, p in zip(class_labels, probs):
            print(f"  {label}: {p:.4f}")

    return pred_label[0]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict risk level from Trivy JSON scan.")
    parser.add_argument("json_path", type=str, help="Path to Trivy JSON scan file")
    args = parser.parse_args()

    predict_risk(args.json_path)
