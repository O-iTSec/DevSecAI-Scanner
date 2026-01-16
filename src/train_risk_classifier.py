import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import random

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# -----------------------------------
# CONFIG
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------------------
# HELPER FUNCTIONS
# -----------------------------------
def label_risk(score):
    """Convert risk score to categorical label."""
    if score < 0.08:
        return "Low"
    elif score < 0.18:
        return "Medium"
    else:
        return "High"

# -----------------------------------
# MAIN FUNCTION
# -----------------------------------
def main():
    """Main training pipeline."""
    # -----------------------------------
    # STEP 1: LOAD DATA
    # -----------------------------------
    print(f"[*] Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)

    print(f"[i] Dataset loaded: {len(df)} records, {len(df.columns)} features")
    print(df.head(3))

    # -----------------------------------
    # STEP 2: LABEL RISK LEVELS
    # -----------------------------------
    df["risk_label"] = df["risk_score"].apply(label_risk)
    print("\n[+] Risk labels assigned:")
    print(df["risk_label"].value_counts())

    # -----------------------------------
    # OPTIONAL: INJECT LABEL NOISE (simulate human judgement errors)
    # -----------------------------------
    NOISE_PROBABILITY = 0.05   # 5% label noise

    def randomize_label(label):
        labels = ["Low", "Medium", "High"]
        labels.remove(label)  # can't pick the same label
        return random.choice(labels)

    df["risk_label_noisy"] = df["risk_label"].apply(
        lambda lbl: randomize_label(lbl) if random.random() < NOISE_PROBABILITY else lbl
    )

    print("\n[+] Noise injection applied:")
    print(df["risk_label_noisy"].value_counts())

    # -----------------------------------
    # STEP 3: SELECT FEATURES
    # -----------------------------------
    feature_cols = [
        "total_vulns", "critical", "high", "medium", "low",
        "fixable_vulns", "fixable_ratio", "high_risk_ratio", "severity_index"
    ]
    X = df[feature_cols].fillna(0)
    # Use noisy labels for training
    y = df["risk_label_noisy"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Encode labels for XGBoost (if needed)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # -----------------------------------
    # STEP 4: TRAIN MODELS
    # -----------------------------------
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=200, solver='lbfgs')
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42, eval_metric='mlogloss')

    results = {}

    for name, model in models.items():
        print(f"\n[*] Training {name}...")
        # Use encoded labels for XGBoost, original labels for others
        if name == "XGBoost":
            model.fit(X_train_scaled, y_train_encoded)
            preds_encoded = model.predict(X_test_scaled)
            preds = label_encoder.inverse_transform(preds_encoded)
        else:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        
        report = classification_report(y_test, preds, output_dict=True)
        results[name] = report

        print(f"\n=== {name} CLASSIFICATION REPORT ===")
        print(classification_report(y_test, preds))
        print(confusion_matrix(y_test, preds))

        # Save model
        joblib.dump(model, os.path.join(MODEL_PATH, f"{name.lower()}_risk_model.pkl"))

    # Save scaler and label encoder for later use
    joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(MODEL_PATH, "label_encoder.pkl"))

    print(f"\n[✓] Models saved to: {MODEL_PATH}")

    # -----------------------------------
    # STEP 5: EVALUATION & VISUALIZATION
    # -----------------------------------
    # Plot confusion matrix for best model
    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n[🏆] Best model: {best_model_name}")

    best_model = models[best_model_name]
    preds_best = best_model.predict(X_test_scaled)
    # Decode predictions if XGBoost (it uses encoded labels)
    if best_model_name == "XGBoost":
        preds_best = label_encoder.inverse_transform(preds_best)
    cm = confusion_matrix(y_test, preds_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PATH, f"confusion_matrix_{best_model_name.lower()}.png"))
    print(f"[✓] Confusion matrix saved to: {os.path.join(MODEL_PATH, f'confusion_matrix_{best_model_name.lower()}.png')}")
    plt.close()

    # Feature importance (if RandomForest)
    if best_model_name == "RandomForest":
        importances = best_model.feature_importances_
        feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
        feat_df.sort_values(by="importance", ascending=False, inplace=True)
        print("\n=== Feature Importance (RandomForest) ===")
        print(feat_df)
        plt.figure(figsize=(8, 4))
        sns.barplot(data=feat_df, x="importance", y="feature", hue="feature", palette="magma", legend=False)
        plt.title("Feature Importance - RandomForest")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_PATH, "feature_importance_randomforest.png"))
        print(f"[✓] Feature importance plot saved to: {os.path.join(MODEL_PATH, 'feature_importance_randomforest.png')}")
        plt.close()

    print("\n[✓] Training complete!")


if __name__ == "__main__":
    main()
