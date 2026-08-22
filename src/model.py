"""
Retention Risk - Phase 4: Modeling
Reads data/customer_features.parquet

Trains two models:
1. Logistic Regression (baseline)
2. XGBoost (tree-based, non-linear)
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from xgboost import XGBClassifier

FEATURES_PATH = "data/customer_features.parquet"
MODEL_DIR = "models"


def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    X = df.drop(columns=["CustomerID", "AtRisk", "Recency"])
    y = df["AtRisk"]
    return X, y, df["CustomerID"]


def evaluate(name: str, y_test, y_pred, y_proba) -> None:
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["NotAtRisk", "AtRisk"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Average Precision (PR-AUC): {average_precision_score(y_test, y_proba):.3f}")
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    X, y, customer_ids = load_features()

    print(f"Feature columns used: {list(X.columns)}")
    print(f"Class balance: {y.value_counts(normalize=True).to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # --- Baseline: Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)

    logreg_pred = logreg.predict(X_test_scaled)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    evaluate("Logistic Regression (baseline)", y_test, logreg_pred, logreg_proba)

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    evaluate("XGBoost", y_test, xgb_pred, xgb_proba)

    # --- Feature importance (quick look, SHAP comes in Phase 5) ---
    importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n=== XGBoost feature importances ===")
    print(importances)

    # --- Save the model that performs better (compare ROC-AUC) ---
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)

    logreg_auc = roc_auc_score(y_test, logreg_proba)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    if xgb_auc >= logreg_auc:
        joblib.dump(xgb, f"{MODEL_DIR}/model.pkl")
        print(f"\nSaved XGBoost as the chosen model (ROC-AUC {xgb_auc:.3f} vs "
              f"LogReg {logreg_auc:.3f})")
    else:
        joblib.dump({"model": logreg, "scaler": scaler}, f"{MODEL_DIR}/model.pkl")
        print(f"\nSaved Logistic Regression as the chosen model (ROC-AUC "
              f"{logreg_auc:.3f} vs XGBoost {xgb_auc:.3f})")

    X_test.assign(CustomerID=customer_ids.loc[X_test.index].values,
                  AtRisk=y_test.values).to_parquet(f"{MODEL_DIR}/test_set.parquet", index=False)
    print(f"Saved test set for SHAP analysis: {MODEL_DIR}/test_set.parquet")