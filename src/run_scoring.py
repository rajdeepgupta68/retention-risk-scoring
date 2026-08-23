"""
Retention Risk - Scoring Pipeline (run on a schedule)
Regenerates features for every customer and scores them with the trained
model, producing a single output file the Streamlit app reads from.
"""

import pandas as pd
import numpy as np
import joblib
import shap

CLEANED_PATH = "data/cleaned_transactions.parquet"
MODEL_PATH = "models/model.pkl"
OUTPUT_PATH = "data/latest_scores.parquet"

import importlib.util
spec = importlib.util.spec_from_file_location("feat_eng", "src/feature_engineering.py")
feat_eng = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat_eng)


def score_all_customers():
    df = pd.read_parquet(CLEANED_PATH)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    snapshot_date = df["InvoiceDate"].max()

    rfm = feat_eng.build_rfm(df, snapshot_date)
    diversity = feat_eng.build_diversity_features(df)
    trend = feat_eng.build_trend_feature(df, snapshot_date)

    features = rfm.merge(diversity, on="CustomerID", how="left")
    features = features.merge(trend, on="CustomerID", how="left")
    features = features.drop(columns=["LastPurchaseDate", "FirstPurchaseDate"])

    model = joblib.load(MODEL_PATH)
    feature_cols = [c for c in features.columns
                    if c not in ("CustomerID", "Recency")]  # Recency excluded, same as training
    X = features[feature_cols]

    risk_scores = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    # For each customer, find the top 3 features contributing to their risk score
    reasons = []
    for i in range(len(X)):
        row = shap_df.iloc[i]
        top_feats = row.abs().sort_values(ascending=False).index[:3]
        parts = []
        for feat in top_feats:
            val = X.iloc[i][feat]
            direction = "↑ risk" if row[feat] > 0 else "↓ risk"
            parts.append(f"{feat}={val:.1f} ({direction})")
        reasons.append("; ".join(parts))

    output = pd.DataFrame({
        "CustomerID": features["CustomerID"].values,
        "RiskScore": risk_scores,
        "Prediction": predictions,
        "TopReasons": reasons,
        "Recency": rfm["Recency"].values,  # kept for display only, not used in scoring
        "Frequency": features["Frequency"].values,
        "Monetary": features["Monetary"].values,
        "Tenure": features["Tenure"].values,
    }).sort_values("RiskScore", ascending=False)

    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Scored {len(output)} customers.")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"\nTop 5 highest risk:\n{output.head()}")

    return output


if __name__ == "__main__":
    score_all_customers()