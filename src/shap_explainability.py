"""
Retention Risk - Phase 5: SHAP Explainability
Reads models/model.pkl and models/test_set.parquet (outputs of Phase 4).

Produces:
1. Global summary plot -- which features drive risk overall, and in
   which direction (high/low value pushing risk up or down).
2. Global bar plot -- mean absolute SHAP value per feature, a simpler
   "importance ranking" view.
3. Per-customer explanations for a few illustrative cases (a confident
   true positive, a false positive, a false negative) -- these are the
   sentences a retention team would actually want to read.
4. A saved CSV of SHAP values per customer
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

MODEL_PATH = "models/model.pkl"
TEST_SET_PATH = "models/test_set.parquet"
OUTPUT_DIR = "models"


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    test_df = pd.read_parquet(TEST_SET_PATH)

    feature_cols = [c for c in test_df.columns if c not in ("CustomerID", "AtRisk")]
    X_test = test_df[feature_cols]
    y_test = test_df["AtRisk"]
    customer_ids = test_df["CustomerID"]

    return model, X_test, y_test, customer_ids, feature_cols


def explain_customer(shap_row: pd.Series, feature_values: pd.Series,
                      base_value: float, top_n: int = 3) -> str:
    """
    Turn one customer's SHAP values into a plain-English summary:
    the top N features pushing risk up or down, with their actual values.
    """
    sorted_features = shap_row.abs().sort_values(ascending=False).index[:top_n]

    lines = []
    for feat in sorted_features:
        contribution = shap_row[feat]
        direction = "increases" if contribution > 0 else "decreases"
        lines.append(
            f"  - {feat} = {feature_values[feat]:.1f} "
            f"({direction} risk, SHAP {contribution:+.3f})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    model, X_test, y_test, customer_ids, feature_cols = load_model_and_data()

    print(f"Test set: {len(X_test)} customers, {len(feature_cols)} features")

    # --- Compute SHAP values ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    base_value = explainer.expected_value

    shap_df = pd.DataFrame(shap_values, columns=feature_cols, index=X_test.index)

    # --- Global plots ---
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/shap_summary.png")

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_importance_bar.png", dpi=150)
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/shap_importance_bar.png")

    # --- Predictions, for picking illustrative examples ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "CustomerID": customer_ids.values,
        "Actual": y_test.values,
        "Predicted": y_pred,
        "RiskScore": y_proba,
    }, index=X_test.index)

    # --- Select illustrative customers ---
    true_positive = results[(results.Actual == 1) & (results.Predicted == 1)] \
        .sort_values("RiskScore", ascending=False).head(1)
    false_positive = results[(results.Actual == 0) & (results.Predicted == 1)] \
        .sort_values("RiskScore", ascending=False).head(1)
    false_negative = results[(results.Actual == 1) & (results.Predicted == 0)] \
        .sort_values("RiskScore").head(1)

    print("\n" + "=" * 60)
    for label, subset in [
        ("Confident TRUE POSITIVE (correctly flagged high-risk)", true_positive),
        ("FALSE POSITIVE (flagged, but customer stayed active)", false_positive),
        ("FALSE NEGATIVE (missed, customer actually went quiet)", false_negative),
    ]:
        if subset.empty:
            continue
        idx = subset.index[0]
        cust_id = subset.loc[idx, "CustomerID"]
        score = subset.loc[idx, "RiskScore"]

        print(f"\n{label}")
        print(f"CustomerID: {cust_id}, Risk score: {score:.2f}")
        print("Top contributing factors:")
        print(explain_customer(shap_df.loc[idx], X_test.loc[idx], base_value))

        # individual waterfall plot for this customer
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            base_value, shap_values[X_test.index.get_loc(idx)],
            X_test.loc[idx], show=False
        )
        plt.tight_layout()
        fname = f"{OUTPUT_DIR}/shap_waterfall_{cust_id}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved: {fname}")

    print("\n" + "=" * 60)

    # --- Export SHAP values for the app ---
    export = shap_df.copy()
    export.insert(0, "CustomerID", customer_ids.values)
    export.insert(1, "RiskScore", y_proba)
    export.insert(2, "Predicted", y_pred)
    export.to_csv(f"{OUTPUT_DIR}/shap_values_per_customer.csv", index=False)
    print(f"Saved: {OUTPUT_DIR}/shap_values_per_customer.csv (feeds Phase 6 app)")