"""
Retention Risk - Scoring App (Streamlit)
Reads data/latest_scores.parquet (output of run_scoring.py).

"""

import streamlit as st
import pandas as pd

SCORES_PATH = "data/latest_scores.parquet"

st.set_page_config(page_title="Retention Risk Scoring", layout="wide")
st.title("Customer Retention Risk Scoring")
st.caption(
    "Predicts which customers are likely to go quiet, based on their "
    "purchase history. Scores refresh whenever the scoring pipeline "
    "re-runs - this app just reads the latest output."
)


@st.cache_data
def load_scores():
    return pd.read_parquet(SCORES_PATH)


scores = load_scores()

tab1, tab2 = st.tabs(["Look up a customer", "Highest-risk customers"])

with tab1:
    customer_id = st.selectbox(
        "Select a Customer ID", scores["CustomerID"].sort_values().unique()
    )
    row = scores[scores["CustomerID"] == customer_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{row['RiskScore']:.0%}")
    col2.metric("Prediction", "At Risk" if row["Prediction"] == 1 else "Not At Risk")
    col3.metric("Days Since Last Purchase", f"{row['Recency']:.0f}")

    st.subheader("Why this score?")
    reasons = row["TopReasons"].split("; ")
    for r in reasons:
        st.write(f"- {r}")

    st.subheader("Purchase history summary")
    st.write(
        f"**Frequency:** {row['Frequency']:.0f} orders  \n"
        f"**Total spend:** £{row['Monetary']:.2f}  \n"
        f"**Customer for:** {row['Tenure']:.0f} days"
    )

with tab2:
    st.subheader("Top 20 highest-risk customers")
    top20 = scores.head(20)[
        ["CustomerID", "RiskScore", "Frequency", "Monetary", "Recency", "TopReasons"]
    ]
    st.dataframe(
        top20.style.format({"RiskScore": "{:.0%}", "Monetary": "£{:.2f}"}),
        use_container_width=True,
    )

    st.subheader("Risk score distribution")

    bin_edges = [i / 10 for i in range(11)]
    bin_labels = [f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%" for i in range(10)]
    binned = pd.cut(scores["RiskScore"], bins=bin_edges, labels=bin_labels, include_lowest=True)
    dist = binned.value_counts().reindex(bin_labels)
    st.bar_chart(dist)

st.caption(
    f"Scoring this run: {len(scores)} customers. "
)