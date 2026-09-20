import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="HAR: RNN vs LSTM vs GRU", layout="wide")
st.title("Human Activity Recognition — RNN vs LSTM vs GRU")
st.caption("UCI HAR dataset · smartphone sensor data · 6 activities")


@st.cache_data(ttl=30)
def get_results():
    r = requests.get(f"{API_BASE_URL}/results", timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30)
def get_samples(limit=200):
    r = requests.get(f"{API_BASE_URL}/samples", params={"limit": limit}, timeout=30)
    r.raise_for_status()
    return r.json()


def predict(index, model="all"):
    r = requests.post(f"{API_BASE_URL}/predict", json={"index": index, "model": model}, timeout=30)
    r.raise_for_status()
    return r.json()


try:
    health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
except Exception:
    st.error(
        f"Cannot reach the API at {API_BASE_URL}. "
        "Start it with `./.venv/bin/python -m backend.main` (after running training)."
    )
    st.stop()

tab_predict, tab_compare, tab_confusion = st.tabs(
    ["🔍 Try a Prediction", "📊 Model Comparison", "🧩 Confusion Matrix"]
)

# ---------------------------------------------------------------------------
# Tab 1: Try a Prediction (the user's query)
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("Pick a test sample and see what each model predicts")

    samples = get_samples(limit=health["num_test_samples"])
    sample_df = pd.DataFrame(samples["items"])

    col_a, col_b = st.columns([1, 2])
    with col_a:
        activity_filter = st.selectbox(
            "Filter by actual activity", ["All"] + sorted(sample_df["actual_activity"].unique())
        )
        filtered = (
            sample_df
            if activity_filter == "All"
            else sample_df[sample_df["actual_activity"] == activity_filter]
        )
        index = st.selectbox(
            "Test sample index",
            filtered["index"].tolist(),
            format_func=lambda i: f"#{i}  (subject {sample_df.loc[sample_df['index'] == i, 'subject'].values[0]})",
        )
        run = st.button("Predict", type="primary")

    if run or index is not None:
        result = predict(index, model="all")
        actual = result["actual_activity"]

        st.markdown(f"### Actual activity: **{actual}**")
        cols = st.columns(3)
        for col, (model_name, pred) in zip(cols, result["predictions"].items()):
            with col:
                icon = "✅" if pred["correct"] else "❌"
                st.metric(
                    label=f"{icon} {model_name}",
                    value=pred["predicted_activity"],
                    delta=f"{pred['confidence'] * 100:.1f}% confidence",
                )
                probs = pred["probabilities"]
                fig = go.Figure(
                    go.Bar(
                        x=list(probs.values()),
                        y=list(probs.keys()),
                        orientation="h",
                        marker_color=["#2e7d32" if k == actual else "#90a4ae" for k in probs],
                    )
                )
                fig.update_layout(
                    height=220,
                    margin=dict(l=0, r=0, t=10, b=10),
                    xaxis=dict(range=[0, 1], title="probability"),
                )
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Model Comparison
# ---------------------------------------------------------------------------
with tab_compare:
    st.subheader("Test accuracy comparison")
    results = get_results()
    res = results["results"]

    acc_df = pd.DataFrame(
        {
            "Model": list(res.keys()),
            "Test Accuracy (%)": [res[m]["test_accuracy"] * 100 for m in res],
            "Train time (s)": [res[m]["train_seconds"] for m in res],
        }
    ).round(2)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(acc_df, hide_index=True, use_container_width=True)
        st.info(f"Best model: **{results['best_model']}**")
    with col2:
        fig = go.Figure(go.Bar(x=acc_df["Model"], y=acc_df["Test Accuracy (%)"]))
        fig.update_layout(yaxis=dict(range=[0, 100]), title="Test Accuracy by Model")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Training curves")
    curve_col1, curve_col2 = st.columns(2)
    with curve_col1:
        fig = go.Figure()
        for name, r in res.items():
            fig.add_trace(go.Scatter(y=r["val_accuracy"], name=name, mode="lines"))
        fig.update_layout(title="Validation Accuracy", xaxis_title="Epoch")
        st.plotly_chart(fig, use_container_width=True)
    with curve_col2:
        fig = go.Figure()
        for name, r in res.items():
            fig.add_trace(go.Scatter(y=r["val_loss"], name=name, mode="lines"))
        fig.update_layout(title="Validation Loss", xaxis_title="Epoch")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Confusion Matrix + classification report (best model)
# ---------------------------------------------------------------------------
with tab_confusion:
    results = get_results()
    class_names = results["class_names"]
    best_name = results["best_model"]
    st.subheader(f"Confusion Matrix — {best_name} (best model)")

    cm = results["confusion_matrix"]
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification report")
    report = results["classification_report"]
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)
