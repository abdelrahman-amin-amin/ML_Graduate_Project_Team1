import json
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, CatBoostRegressor

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Solar Farm Fault Detection", page_icon="☀️", layout="wide")
st.title("☀️ Solar Farm Fault Detection & Power Prediction")
st.caption(
    "Type in a single sensor reading and get predictions from all 6 models: "
    "expected AC power, whether the device is faulted, and (if faulted) which fault it is."
)

ARTIFACT_DIR = "deployment_artifacts"


# ------------------------------------------------------------
# Load models + metadata (cached so it only happens once)
# ------------------------------------------------------------
@st.cache_resource
def load_everything():
    ac_model = CatBoostRegressor()
    ac_model.load_model(f"{ARTIFACT_DIR}/ac_power_model.cbm")

    is_faulted_model = CatBoostClassifier()
    is_faulted_model.load_model(f"{ARTIFACT_DIR}/is_faulted_model.cbm")

    fault_targets = [
        "fault_soiling",
        "fault_inverter_overheat",
        "fault_tracker_stuck",
        "fault_dc_string_outage",
    ]
    fault_models = {}
    for target in fault_targets:
        m = CatBoostClassifier()
        m.load_model(f"{ARTIFACT_DIR}/{target}_model.cbm")
        fault_models[target] = m

    with open(f"{ARTIFACT_DIR}/feature_lists.json") as f:
        feature_lists = json.load(f)

    with open(f"{ARTIFACT_DIR}/feature_stats.json") as f:
        feature_stats = json.load(f)

    return ac_model, is_faulted_model, fault_models, feature_lists, feature_stats


try:
    ac_model, is_faulted_model, fault_models, feature_lists, feature_stats = load_everything()
except Exception as e:
    st.error(
        "Could not load model artifacts. Make sure the `deployment_artifacts/` "
        "folder (from the export script) is in the same repo as this app.py.\n\n"
        f"Details: {e}"
    )
    st.stop()


# ------------------------------------------------------------
# Helper: build number_input widgets for a list of features
# ------------------------------------------------------------
def build_input_form(feature_names, key_prefix):
    values = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        stats = feature_stats.get(feat, {"min": -100.0, "max": 1000.0, "mean": 0.0})
        with cols[i % 3]:
            values[feat] = st.number_input(
                feat,
                min_value=float(stats["min"]),
                max_value=float(stats["max"]) if stats["max"] > stats["min"] else float(stats["min"]) + 1.0,
                value=float(stats["mean"]),
                key=f"{key_prefix}_{feat}",
            )
    return values


def to_row(values, feature_order):
    return pd.DataFrame([[values[f] for f in feature_order]], columns=feature_order)


# ------------------------------------------------------------
# Tabs: one per prediction task
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["⚡ AC Power Prediction", "🔍 Fault Detection", "🧩 Fault Type"])

with tab1:
    st.subheader("Predict expected AC power output")
    ac_features = feature_lists["ac_power_features"]
    ac_values = build_input_form(ac_features, "ac")

    if st.button("Predict AC Power", type="primary"):
        row = to_row(ac_values, ac_features)
        pred = ac_model.predict(row)[0]
        st.success(f"Predicted AC Power: **{pred:,.2f}**")

with tab2:
    st.subheader("Is this reading faulted?")
    fault_det_features = feature_lists["is_faulted_features"]
    fault_det_values = build_input_form(fault_det_features, "isf")

    if st.button("Check for Fault", type="primary"):
        row = to_row(fault_det_values, fault_det_features)
        pred = is_faulted_model.predict(row)[0]
        proba = is_faulted_model.predict_proba(row)[0][1]

        if int(pred) == 1:
            st.error(f"⚠️ Fault detected (confidence: {proba:.1%})")
        else:
            st.success(f"✅ No fault detected (confidence: {1 - proba:.1%})")

with tab3:
    st.subheader("If faulted, which fault type is it?")
    st.caption("Run this once a fault has been flagged in the tab above.")
    fault_type_features = feature_lists["fault_type_features"]
    fault_type_values = build_input_form(fault_type_features, "ft")

    if st.button("Classify Fault Type", type="primary"):
        row = to_row(fault_type_values, fault_type_features)

        st.write("**Probability of each fault type:**")
        scores = {}
        for target, model in fault_models.items():
            proba = model.predict_proba(row)[0][1]
            scores[target] = proba

        scores_df = pd.DataFrame(
            {"Fault Type": list(scores.keys()), "Probability": list(scores.values())}
        ).sort_values("Probability", ascending=False)

        st.bar_chart(scores_df.set_index("Fault Type"))
        st.dataframe(scores_df, use_container_width=True, hide_index=True)

        top_fault = scores_df.iloc[0]
        st.info(f"Most likely: **{top_fault['Fault Type']}** ({top_fault['Probability']:.1%})")

st.divider()
st.caption(
    "Note: fields like `_lag1`, `_roll3`, or `_diff` represent engineered features "
    "(previous reading / rolling average / change from last reading). Enter them "
    "as if you already computed them for this reading."
)
