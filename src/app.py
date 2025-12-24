import os
import json
import streamlit as st
import pandas as pd
import hopsworks
import joblib

# -----------------------------
# Config
# -----------------------------
# Simple rule:
# - Local run (VS Code): use src/config.py
# - HF Spaces: use environment variables
try:
    import config
    API_KEY = config.HOPSWORKS_API_KEY
    PROJECT_NAME = config.HOPSWORKS_PROJECT_NAME
except Exception:
    try:
        from src import config  # sometimes config is inside src/
        API_KEY = config.HOPSWORKS_API_KEY
        PROJECT_NAME = config.HOPSWORKS_PROJECT_NAME
    except Exception:
        API_KEY = os.environ.get("HOPSWORKS_API_KEY")
        PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME", "id2223_lab1_G22")

MODEL_NAME = "police_crime_model"
MODEL_VERSION = 13  # <-- set to the model version you want

if not API_KEY:
    st.error("Missing HOPSWORKS_API_KEY (set env var or src/config.py).")
    st.stop()

# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Police Event Predictor", page_icon="👮", layout="wide")
st.title("👮 Swedish Police Events Predictor")
st.write("Predict police event type based on **city**, **day of week**, **hour**, and **precipitation**.")

# -----------------------------
# Load model + artifacts
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_artifacts():
    """
    Download model from Hopsworks Model Registry and load the sklearn Pipeline (model.pkl).
    Also read meta.json + confusion_matrix.png if they exist in the model folder.
    """
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    mr = project.get_model_registry()

    model_obj = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
    model_dir = model_obj.download()

    # We saved the full sklearn Pipeline during training as model.pkl
    pipeline = joblib.load(os.path.join(model_dir, "model.pkl"))

    meta = None
    meta_path = os.path.join(model_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    if not os.path.exists(cm_path):
        cm_path = None

    return pipeline, meta, cm_path


with st.spinner("Downloading model from Hopsworks..."):
    try:
        model, meta, train_cm_path = load_model_and_artifacts()
        st.success(f"Model loaded: {MODEL_NAME} (v{MODEL_VERSION})")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# -----------------------------
# Load cities from Feature Group
# -----------------------------
@st.cache_data(show_spinner=False, ttl=600)
def load_cities():
    """
    IMPORTANT:
    Do NOT pass Hopsworks 'project' into cached functions (it is not hashable).
    We login inside the function instead.
    Cached for 10 minutes.
    """
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("police_events", version=1)
    df = fg.read()

    cities = sorted(df["city"].dropna().astype(str).unique().tolist())
    return cities


cities = load_cities()
if not cities:
    st.warning("No cities found in Feature Group police_events.")
    st.stop()

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Inputs")

selected_city = st.sidebar.selectbox("City", cities)
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Day of week", days)
selected_hour = st.sidebar.slider("Hour", 0, 23, 12)
precip = st.sidebar.slider("Precipitation (mm)", 0.0, 30.0, 0.0, 0.1)

# -----------------------------
# Helpers for probabilities/classes
# -----------------------------
def get_estimator_classes(pipeline):
    """
    Try to get 'classes_' from the last estimator in the Pipeline.
    Works when your last step is RandomForestClassifier (or similar).
    """
    try:
        # Typical: pipeline.named_steps["model"]
        if hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
            est = pipeline.named_steps["model"]
            return list(getattr(est, "classes_", []))
        # Fallback: take the last step
        if hasattr(pipeline, "steps") and len(pipeline.steps) > 0:
            est = pipeline.steps[-1][1]
            return list(getattr(est, "classes_", []))
    except Exception:
        pass
    return []


# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Prediction")

    if st.button("Predict"):
        # IMPORTANT: Our training pipeline expects raw columns:
        # city, day_of_week, hour, precipitation
        X = pd.DataFrame([{
            "city": str(selected_city),
            "day_of_week": str(selected_day),
            "hour": int(selected_hour),
            "precipitation": float(precip),
        }])

        try:
            pred = model.predict(X)[0]
            st.markdown(f"### 🚨 {pred}")

            # Show probabilities if model supports it
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = get_estimator_classes(model)

                # If classes list is empty, create generic names
                if not classes or len(classes) != len(proba):
                    classes = [f"class_{i}" for i in range(len(proba))]

                dfp = (
                    pd.DataFrame({"class": classes, "probability": proba})
                    .sort_values("probability", ascending=False)
                    .reset_index(drop=True)  # removes the confusing first index column
                )

                st.write("Top probabilities:")
                st.dataframe(dfp.head(12), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with right:
    st.subheader("Model info")

    # Instead of big blue JSON boxes, show small metrics + optional raw json in expander
    if meta:
        colA, colB = st.columns(2)
        with colA:
            st.metric("Accuracy", f"{meta.get('accuracy', 0):.3f}")
        with colB:
            st.metric("Balanced Accuracy", f"{meta.get('balanced_accuracy', 0):.3f}")

        st.caption(f"min_count = {meta.get('min_count', '-')}, classes = {len(meta.get('classes', []))}")

        with st.expander("Show meta.json"):
            st.json(meta)
    else:
        st.info("No meta.json found in the model artifacts.")

    if train_cm_path:
        st.write("Confusion matrix (from training):")
        st.image(train_cm_path, use_container_width=True)
    else:
        st.info("No confusion_matrix.png found in the model artifacts.")

# -----------------------------
# Show latest batch/monitoring artifacts if they exist in repo
# -----------------------------
st.divider()
st.subheader("Latest batch + monitoring outputs (if available)")

c1, c2 = st.columns(2)

with c1:
    st.write("Batch forecast charts")
    if os.path.exists("crime_forecast.png"):
        st.image("crime_forecast.png", caption="crime_forecast.png (from inference pipeline)", use_container_width=True)
    else:
        st.caption("crime_forecast.png not found (run inference pipeline / commit it).")

    if os.path.exists("predicted_type_distribution.png"):
        st.image(
            "predicted_type_distribution.png",
            caption="predicted_type_distribution.png (tomorrow predicted types)",
            use_container_width=True
        )
    else:
        st.caption("predicted_type_distribution.png not found (optional).")

with c2:
    st.write("Monitoring (latest real events)")
    if os.path.exists("monitor_confusion_matrix.png"):
        st.image(
            "monitor_confusion_matrix.png",
            caption="monitor_confusion_matrix.png (last real events monitoring)",
            use_container_width=True
        )
    else:
        st.caption("monitor_confusion_matrix.png not found (optional).")

    if os.path.exists("monitor_recent.csv"):
        try:
            df_mon = pd.read_csv("monitor_recent.csv")
            st.write("Recent monitoring table (last rows):")
            st.dataframe(df_mon.tail(30), use_container_width=True, hide_index=True)
        except Exception as e:
            st.caption(f"Could not read monitor_recent.csv: {e}")
    else:
        st.caption("monitor_recent.csv not found (optional).")
