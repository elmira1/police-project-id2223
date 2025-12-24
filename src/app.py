import os
import json
import streamlit as st
import pandas as pd
import hopsworks
import joblib

# -----------------------------
# Config
# -----------------------------
try:
    import config
    API_KEY = config.HOPSWORKS_API_KEY
    PROJECT_NAME = config.HOPSWORKS_PROJECT_NAME
except Exception:
    API_KEY = os.environ.get("HOPSWORKS_API_KEY")
    PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME", "id2223_lab1_G22")

MODEL_NAME = "police_crime_model"
MODEL_VERSION = int(os.environ.get("MODEL_VERSION", "13"))  # you can override via env

if not API_KEY:
    st.error("Missing HOPSWORKS_API_KEY (set it in env vars or src/config.py).")
    st.stop()

st.set_page_config(page_title="Police Event Predictor", page_icon="👮", layout="wide")
st.title("👮 Swedish Police Events Predictor")
st.write("Predict police event type based on **city**, **day**, **hour**, and **precipitation**.")


# -----------------------------
# Load model + optional assets
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_assets():
    """
    Download model artifacts from Hopsworks and load the sklearn Pipeline (model.pkl).
    Also load meta.json / confusion_matrix.png if they exist.
    """
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    mr = project.get_model_registry()

    model_obj = mr.get_model(MODEL_NAME, version=MODEL_VERSION)
    model_dir = model_obj.download()

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


@st.cache_data(show_spinner=False, ttl=600)
def load_city_list():
    """
    Read unique cities from the feature group.
    Cached for 10 minutes to reduce Hopsworks reads.
    """
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("police_events", version=1)
    df = fg.read()
    return sorted(df["city"].dropna().unique().tolist())


with st.spinner("Loading model..."):
    try:
        model, meta, cm_path = load_model_and_assets()
        st.success(f"Model loaded: {MODEL_NAME} (v{MODEL_VERSION})")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

cities = load_city_list()
if not cities:
    st.warning("No cities found in Feature Group police_events.")
    st.stop()


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Inputs")
selected_city = st.sidebar.selectbox("City", cities)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
selected_day = st.sidebar.selectbox("Day of week", days)

selected_hour = st.sidebar.slider("Hour", 0, 23, 12)
precip = st.sidebar.slider("Precipitation (mm)", 0.0, 30.0, 0.0, 0.1)


# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction")

    if st.button("Predict"):
        X = pd.DataFrame([{
            "city": selected_city,
            "day_of_week": selected_day,
            "hour": int(selected_hour),
            "precipitation": float(precip),
        }])

        try:
            pred = model.predict(X)[0]
            st.markdown(f"### 🚨 {pred}")

            # Probabilities (RandomForest supports predict_proba)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]

                # Get class order from the final estimator in the pipeline
                classes = None
                try:
                    # In your training pipeline, the last step name is "model"
                    if hasattr(model, "named_steps") and "model" in model.named_steps:
                        classes = model.named_steps["model"].classes_
                    else:
                        # fallback: last step
                        last_step_name = list(model.named_steps.keys())[-1]
                        classes = model.named_steps[last_step_name].classes_
                except Exception:
                    classes = None

                if classes is None:
                    classes = [f"class_{i}" for i in range(len(proba))]

                dfp = (
                    pd.DataFrame({"class": classes, "prob": proba})
                    .sort_values("prob", ascending=False)
                    .head(10)
                    .reset_index(drop=True)
                )

                st.write("Top probabilities:")
                st.dataframe(dfp, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


with col2:
    st.subheader("Model info")

    if meta:
        st.write("Training metadata:")
        st.json(meta)
    else:
        st.info("No meta.json found in the model artifacts (retrain with updated training script).")

    if cm_path:
        st.write("Confusion matrix (from training):")
        st.image(cm_path, use_container_width=True)
    else:
        st.info("No confusion_matrix.png found in the model artifacts (retrain with updated training script).")


st.divider()
st.subheader("Latest batch forecast chart (optional)")

if os.path.exists("crime_forecast.png"):
    st.image("crime_forecast.png", caption="Generated by inference pipeline", use_container_width=True)
else:
    st.caption("crime_forecast.png not found (run inference pipeline to generate it).")
