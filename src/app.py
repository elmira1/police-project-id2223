# src/app.py
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

if not API_KEY:
    st.error("Missing HOPSWORKS_API_KEY (set it in env vars or src/config.py).")
    st.stop()


# -----------------------------
# Page
# -----------------------------
st.set_page_config(page_title="Police Event Predictor", page_icon="👮", layout="wide")
st.title("👮 Swedish Police Events Predictor")
st.write("Predict police event type based on **city**, **day**, **hour**, and **precipitation**.")


# -----------------------------
# Helpers
# -----------------------------
def get_best_model(mr):
    """Pick best model by balanced_accuracy, fallback to accuracy."""
    try:
        return mr.get_best_model(MODEL_NAME, "balanced_accuracy", "max")
    except Exception:
        return mr.get_best_model(MODEL_NAME, "accuracy", "max")


@st.cache_resource(show_spinner=False)
def load_model_assets():
    """
    Download model artifacts from Hopsworks and load the sklearn Pipeline (model.pkl).
    Also load meta.json and confusion_matrix.png if available.
    """
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    mr = project.get_model_registry()

    best = get_best_model(mr)
    model_dir = best.download()
    pipeline = joblib.load(os.path.join(model_dir, "model.pkl"))

    meta = None
    meta_path = os.path.join(model_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    if not os.path.exists(cm_path):
        cm_path = None

    return best.version, pipeline, meta, cm_path


@st.cache_data(show_spinner=False, ttl=600)
def load_city_list():
    """
    Load unique cities from the feature group.
    No 'project' arg here -> avoids Streamlit unhashable cache error.
    """
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("police_events", version=1)

    # Read only city column (faster + safer)
    df = fg.select(["city"]).read()
    cities = sorted(df["city"].dropna().unique().tolist())
    return cities


# -----------------------------
# Load model
# -----------------------------
with st.spinner("Loading model from Hopsworks..."):
    try:
        model_version, model, meta, cm_path = load_model_assets()
        st.success(f"Model loaded: {MODEL_NAME} (best version = v{model_version})")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


# -----------------------------
# Sidebar inputs
# -----------------------------
cities = load_city_list()
if not cities:
    st.warning("No cities found in Feature Group police_events.")
    st.stop()

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
        X = pd.DataFrame(
            [
                {
                    "city": selected_city,
                    "day_of_week": selected_day,
                    "hour": int(selected_hour),
                    "precipitation": float(precip),
                }
            ]
        )

        try:
            pred = model.predict(X)[0]
            st.markdown(f"### 🚨 {pred}")

            # Probabilities (if supported)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]

                # Get class names from the last estimator in the Pipeline
                estimator = model.steps[-1][1]
                classes = getattr(estimator, "classes_", None)

                if classes is None:
                    classes = [f"class_{i}" for i in range(len(proba))]

                dfp = (
                    pd.DataFrame({"class": list(classes), "prob": proba})
                    .sort_values("prob", ascending=False)
                    .reset_index(drop=True)  # remove confusing index
                )

                st.write("Top probabilities:")
                # If your Streamlit version supports it, hide_index=True is nice.
                try:
                    st.dataframe(dfp.head(10), use_container_width=True, hide_index=True)
                except TypeError:
                    st.dataframe(dfp.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("Model info")
    if meta:
        # show important metrics clearly
        st.metric("Accuracy", f"{meta.get('accuracy', 0):.3f}")
        st.metric("Balanced Accuracy", f"{meta.get('balanced_accuracy', 0):.3f}")
        st.caption(f"Classes kept (min_count={meta.get('min_count')}): {len(meta.get('classes', []))}")
    else:
        st.info("No meta.json found in model artifacts.")

    if cm_path:
        st.write("Confusion matrix (from training):")
        st.image(cm_path, use_container_width=True)
    else:
        st.info("No confusion_matrix.png found in model artifacts.")


# -----------------------------
# Optional dashboard artifacts created by GitHub Actions
# -----------------------------
st.divider()
st.subheader("Daily artifacts (from GitHub Actions)")

a1, a2 = st.columns(2)

with a1:
    if os.path.exists("crime_forecast.png"):
        st.image("crime_forecast.png", caption="crime_forecast.png", use_container_width=True)
    else:
        st.caption("crime_forecast.png not found yet.")

with a2:
    if os.path.exists("predicted_type_distribution.png"):
        st.image("predicted_type_distribution.png", caption="predicted_type_distribution.png", use_container_width=True)
    else:
        st.caption("predicted_type_distribution.png not found yet.")

b1, b2 = st.columns(2)
with b1:
    if os.path.exists("monitor_confusion_matrix.png"):
        st.image("monitor_confusion_matrix.png", caption="monitor_confusion_matrix.png", use_container_width=True)
    else:
        st.caption("monitor_confusion_matrix.png not found yet.")

with b2:
    if os.path.exists("monitor_recent.csv"):
        try:
            df_mon = pd.read_csv("monitor_recent.csv")
            st.write("monitor_recent.csv (last rows):")
            st.dataframe(df_mon.tail(30), use_container_width=True)
        except Exception:
            st.caption("monitor_recent.csv exists but could not be read.")
    else:
        st.caption("monitor_recent.csv not found yet.")
