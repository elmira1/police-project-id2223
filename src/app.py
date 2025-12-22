import streamlit as st
import hopsworks
import joblib
import pandas as pd
import numpy as np
import os

# --- Login Settings ---
api_key = None
project_name = "id2223_lab1_G22"

try:
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except ImportError:
    try:
        from src import config
        api_key = config.HOPSWORKS_API_KEY
        project_name = config.HOPSWORKS_PROJECT_NAME
    except ImportError:
        api_key = os.environ.get('HOPSWORKS_API_KEY')

if api_key is None:
    # If no key, stop the app
    st.error("API Key not found!")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Police Event Predictor", page_icon="👮")

st.title("👮 Swedish Police Events Predictor")
st.write("Predict crime type based on City and Time.")

# --- 1. Load Model from Cloud ---
@st.cache_resource()
def get_model():
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    mr = project.get_model_registry()
    model = mr.get_model("police_crime_model", version=1)
    return model.download()

with st.spinner("Downloading model..."):
    try:
        model_dir = get_model()
        model = joblib.load(os.path.join(model_dir, "police_model.pkl"))
        le_city = joblib.load(os.path.join(model_dir, "city_encoder.pkl"))
        le_day = joblib.load(os.path.join(model_dir, "day_encoder.pkl"))
        le_type = joblib.load(os.path.join(model_dir, "type_encoder.pkl"))
        st.success("System Ready!")
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# --- 2. User Inputs ---
st.sidebar.header("Settings")

cities = list(le_city.classes_)
selected_city = st.sidebar.selectbox("City", cities)

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_day = st.sidebar.selectbox("Day", days)

selected_hour = st.sidebar.slider("Hour", 0, 23, 12)

# --- 3. Prediction Logic ---
if st.button("Predict"):
    # Convert inputs to numbers
    city_num = le_city.transform([selected_city])[0]
    day_num = le_day.transform([selected_day])[0]
    
    features = pd.DataFrame({
        'city_encoded': [city_num],
        'hour': [selected_hour],
        'day_encoded': [day_num]
    })
    
    # Predict
    pred_num = model.predict(features)
    pred_text = le_type.inverse_transform(pred_num)[0]
    
    st.subheader(f"Result for {selected_city}:")
    st.markdown(f"### 🚨 {pred_text}")
    
    # --- Chart ---
    st.write("Confidence:")
    probs = model.predict_proba(features)
    
    # Only show classes that the model learned
    learned_classes = le_type.inverse_transform(model.classes_)
    
    chart_data = pd.DataFrame(probs, columns=learned_classes)
    st.bar_chart(chart_data.T)