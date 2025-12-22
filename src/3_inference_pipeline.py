import os
import pandas as pd
import hopsworks
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
    raise Exception("API Key not found!")

def inference():
    # 1. Connect to Hopsworks
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # 2. Download Model & Encoders
    print("📥 Downloading model assets...")
    retrieved_model = mr.get_model(name="police_crime_model", version=1)
    saved_model_dir = retrieved_model.download()
    
    model = joblib.load(saved_model_dir + "/police_model.pkl")
    le_city = joblib.load(saved_model_dir + "/city_encoder.pkl")
    le_day = joblib.load(saved_model_dir + "/day_encoder.pkl")
    le_type = joblib.load(saved_model_dir + "/type_encoder.pkl")
    
    # 3. Create Batch Data for ALL Known Cities
    print("🔮 Generating batch data for tomorrow...")
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')
    day_name = tomorrow.strftime('%A')
    
    # --- CHANGE HERE: Get ALL cities the model knows ---
    # le_city.classes_ gives us every city found in the training data
    target_cities = le_city.classes_
    print(f"📋 Found {len(target_cities)} cities in the trained model.")
    
    batch_data = []
    
    # Loop through ALL cities
    for city in target_cities:
        # We simulate 24 hours for each city
        for hour in range(24):
            batch_data.append({
                "date": tomorrow_str,
                "city": city,
                "day_of_week": day_name,
                "hour": hour
            })
            
    df_batch = pd.DataFrame(batch_data)
    df_batch['date'] = pd.to_datetime(df_batch['date'])
    
    # 4. Prepare Data (Encoding)
    print("⚙️ Encoding features...")
    
    # Encode Day
    try:
        df_batch['day_encoded'] = le_day.transform(df_batch['day_of_week'])
    except ValueError:
        # Just in case the day name is weird (should not happen with English locale)
        df_batch['day_encoded'] = 0 

    # Encode City (Safe now, because we picked cities FROM the encoder itself)
    df_batch['city_encoded'] = le_city.transform(df_batch['city'])
    
    # Select Features for Model
    X_pred = df_batch[['city_encoded', 'hour', 'day_encoded']]
    
    # 5. Predict
    print(f"🧠 Predicting crimes for {len(df_batch)} rows...")
    y_pred_encoded = model.predict(X_pred)
    
    # Decode results (Number -> Crime Name)
    df_batch['predicted_crime_type'] = le_type.inverse_transform(y_pred_encoded)
    
    # 6. Save to Hopsworks
    print("💾 Saving predictions to Feature Store...")
    
    pred_fg = fs.get_or_create_feature_group(
        name="police_predictions",
        version=1,
        description="Daily crime predictions for all Swedish cities in dataset",
        primary_key=["date", "city", "hour"],
        event_time="date"
    )
    
    pred_fg.insert(df_batch, write_options={"wait_for_job": False})
    
    # 7. Generate Summary Image (Show Top 5 Riskiest Cities)
    # Since we have too many cities now, we can't show all in one chart.
    # Let's show which cities have the most predicted events.
    print("📊 Generating summary chart...")
    
    # Count how many crimes are predicted per city for tomorrow
    city_crime_counts = df_batch.groupby('city')['predicted_crime_type'].count().sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=city_crime_counts.values, y=city_crime_counts.index, palette="magma")
    plt.title(f"Top 10 Cities with Highest Predicted Activity for {tomorrow_str}")
    plt.xlabel("Number of Predicted Events (24h)")
    plt.ylabel("City")
    plt.tight_layout()
    
    img_path = "crime_forecast.png"
    plt.savefig(img_path)
    plt.close()
    
    # 8. Upload Image
    print("☁️ Uploading image...")
    dataset_api = project.get_dataset_api()
    if not dataset_api.exists("Resources/police_project"):
        dataset_api.mkdir("Resources/police_project")
    dataset_api.upload(img_path, "Resources/police_project", overwrite=True)
    
    print("🎉 Done! Predictions for ALL cities saved.")

if __name__ == "__main__":
    inference()