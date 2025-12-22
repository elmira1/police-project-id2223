import os
import pandas as pd
import hopsworks
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests

# --- Settings & Path ---
api_key = None
project_name = "id2223_lab1_G22"
CACHE_FILE = "src/city_coords.json" # NEW: To get coordinates for weather

try:
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except ImportError:
    api_key = os.environ.get('HOPSWORKS_API_KEY')

if api_key is None:
    raise Exception("API Key not found!")

# NEW: Helper function to get tomorrow's rain forecast
def get_tomorrow_weather(city_list):
    if not os.path.exists(CACHE_FILE):
        return {city: 0.0 for city in city_list}
    
    with open(CACHE_FILE, "r", encoding='utf-8') as f:
        coords_map = json.load(f)
    
    weather_forecast = {}
    print("🌦️ Fetching tomorrow's weather forecast for all cities...")
    for city in city_list:
        if city in coords_map:
            lat, lon = coords_map[city]['lat'], coords_map[city]['lon']
            # forecast_days=2 gives today and tomorrow
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=auto&forecast_days=2"
            try:
                res = requests.get(url).json()
                weather_forecast[city] = res['daily']['precipitation_sum'][1] # [1] is tomorrow
            except:
                weather_forecast[city] = 0.0
        else:
            weather_forecast[city] = 0.0
    return weather_forecast

def inference():
    # 1. Connect
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # 2. Download Model (NEW: Use the version trained with weather!)
    print("📥 Downloading smart model assets...")
    retrieved_model = mr.get_model(name="police_crime_model", version=10) # UPDATE VERSION IF NEEDED
    saved_model_dir = retrieved_model.download()
    
    model = joblib.load(saved_model_dir + "/police_model.pkl")
    le_city = joblib.load(saved_model_dir + "/city_encoder.pkl")
    le_day = joblib.load(saved_model_dir + "/day_encoder.pkl")
    le_type = joblib.load(saved_model_dir + "/type_encoder.pkl")
    
    # 3. Create Batch Data
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow_str = tomorrow.strftime('%Y-%m-%d')
    day_name = tomorrow.strftime('%A')
    
    target_cities = le_city.classes_
    weather_map = get_tomorrow_weather(target_cities) # NEW: Get forecast
    
    batch_data = []
    print(f"🔮 Generating predictions for {len(target_cities)} cities...")
    
    for city in target_cities:
        rain = weather_map.get(city, 0.0)
        for hour in range(24):
            batch_data.append({
                "date": tomorrow_str,
                "city": city,
                "day_of_week": day_name,
                "hour": hour,
                "precipitation": rain # NEW: Add rain feature
            })
            
    df_batch = pd.DataFrame(batch_data)
    
    # 4. Encoding
    df_batch['day_of_week_encoded'] = le_day.transform(df_batch['day_of_week'])
    df_batch['city_encoded'] = le_city.transform(df_batch['city'])
    
    # NEW: Select Features matching the Training Pipeline
    X_pred = df_batch[['city_encoded', 'hour', 'day_of_week_encoded', 'precipitation']]
    
    # 5. Predict
    y_pred_encoded = model.predict(X_pred)
    df_batch['predicted_crime_type'] = le_type.inverse_transform(y_pred_encoded)
    
# 1. FIX: Convert 'date' string to actual datetime objects
    df_batch['date'] = pd.to_datetime(df_batch['date'])

    # 2. FIX: Drop the old columns that cause conflicts in version 1
    # We keep only what the model and FG need
    cols_to_keep = ["date", "city", "hour", "day_of_week", "precipitation", "predicted_crime_type", "day_of_week_encoded"]
    df_to_insert = df_batch[cols_to_keep]

    # 6. Save to Hopsworks
    print("💾 Saving predictions to Feature Store...")
    
    # 3. FIX: Use Version 2 to create a fresh schema with weather data
    pred_fg = fs.get_or_create_feature_group(
        name="police_predictions",
        version=2,  # <--- Change this from 1 to 2
        primary_key=["date", "city", "hour"],
        event_time="date",
        description="Daily crime predictions with weather data"
    )
    
    pred_fg.insert(df_to_insert, write_options={"wait_for_job": False})
    
    # 7. Chart: Top Cities with predicted rain and crime
    # Show top 10 rainiest cities and their predicted most frequent crime
    city_summary = df_batch.groupby('city').agg({
        'precipitation': 'mean',
        'predicted_crime_type': lambda x: x.mode()[0]
    }).sort_values('precipitation', ascending=False).head(10)

    plt.figure(figsize=(14, 7))
    sns.barplot(x=city_summary.index, y=city_summary['precipitation'], palette="Blues_d")
    
    # Add crime labels on top of bars
    for i, city in enumerate(city_summary.index):
        plt.text(i, city_summary['precipitation'][i], city_summary['predicted_crime_type'][i], 
                 rotation=45, ha='center', va='bottom', fontsize=9)

    plt.title(f"Predicted Weather & Crime for {tomorrow_str} (Top 10 Rainiest Cities)")
    plt.ylabel("Expected Rain (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("crime_forecast.png")
    
    print("🎉 All done! Smart predictions are live.")

if __name__ == "__main__":
    inference()