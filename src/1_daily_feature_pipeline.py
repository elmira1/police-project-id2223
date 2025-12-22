import os
import requests
import pandas as pd
import hopsworks
import json
import time
from geopy.geocoders import Nominatim

# --- Login Settings ---
api_key = None
project_name = "id2223_lab1_G22"
CACHE_FILE = "src/city_coords.json" # Path for your JSON cache

try:
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except ImportError:
    api_key = os.environ.get('HOPSWORKS_API_KEY')

if api_key is None:
    raise Exception("API Key not found!")

# --- NEW FUNCTION: Smart City Coordinates with JSON Cache ---
def get_smart_coordinates(city_list):
    """
    Check JSON file for coordinates. If city is new, use Geocoder and update JSON.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding='utf-8') as f:
            try:
                city_cache = json.load(f)
            except:
                city_cache = {}
    else:
        city_cache = {}

    geolocator = Nominatim(user_agent="sweden_police_app_v2")
    has_updates = False

    for city in city_list:
        if city not in city_cache:
            print(f"🔍 New city detected: '{city}'. Fetching coordinates...")
            try:
                location = geolocator.geocode(f"{city}, Sweden")
                if location:
                    city_cache[city] = {"lat": location.latitude, "lon": location.longitude}
                    has_updates = True
                    time.sleep(1.1) # Respect API limits
                else:
                    city_cache[city] = {"lat": 59.3293, "lon": 18.0686} # Fallback
            except:
                city_cache[city] = {"lat": 59.3293, "lon": 18.0686}

    if has_updates:
        with open(CACHE_FILE, "w", encoding='utf-8') as f:
            json.dump(city_cache, f, indent=4, ensure_ascii=False)
            print("✅ JSON cache updated.")
    
    return city_cache

# --- NEW FUNCTION: Fetch Weather for each city ---
def fetch_weather(df, coords_map):
    """
    Fetch precipitation data for each city using coordinates.
    """
    print("🌦️ Fetching weather data...")
    weather_map = {}
    for city in df['city'].unique():
        lat = coords_map[city]['lat']
        lon = coords_map[city]['lon']
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&timezone=auto&forecast_days=1"
        try:
            res = requests.get(url).json()
            weather_map[city] = res['daily']['precipitation_sum'][0]
        except:
            weather_map[city] = 0.0
    return weather_map

def get_police_data():
    url = "https://polisen.se/api/events"
    print(f"Downloading data from: {url}")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        
        # 1. Get city name from location
        df['city'] = df['location'].apply(lambda x: x['name'])
        
        # --- SMART ADDITION: COORDINATES & WEATHER ---
        coords_map = get_smart_coordinates(df['city'].unique())
        weather_map = fetch_weather(df, coords_map)
        df['precipitation'] = df['city'].map(weather_map)
        # ----------------------------------------------
        
        # 2. Fix date and time
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        
        # 3. Select final columns (Added precipitation)
        df = df[['id', 'datetime', 'type', 'city', 'hour', 'day_of_week', 'precipitation']]
        df['id'] = df['id'].astype(str)
        
        return df
    return None

def to_hopsworks(df):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    
    police_fg = fs.get_or_create_feature_group(
        name="police_events",
        version=1,
        primary_key=["id"],
        description="Police events with weather data"
    )
    
    print("Uploading data...")
    police_fg.insert(df)
    print("✅ Success! Data with weather uploaded.")

if __name__ == "__main__":
    df = get_police_data()
    if df is not None:
        to_hopsworks(df)