import os
import requests
import pandas as pd
import hopsworks

# --- Login Settings ---
api_key = None
project_name = "id2223_lab1_G22"

try:
    # Try to load from local config file
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except ImportError:
    try:
        # Try importing if running from root folder
        from src import config
        api_key = config.HOPSWORKS_API_KEY
        project_name = config.HOPSWORKS_PROJECT_NAME
    except ImportError:
        # If running on GitHub, get secret key
        api_key = os.environ.get('HOPSWORKS_API_KEY')

if api_key is None:
    raise Exception("API Key not found! Check config.py or GitHub Secrets.")

def get_police_data():
    url = "https://polisen.se/api/events"
    print(f"Downloading data from: {url}")
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        
        # --- Data Cleaning ---
        
        # 1. Get city name from location
        df['city'] = df['location'].apply(lambda x: x['name'])
        
        # 2. Get GPS coordinates
        df['gps'] = df['location'].apply(lambda x: x['gps'])
        df[['lat', 'long']] = df['gps'].str.split(',', expand=True)
        
        # 3. Fix date and time
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.day_name()
        
        # 4. Select useful columns
        df = df[['id', 'datetime', 'type', 'city', 'lat', 'long', 'hour', 'day_of_week']]
        
        # ID must be string for Hopsworks
        df['id'] = df['id'].astype(str)
        
        return df
    else:
        print("Error: Could not download data.")
        return None

def to_hopsworks(df):
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    
    # Create or get the table (Feature Group)
    police_fg = fs.get_or_create_feature_group(
        name="police_events",
        version=1,
        primary_key=["id"],
        description="Daily police events in Sweden"
    )
    
    print("Uploading data...")
    police_fg.insert(df)
    print("✅ Success! Data uploaded.")

if __name__ == "__main__":
    df = get_police_data()
    if df is not None:
        to_hopsworks(df)