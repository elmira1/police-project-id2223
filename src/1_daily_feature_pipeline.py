# src/1_daily_feature_pipeline.py
import os
import json
import time
import requests
import pandas as pd
import hopsworks
from datetime import date
from geopy.geocoders import Nominatim

# --- Login Settings ---
api_key = None
project_name = "id2223_lab1_G22"

CACHE_FILE = "src/city_coords.json"          # city -> {lat, lon}
WEATHER_CACHE_FILE = "src/weather_cache.json" # (city|yyyy-mm-dd) -> precipitation

try:
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except ImportError:
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT_NAME", project_name)

if not api_key:
    raise Exception("API Key not found! Set HOPSWORKS_API_KEY.")

SESSION = requests.Session()

# ------------------------
# Utilities: load/save cache
# ------------------------
def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ------------------------
# City coordinates (cached)
# ------------------------
def get_smart_coordinates(city_list):
    city_cache = _load_json(CACHE_FILE)

    geolocator = Nominatim(user_agent="id2223-police-project-g22")
    updated = False

    for city in sorted(set(city_list)):
        if not city or city.strip() == "":
            continue
        if city not in city_cache:
            print(f"🔍 New city: '{city}' -> geocoding...")
            try:
                location = geolocator.geocode(f"{city}, Sweden", timeout=10)
                if location:
                    city_cache[city] = {"lat": float(location.latitude), "lon": float(location.longitude)}
                else:
                    # Stockholm fallback
                    city_cache[city] = {"lat": 59.3293, "lon": 18.0686}
                updated = True
                time.sleep(1.1)  # be nice to Nominatim
            except Exception:
                city_cache[city] = {"lat": 59.3293, "lon": 18.0686}
                updated = True

    if updated:
        _save_json(CACHE_FILE, city_cache)
        print("✅ City coordinate cache updated.")

    return city_cache

# ------------------------
# Weather (historical) via Open-Meteo Archive API (cached)
# ------------------------
def get_precipitation_for_city_date(city: str, lat: float, lon: float, yyyy_mm_dd: str, weather_cache: dict) -> float:
    key = f"{city}|{yyyy_mm_dd}"
    if key in weather_cache:
        return float(weather_cache[key])

    # Archive API (historical) - daily precipitation sum
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={yyyy_mm_dd}&end_date={yyyy_mm_dd}"
        "&daily=precipitation_sum"
        "&timezone=UTC"
    )

    try:
        res = SESSION.get(url, timeout=20).json()
        precip = float(res["daily"]["precipitation_sum"][0])
    except Exception:
        precip = 0.0

    weather_cache[key] = precip
    return precip

def add_weather_feature(df: pd.DataFrame, coords_map: dict) -> pd.DataFrame:
    print("🌦️ Adding historical precipitation (Archive API)...")
    weather_cache = _load_json(WEATHER_CACHE_FILE)
    updated = False

    # create a date column
    df["event_date"] = df["datetime"].dt.date.astype(str)

    # fetch only unique (city, date)
    pairs = df[["city", "event_date"]].drop_duplicates()

    precip_map = {}
    for _, row in pairs.iterrows():
        city = row["city"]
        d = row["event_date"]
        lat = coords_map.get(city, {}).get("lat", 59.3293)
        lon = coords_map.get(city, {}).get("lon", 18.0686)
        precip = get_precipitation_for_city_date(city, lat, lon, d, weather_cache)
        precip_map[f"{city}|{d}"] = precip
        updated = True

    df["precipitation"] = df.apply(lambda r: precip_map.get(f"{r['city']}|{r['event_date']}", 0.0), axis=1)

    if updated:
        _save_json(WEATHER_CACHE_FILE, weather_cache)
        print("✅ Weather cache updated.")

    return df

# ------------------------
# Police data
# ------------------------
def get_police_data() -> pd.DataFrame | None:
    url = "https://polisen.se/api/events"
    print(f"⬇️ Downloading police events: {url}")

    try:
        response = SESSION.get(url, timeout=30)
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

    if response.status_code != 200:
        print(f"❌ Bad status: {response.status_code}")
        return None

    data = response.json()
    df = pd.DataFrame(data)

    if df.empty:
        print("⚠️ No data returned.")
        return None

    # city from location.name (safe)
    def _extract_city(loc):
        try:
            return loc.get("name", "Unknown")
        except Exception:
            return "Unknown"

    df["city"] = df["location"].apply(_extract_city)

    # parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()

    # coords + weather
    coords_map = get_smart_coordinates(df["city"].unique())
    df = add_weather_feature(df, coords_map)

    # final columns
    df = df[["id", "datetime", "type", "city", "hour", "day_of_week", "precipitation"]].copy()
    df["id"] = df["id"].astype(str)

    # protect against duplicates inside a single run
    df = df.drop_duplicates(subset=["id"])

    return df

# ------------------------
# Hopsworks
# ------------------------
def to_hopsworks(df: pd.DataFrame):
    print("🔌 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    # Better primary key: id + datetime (safer), and event_time set
    police_fg = fs.get_or_create_feature_group(
        name="police_events",
        version=1,
        primary_key=["id", "datetime"],
        event_time="datetime",
        description="Swedish Police events + historical precipitation (Open-Meteo Archive)"
    )

    print(f"⬆️ Uploading {len(df)} rows...")
    police_fg.insert(df, write_options={"wait_for_job": False})
    print("✅ Done. Features uploaded.")

if __name__ == "__main__":
    df = get_police_data()
    if df is not None and not df.empty:
        to_hopsworks(df)
