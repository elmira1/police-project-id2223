# src/1_daily_feature_pipeline.py
import os
import json
import time
import requests
import pandas as pd
import hopsworks
from geopy.geocoders import Nominatim

# ✅ Add label grouping
from label_map import add_type_group


# -----------------------------
# Config / Secrets
# -----------------------------
api_key = None
project_name = "id2223_lab1_G22"

FG_VERSION = 6

CACHE_FILE = "src/city_coords.json"            # city -> {lat, lon}
WEATHER_CACHE_FILE = "src/weather_cache.json"  # (city|yyyy-mm-dd) -> precipitation

try:
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except Exception:
    api_key = os.environ.get("HOPSWORKS_API_KEY")
    project_name = os.environ.get("HOPSWORKS_PROJECT_NAME", project_name)

if not api_key:
    raise RuntimeError("API Key not found! Set HOPSWORKS_API_KEY.")

SESSION = requests.Session()


# -----------------------------
# Utilities: load/save JSON cache
# -----------------------------
def _load_json(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# City coordinates (cached)
# -----------------------------
def get_smart_coordinates(city_list):
    city_cache = _load_json(CACHE_FILE)

    geolocator = Nominatim(user_agent="id2223-police-project-g22")
    updated = False

    for city in sorted(set(city_list)):
        if not city or str(city).strip() == "":
            continue

        if city not in city_cache:
            print(f"🔍 New city: '{city}' -> geocoding...")
            try:
                location = geolocator.geocode(f"{city}, Sweden", timeout=10)
                if location:
                    city_cache[city] = {
                        "lat": float(location.latitude),
                        "lon": float(location.longitude),
                    }
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


# -----------------------------
# Weather (historical) via Open-Meteo Archive API (cached)
# -----------------------------
def get_weather_for_city_date(
    city: str,
    lat: float,
    lon: float,
    yyyy_mm_dd: str,
    weather_cache: dict,
) -> float:
    key = f"{city}|{yyyy_mm_dd}"
    if key in weather_cache:
        return (float(v) for v in weather_cache[key])

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={yyyy_mm_dd}&end_date={yyyy_mm_dd}"
        "&daily=precipitation_sum"
        "&daily=temperature_2m_mean"
        "&daily=wind_speed_10m_mean"
        "&timezone=UTC"
    )

    try:
        res = SESSION.get(url, timeout=20).json()
        precip = float(res["daily"]["precipitation_sum"][0])
        temp = float(res["daily"]["temperature_2m_mean"][0])
        wind = float(res["daily"]["wind_speed_10m_mean"][0])
    except Exception:
        precip = 0.0
        temp = 0.0
        wind = 0.0

    weather_cache[key] = precip, temp, wind
    return precip, temp, wind


def add_weather_feature(df: pd.DataFrame, coords_map: dict) -> pd.DataFrame:
    print("🌦️ Adding historical weather (Archive API)...")
    weather_cache = _load_json(WEATHER_CACHE_FILE)
    updated = False

    # string date for caching
    df["event_date"] = df["datetime"].dt.date.astype(str)

    pairs = df[["city", "event_date"]].drop_duplicates()

    weather_map = {}
    for _, row in pairs.iterrows():
        city = row["city"]
        d = row["event_date"]
        lat = coords_map.get(city, {}).get("lat", 59.3293)
        lon = coords_map.get(city, {}).get("lon", 18.0686)

        precip, temp, wind = get_weather_for_city_date(city, lat, lon, d, weather_cache)
        weather_map[f"{city}|{d}"] = precip, temp, wind
        updated = True

    df["precipitation"] = df.apply(
        lambda r: float(weather_map.get(f"{r['city']}|{r['event_date']}", 0.0)[0]),
        axis=1,
    )
    df["temperature"] = df.apply(
        lambda r: float(weather_map.get(f"{r['city']}|{r['event_date']}", 0.0)[1]),
        axis=1,
    )
    df["wind"] = df.apply(
        lambda r: float(weather_map.get(f"{r['city']}|{r['event_date']}", 0.0)[2]),
        axis=1,
    )

    if updated:
        _save_json(WEATHER_CACHE_FILE, weather_cache)
        print("✅ Weather cache updated.")

    df = df.drop(columns=["event_date"], errors="ignore")
    return df


# -----------------------------
# Police data
# -----------------------------
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

    # city from location.name
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
    coords_map = get_smart_coordinates(df["city"].dropna().unique())
    df = add_weather_feature(df, coords_map)

    # keep only needed columns
    df = df[["id", "datetime", "city", "day_of_week", "hour", "precipitation", "temperature", "wind", "type"]].copy()

    # types
    df["id"] = df["id"].astype(str)
    df["city"] = df["city"].astype(str)
    df["type"] = df["type"].astype(str)
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0.0).astype(float)
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").fillna(0.0).astype(float)
    df["wind"] = pd.to_numeric(df["wind"], errors="coerce").fillna(0.0).astype(float)

    # protect against duplicates inside same run
    df = df.drop_duplicates(subset=["id"])

    return df


# -----------------------------
# Hopsworks
# -----------------------------
def to_hopsworks(df: pd.DataFrame):
    print("🔌 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()

    # ✅ Add grouped label BEFORE insert
    df = add_type_group(df, src_col="type", dst_col="type_group")

    # ✅ Ensure schema/columns are exactly what we want
    required_cols = ["id", "datetime", "city", "day_of_week", "hour", "precipitation", "temperature", "wind", "type", "type_group"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns before insert: {missing}")

    df = df[required_cols].copy()

    police_fg = fs.get_or_create_feature_group(
        name="police_events",
        version=FG_VERSION,
        primary_key=["id"],
        event_time="datetime",
        description="Police events with weather + grouped label (type_group)",
    )

    print(f"⬆️ Uploading {len(df)} rows...")
    police_fg.insert(df, write_options={"wait_for_job": False})
    print("✅ Done. Features uploaded.")


if __name__ == "__main__":
    df = get_police_data()
    if df is not None and not df.empty:
        to_hopsworks(df)
    else:
        print("⚠️ Nothing to upload.")
