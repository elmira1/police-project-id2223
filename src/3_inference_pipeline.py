# src/3_inference_pipeline.py
import os
import json
import datetime as dt
import concurrent.futures

import pandas as pd
import requests
import hopsworks
import joblib

import matplotlib
matplotlib.use("Agg")  # safe for GitHub Actions / servers
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# -----------------------------
# Config
# -----------------------------
# Read Hopsworks credentials from src/config.py if available.
# Otherwise use environment variables (GitHub Actions / HF Spaces).
try:
    import config
    API_KEY = config.HOPSWORKS_API_KEY
    PROJECT_NAME = config.HOPSWORKS_PROJECT_NAME
except Exception:
    API_KEY = os.environ.get("HOPSWORKS_API_KEY")
    PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME", "id2223_lab1_G22")

CACHE_FILE = "src/city_coords.json"
MODEL_NAME = "police_crime_model"

# Feature Group versions (keep stable)
EVENTS_FG_NAME = "police_events"
EVENTS_FG_VERSION = 1

PRED_FG_NAME = "police_predictions"
PRED_FG_VERSION = 4   # you already created v4 successfully, keep it

if not API_KEY:
    raise RuntimeError("Missing HOPSWORKS_API_KEY")


# -----------------------------
# Weather helper (parallel)
# -----------------------------
def get_tomorrow_weather(city_list):
    """
    Fetch tomorrow precipitation for each city using Open-Meteo.
    Uses cached city coordinates from CACHE_FILE.
    Parallel requests to speed up.
    """
    if not os.path.exists(CACHE_FILE):
        print("⚠️ No coord cache found. Using 0.0 precipitation.")
        return {c: 0.0 for c in city_list}

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        coords_map = json.load(f)

    def fetch_city(city):
        info = coords_map.get(city)
        if not info:
            return city, 0.0

        lat, lon = info["lat"], info["lon"]
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=precipitation_sum"
            "&timezone=auto"
            "&forecast_days=2"
        )
        try:
            r = requests.get(url, timeout=10).json()
            rain = float(r["daily"]["precipitation_sum"][1])  # index 1 = tomorrow
            return city, rain
        except Exception:
            return city, 0.0

    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(fetch_city, c) for c in city_list]
        for fu in concurrent.futures.as_completed(futures):
            c, rain = fu.result()
            out[c] = rain

    return out


def save_confusion_matrix_png(cm, labels, out_path: str, title: str):
    """
    Save a confusion matrix figure as PNG.
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick = range(len(labels))
    plt.xticks(tick, labels, rotation=90)
    plt.yticks(tick, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def get_best_model(mr):
    """
    Get best model automatically.
    1) Try max balanced_accuracy
    2) Fallback: latest model by created_timestamp
    """
    try:
        return mr.get_best_model(name=MODEL_NAME, metric="balanced_accuracy", direction="max")
    except Exception as e:
        print(f"⚠️ get_best_model(balanced_accuracy) failed: {e}")
        print("➡️ Fallback to latest model by created_timestamp")
        return mr.get_best_model(name=MODEL_NAME, metric="created_timestamp", direction="max")


# -----------------------------
# Main
# -----------------------------
def inference_and_monitor():
    """
    1) Batch inference for tomorrow for all cities x 24 hours and store predictions
    2) Monitoring: evaluate on latest real events (true label exists) and save CM + table
    """
    print("✅ RUNNING inference_and_monitor (auto-best-model)")

    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # ---- Load best model
    print("📥 Fetching best model from registry...")
    model_obj = get_best_model(mr)
    model_version = getattr(model_obj, "version", "unknown")
    print(f"✅ Using model version: {model_version}")

    model_dir = model_obj.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))  # sklearn Pipeline

    # ---- Read latest events
    fg_events = fs.get_feature_group(EVENTS_FG_NAME, version=EVENTS_FG_VERSION)
    df_events = fg_events.read()

    cities = sorted(df_events["city"].dropna().unique().tolist())

    tomorrow = dt.date.today() + dt.timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    day_name = tomorrow.strftime("%A")

    # ---- Forecast weather
    print(f"🌦️ Fetching tomorrow precipitation for {len(cities)} cities...")
    weather_map = get_tomorrow_weather(cities)

    # ---- Build batch dataframe for tomorrow
    rows = []
    for city in cities:
        rain = float(weather_map.get(city, 0.0))
        for hour in range(24):
            rows.append({
                "date": pd.to_datetime(tomorrow_str),
                "city": city,
                "day_of_week": day_name,
                "hour": int(hour),
                "precipitation": rain,
            })

    df_batch = pd.DataFrame(rows)

    # Pipeline expects raw columns (strings + numeric)
    X_pred = df_batch[["city", "day_of_week", "hour", "precipitation"]]
    print(f"🔮 Predicting for {len(cities)} cities x 24 hours...")
    df_batch["predicted_type"] = model.predict(X_pred).astype(str)

    # ---- Save distribution chart (tomorrow)
    dist = df_batch["predicted_type"].value_counts()
    top_k = 12
    dist_top = dist.head(top_k)

    plt.figure(figsize=(12, 5))
    plt.bar(dist_top.index.astype(str), dist_top.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Predicted Event Type Distribution (Tomorrow) - {tomorrow_str}")
    plt.ylabel("Count (city x hour)")
    plt.tight_layout()
    plt.savefig("predicted_type_distribution.png")
    plt.close()
    print("📊 Saved: predicted_type_distribution.png")

    # ---- Store predictions in Feature Group
    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["date", "city", "hour"],
        event_time="date",
        description="Daily predicted police event types (city/day/hour + precipitation)"
    )

    df_to_insert = df_batch[["date", "city", "hour", "day_of_week", "precipitation", "predicted_type"]].copy()

    print("💾 Inserting predictions to Hopsworks...")
    pred_fg.insert(df_to_insert, write_options={"wait_for_job": False})
    print("✅ Predictions inserted.")

    # ---- Simple forecast chart: top 10 rainiest cities
    top = (
        df_to_insert.groupby("city", as_index=False)
        .agg(precipitation=("precipitation", "mean"))
        .sort_values("precipitation", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(12, 5))
    plt.bar(top["city"], top["precipitation"])
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top 10 Rainiest Cities (Forecast) - {tomorrow_str}")
    plt.ylabel("Precipitation (mm)")
    plt.tight_layout()
    plt.savefig("crime_forecast.png")
    plt.close()
    print("📊 Saved: crime_forecast.png")

    # -----------------------------
    # Monitoring on latest REAL events
    # -----------------------------
    N = 200
    df_recent = df_events.sort_values("datetime").tail(N).copy()
    df_recent = df_recent.dropna(subset=["city", "day_of_week", "hour", "precipitation", "type"])

    X_recent = df_recent[["city", "day_of_week", "hour", "precipitation"]]
    y_true = df_recent["type"].astype(str)
    y_hat = model.predict(X_recent).astype(str)

    # Save recent table (last 30 rows)
    out_table = pd.DataFrame({
        "datetime": df_recent["datetime"].astype(str).values,
        "city": df_recent["city"].values,
        "hour": df_recent["hour"].values,
        "day_of_week": df_recent["day_of_week"].values,
        "precipitation": df_recent["precipitation"].values,
        "true_type": y_true.values,
        "pred_type": y_hat,
    }).tail(30)

    out_table.to_csv("monitor_recent.csv", index=False, encoding="utf-8")
    print("🧾 Saved: monitor_recent.csv")

    labels = sorted(list(set(y_true.unique()) | set(pd.Series(y_hat).unique())))
    cm = confusion_matrix(y_true, y_hat, labels=labels)
    save_confusion_matrix_png(
        cm, labels, "monitor_confusion_matrix.png",
        title=f"Monitoring Confusion Matrix (Last {len(df_recent)} real events)"
    )
    print("🧠 Saved: monitor_confusion_matrix.png")

    print("🎉 Inference + Monitoring finished successfully!")


if __name__ == "__main__":
    inference_and_monitor()
