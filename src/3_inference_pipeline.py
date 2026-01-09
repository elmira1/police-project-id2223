# src/3_inference_pipeline.py
import os
import json
import datetime as dt
import concurrent.futures

import pandas as pd
import requests
import hopsworks
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


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

if not API_KEY:
    raise RuntimeError("Missing HOPSWORKS_API_KEY")

CACHE_FILE = "src/city_coords.json"

MODEL_NAME = "police_crime_model"

EVENTS_FG_NAME = "police_events"
EVENTS_FG_VERSION = 2  # must match your daily FG version

# ✅ NEW schema for grouped predictions -> use NEW version
PRED_FG_NAME = "police_predictions"
PRED_FG_VERSION = 5  # <-- changed from 4 to 5


# -----------------------------
# Weather helper (parallel)
# -----------------------------
def get_tomorrow_weather(city_list):
    if not os.path.exists(CACHE_FILE):
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
            rain = float(r["daily"]["precipitation_sum"][1])  # tomorrow
            return city, rain
        except Exception:
            return city, 0.0

    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(fetch_city, c) for c in city_list]
        for fu in concurrent.futures.as_completed(futures):
            c, rain = fu.result()
            out[c] = rain
    return out


def save_confusion_matrix_png(cm, labels, out_path: str, title: str):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick = range(len(labels))
    plt.xticks(tick, labels, rotation=90)
    plt.yticks(tick, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def get_best_model(mr):
    try:
        return mr.get_best_model(MODEL_NAME, "balanced_accuracy", "max")
    except Exception:
        return mr.get_best_model(MODEL_NAME, "accuracy", "max")


def predict_group(model_bundle, X: pd.DataFrame) -> pd.Series:
    pipe = model_bundle["pipeline"]
    le = model_bundle["label_encoder"]
    pred_int = pipe.predict(X)
    pred = le.inverse_transform(pred_int.astype(int))
    return pd.Series(pred, index=X.index, dtype="string")


def inference_and_monitor():
    print("✅ RUNNING inference_and_monitor (grouped labels)")

    print("🔌 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    # ---- Load best model ----
    print("📥 Fetching best model from registry...")
    best = get_best_model(mr)
    print(f"✅ Using model version: {best.version}")
    model_dir = best.download()

    model_bundle = joblib.load(os.path.join(model_dir, "model.pkl"))
    feature_cols = model_bundle["feature_cols"]

    # ---- Read events ----
    print("📥 Reading events from Feature Store...")
    fg_events = fs.get_feature_group(EVENTS_FG_NAME, version=EVENTS_FG_VERSION)

    query = fg_events.select(
        ["datetime", "city", "hour", "day_of_week", "precipitation", "type_group"]
    )
    df_events = query.read().dropna(subset=["city", "day_of_week", "hour", "precipitation"])

    cities = sorted(df_events["city"].dropna().unique().tolist())

    # -----------------------------
    # 1) Forecast for tomorrow (batch)
    # -----------------------------
    tomorrow = dt.date.today() + dt.timedelta(days=1)
    tomorrow_str = tomorrow.strftime("%Y-%m-%d")
    day_name = tomorrow.strftime("%A")

    print(f"🌦️ Fetching tomorrow precipitation for {len(cities)} cities...")
    weather_map = get_tomorrow_weather(cities)

    rows = []
    for city in cities:
        rain = float(weather_map.get(city, 0.0))
        for hour in range(24):
            rows.append(
                {
                    "date": pd.to_datetime(tomorrow_str),
                    "city": city,
                    "day_of_week": day_name,
                    "hour": int(hour),
                    "precipitation": rain,
                }
            )
    df_batch = pd.DataFrame(rows)

    print(f"🔮 Predicting for {len(cities)} cities x 24 hours = {len(df_batch)} rows...")
    X_pred = df_batch[feature_cols].copy()
    df_batch["predicted_type_group"] = predict_group(model_bundle, X_pred)

    # Chart 1: rainiest cities
    top = (
        df_batch.groupby("city", as_index=False)
        .agg(precipitation=("precipitation", "mean"))
        .sort_values("precipitation", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(12, 5))
    plt.bar(top["city"].astype(str), top["precipitation"].values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Top 10 Rainiest Cities (Forecast) - {tomorrow_str}")
    plt.ylabel("Precipitation (mm)")
    plt.tight_layout()
    plt.savefig("crime_forecast.png")
    plt.close()
    print("📊 Saved chart: crime_forecast.png")

    # Chart 2: distribution of predicted groups
    dist = df_batch["predicted_type_group"].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(dist.index.astype(str), dist.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Predicted Event Group Distribution (Tomorrow) - {tomorrow_str}")
    plt.ylabel("Count (city x hour)")
    plt.tight_layout()
    plt.savefig("predicted_type_distribution.png")
    plt.close()
    print("📊 Saved chart: predicted_type_distribution.png")

    # ---- Save predictions to FG (NEW VERSION) ----
    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["date", "city", "hour"],
        event_time="date",
        description="Daily predicted police event type groups (7 classes) per city/hour + precipitation",
    )

    df_to_insert = df_batch[
        ["date", "city", "hour", "day_of_week", "precipitation", "predicted_type_group"]
    ].copy()

    print("💾 Inserting predictions to Hopsworks...")
    pred_fg.insert(df_to_insert, write_options={"wait_for_job": False})
    print("✅ Predictions inserted.")

    # -----------------------------
    # 2) Monitoring on latest real events
    # -----------------------------
    N = 200
    df_recent = df_events.sort_values("datetime").tail(N).dropna(
        subset=["city", "day_of_week", "hour", "precipitation", "type_group"]
    )

    X_recent = df_recent[feature_cols].copy()
    y_true = df_recent["type_group"].astype(str)
    y_hat = predict_group(model_bundle, X_recent).astype(str)

    out_table = pd.DataFrame(
        {
            "datetime": df_recent["datetime"].astype(str).values,
            "city": df_recent["city"].values,
            "hour": df_recent["hour"].values,
            "day_of_week": df_recent["day_of_week"].values,
            "precipitation": df_recent["precipitation"].values,
            "true_group": y_true.values,
            "pred_group": y_hat.values,
        }
    ).tail(30)

    out_table.to_csv("monitor_recent.csv", index=False, encoding="utf-8")
    print("🧾 Saved: monitor_recent.csv")

    labels = sorted(list(set(y_true.unique()) | set(y_hat.unique())))
    cm = confusion_matrix(y_true, y_hat, labels=labels)

    save_confusion_matrix_png(
        cm,
        labels,
        "monitor_confusion_matrix.png",
        title=f"Monitoring Confusion Matrix (Last {len(df_recent)} real events) - 7 groups",
    )
    print("🧠 Saved: monitor_confusion_matrix.png")

    print("🎉 Inference + Monitoring finished successfully!")


if __name__ == "__main__":
    inference_and_monitor()
