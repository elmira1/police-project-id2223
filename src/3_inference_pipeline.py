# src/3_inference_pipeline.py
import os
import json
import datetime as dt
import concurrent.futures
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import hopsworks
import joblib

import matplotlib
matplotlib.use("Agg")  # important for GitHub Actions
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None


# ============================================================
# IMPORTANT: Custom class needed for loading pickled model.pkl
# ============================================================
class PoliceXGBModel:
    """
    Your model.pkl was saved with this class in training.
    It MUST exist at inference runtime for joblib.load() to work.
    """

    def predict(self, X: pd.DataFrame):
        # model object inside can be in pipeline/model attributes
        pipe = getattr(self, "pipeline", None) or getattr(self, "model", None)
        if pipe is None:
            raise AttributeError("PoliceXGBModel missing 'pipeline' or 'model' attribute")
        return pipe.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        pipe = getattr(self, "pipeline", None) or getattr(self, "model", None)
        if pipe is None:
            raise AttributeError("PoliceXGBModel missing 'pipeline' or 'model' attribute")
        if not hasattr(pipe, "predict_proba"):
            raise AttributeError("Underlying estimator has no predict_proba()")
        return pipe.predict_proba(X)


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

PRED_FG_NAME = "police_predictions"
PRED_FG_VERSION = 4  # keep stable

# MUST match training pipeline
MIN_COUNT = 10
N_MONITOR = 200


# -----------------------------
# Time helper
# -----------------------------
def stockholm_today_date() -> dt.date:
    if ZoneInfo is None:
        return dt.date.today()
    try:
        return dt.datetime.now(ZoneInfo("Europe/Stockholm")).date()
    except Exception:
        return dt.date.today()


# -----------------------------
# Weather helper (parallel)
# -----------------------------
def get_tomorrow_weather(city_list: List[str]) -> Dict[str, float]:
    if not os.path.exists(CACHE_FILE):
        return {c: 0.0 for c in city_list}

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        coords_map = json.load(f)

    def fetch_city(city: str) -> Tuple[str, float]:
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

    out: Dict[str, float] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(fetch_city, c) for c in city_list]
        for fu in concurrent.futures.as_completed(futures):
            c, rain = fu.result()
            out[c] = float(rain)
    return out


# -----------------------------
# Plot helper
# -----------------------------
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


# -----------------------------
# Model picking helper
# -----------------------------
def get_best_model(mr):
    try:
        return mr.get_best_model(MODEL_NAME, "balanced_accuracy", "max")
    except Exception:
        return mr.get_best_model(MODEL_NAME, "accuracy", "max")


def get_estimator(model_obj):
    """
    If model is wrapper, use its internal estimator.
    Otherwise return itself.
    """
    if hasattr(model_obj, "pipeline"):
        return model_obj.pipeline
    if hasattr(model_obj, "model"):
        return model_obj.model
    return model_obj


def expected_n_features(est) -> Optional[int]:
    # sklearn API
    n = getattr(est, "n_features_in_", None)
    if isinstance(n, int):
        return n

    # xgboost booster
    try:
        booster = est.get_booster()
        return int(booster.num_features())
    except Exception:
        return None


# -----------------------------
# ONE-HOT: build reference columns from training-like data
# -----------------------------
def build_reference_feature_columns(df_events: pd.DataFrame, min_count: int) -> List[str]:
    """
    Rebuild the exact one-hot feature columns set from the same logic as training:
    - filter rare labels by min_count
    - get_dummies on city/day_of_week
    - keep hour/precipitation numeric
    """
    df = df_events.copy()
    df = df.dropna(subset=["city", "day_of_week", "hour", "precipitation", "type"])

    vc = df["type"].value_counts()
    keep = vc[vc >= min_count].index
    df = df[df["type"].isin(keep)].copy()

    X_ref = df[["city", "day_of_week", "hour", "precipitation"]].copy()
    X_ref["hour"] = pd.to_numeric(X_ref["hour"], errors="coerce").fillna(0).astype("int32")
    X_ref["precipitation"] = pd.to_numeric(X_ref["precipitation"], errors="coerce").fillna(0.0).astype("float32")

    X_ref_oh = pd.get_dummies(X_ref, columns=["city", "day_of_week"])
    return list(X_ref_oh.columns)


def make_onehot_matrix(X_raw: pd.DataFrame, ref_cols: List[str], exp_n: Optional[int]) -> pd.DataFrame:
    X = X_raw.copy()
    X["hour"] = pd.to_numeric(X["hour"], errors="coerce").fillna(0).astype("int32")
    X["precipitation"] = pd.to_numeric(X["precipitation"], errors="coerce").fillna(0.0).astype("float32")

    X_oh = pd.get_dummies(X, columns=["city", "day_of_week"])
    X_oh = X_oh.reindex(columns=ref_cols, fill_value=0)

    # If still mismatch, force to expected shape (last resort)
    if exp_n is not None and X_oh.shape[1] != exp_n:
        if X_oh.shape[1] > exp_n:
            X_oh = X_oh.iloc[:, :exp_n]
        else:
            # pad extra zero columns
            missing = exp_n - X_oh.shape[1]
            for i in range(missing):
                X_oh[f"__pad_{i}__"] = 0

    # ensure numeric
    for c in X_oh.columns:
        X_oh[c] = pd.to_numeric(X_oh[c], errors="coerce").fillna(0.0)

    return X_oh


def predict_safely(model_obj, X_raw: pd.DataFrame, ref_cols: List[str]) -> pd.Series:
    est = get_estimator(model_obj)
    exp_n = expected_n_features(est)

    # If model expects many features, it's one-hot trained => use one-hot matrix
    if exp_n is not None and exp_n > X_raw.shape[1]:
        X_mat = make_onehot_matrix(X_raw, ref_cols, exp_n)
        return est.predict(X_mat)

    # Otherwise, try raw (in case it's a full sklearn Pipeline)
    return model_obj.predict(X_raw)


# -----------------------------
# Main
# -----------------------------
def inference_and_monitor():
    print("✅ RUNNING inference_and_monitor (auto-best-model)")
    print("🔌 Connecting to Hopsworks...")

    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("📥 Fetching best model from registry...")
    best = get_best_model(mr)
    print(f"✅ Using model version: {best.version}")
    model_dir = best.download()

    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl not found in downloaded model dir: {model_dir}")

    model = joblib.load(model_path)

    fg_events = fs.get_feature_group("police_events", version=1)
    query = fg_events.select(["datetime", "city", "hour", "day_of_week", "precipitation", "type"])

    print("📥 Reading events from Feature Store...")
    df_events = query.read()

    # Build reference one-hot columns (THIS is the key fix!)
    print(f"🧩 Building reference one-hot columns (min_count={MIN_COUNT}) ...")
    ref_cols = build_reference_feature_columns(df_events, min_count=MIN_COUNT)
    print(f"✅ Reference feature columns: {len(ref_cols)}")

    # Cities list
    cities = sorted(df_events["city"].dropna().unique().tolist())
    if not cities:
        raise RuntimeError("No cities found in police_events.")

    # 1) Forecast tomorrow
    today = stockholm_today_date()
    tomorrow = today + dt.timedelta(days=1)
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
    X_pred = df_batch[["city", "day_of_week", "hour", "precipitation"]]
    df_batch["predicted_type"] = predict_safely(model, X_pred, ref_cols).astype(str)

    # Charts
    top = (
        df_batch.groupby("city", as_index=False)
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
    print("📊 Saved chart: crime_forecast.png")

    dist = df_batch["predicted_type"].value_counts().head(12)
    plt.figure(figsize=(12, 5))
    plt.bar(dist.index.astype(str), dist.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Predicted Event Type Distribution (Tomorrow) - {tomorrow_str}")
    plt.ylabel("Count (city x hour)")
    plt.tight_layout()
    plt.savefig("predicted_type_distribution.png")
    plt.close()
    print("📊 Saved chart: predicted_type_distribution.png")

    # Insert predictions to Hopsworks FG
    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["date", "city", "hour"],
        event_time="date",
        description="Daily predicted police event types (city/day/hour + precipitation)",
    )

    df_to_insert = df_batch[["date", "city", "hour", "day_of_week", "precipitation", "predicted_type"]].copy()

    print("💾 Inserting predictions to Hopsworks...")
    pred_fg.insert(df_to_insert, write_options={"wait_for_job": False})
    print("✅ Predictions inserted.")

    # 2) Monitoring on latest real events
    df_recent = df_events.sort_values("datetime").tail(N_MONITOR).dropna(
        subset=["city", "day_of_week", "hour", "precipitation", "type"]
    ).copy()

    X_recent = df_recent[["city", "day_of_week", "hour", "precipitation"]]
    y_true = df_recent["type"].astype(str)
    y_hat = predict_safely(model, X_recent, ref_cols).astype(str)

    out_table = pd.DataFrame(
        {
            "datetime": df_recent["datetime"].astype(str).values,
            "city": df_recent["city"].values,
            "hour": df_recent["hour"].values,
            "day_of_week": df_recent["day_of_week"].values,
            "precipitation": df_recent["precipitation"].values,
            "true_type": y_true.values,
            "pred_type": y_hat,
        }
    ).tail(50)

    out_table.to_csv("monitor_recent.csv", index=False, encoding="utf-8")
    print("🧾 Saved: monitor_recent.csv")

    labels = sorted(list(set(y_true.unique()) | set(pd.Series(y_hat).unique())))
    cm = confusion_matrix(y_true, y_hat, labels=labels)

    save_confusion_matrix_png(
        cm,
        labels,
        "monitor_confusion_matrix.png",
        title=f"Monitoring Confusion Matrix (Last {len(df_recent)} real events)",
    )
    print("🧠 Saved: monitor_confusion_matrix.png")

    print("🎉 Inference + Monitoring finished successfully!")


if __name__ == "__main__":
    inference_and_monitor()
