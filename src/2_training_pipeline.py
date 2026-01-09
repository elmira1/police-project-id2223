# src/2_training_pipeline.py
import os
import json
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import hopsworks

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


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


FG_NAME = "police_events"
FG_VERSION = 2  # <-- important (we added type_group)

MODEL_NAME = "police_crime_model"

FEATURE_COLS = ["city", "day_of_week", "hour", "precipitation"]
LABEL_COL = "type_group"


def build_pipeline() -> Pipeline:
    cat_cols = ["city", "day_of_week"]
    num_cols = ["hour", "precipitation"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
    )

    pipe = Pipeline([("preprocess", pre), ("model", clf)])
    return pipe


def compute_sample_weights(y_int: np.ndarray) -> np.ndarray:
    # Inverse-frequency weights (helps imbalance)
    unique, counts = np.unique(y_int, return_counts=True)
    freq = dict(zip(unique, counts))
    w = np.array([1.0 / freq[c] for c in y_int], dtype=float)
    w = w / np.mean(w)
    return w


def train_model(min_count: int = 30, test_size: float = 0.2) -> None:
    print("🔌 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("📥 Loading Feature Group...")
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.read()

    # Basic cleaning
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()

    # Filter rare classes
    vc = df[LABEL_COL].value_counts()
    keep = vc[vc >= min_count].index.tolist()
    df = df[df[LABEL_COL].isin(keep)].copy()

    print(f"✅ Kept {len(keep)} grouped classes (min_count={min_count})")
    print(f"✅ Rows after filtering: {len(df)}")

    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(str).copy()

    # Encode labels
    le = LabelEncoder()
    y_int = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=42, stratify=y_int
    )

    print(f"✅ Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    pipe = build_pipeline()

    sample_w = compute_sample_weights(y_train)

    print("🧠 Training model (XGBoost) on 7 grouped classes...")
    pipe.fit(X_train, y_train, model__sample_weight=sample_w)

    # Evaluate
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ Balanced Accuracy: {bacc:.3f}")
    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

    # Save bundle (NO custom class -> no pickle issues)
    model_bundle = {
        "pipeline": pipe,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "trained_at": datetime.utcnow().isoformat(),
        "min_count": int(min_count),
        "test_size": float(test_size),
        "metrics": {"accuracy": float(acc), "balanced_accuracy": float(bacc)},
        "classes": le.classes_.tolist(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        joblib.dump(model_bundle, model_path)

        meta_path = os.path.join(tmpdir, "metrics.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(model_bundle["metrics"], f, indent=2)

        print("📤 Uploading model to Hopsworks Model Registry...")
        model = mr.python.create_model(
            name=MODEL_NAME,
            metrics=model_bundle["metrics"],
            description="XGBoost classifier on grouped police event types (7 classes) using city/day/hour/precipitation.",
        )
        model.save(tmpdir)

    print("🎉 Training pipeline finished successfully!")


if __name__ == "__main__":
    # You can tweak these safely
    train_model(min_count=30, test_size=0.2)
