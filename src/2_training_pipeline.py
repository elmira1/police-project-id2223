import os
import json
import hopsworks
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

from hsml.schema import Schema
from hsml.model_schema import ModelSchema


# -----------------------------
# Config
# -----------------------------
# Try to read Hopsworks credentials from src/config.py first.
# If config.py is not available, fall back to environment variables.
try:
    import config
    API_KEY = config.HOPSWORKS_API_KEY
    PROJECT_NAME = config.HOPSWORKS_PROJECT_NAME
except Exception:
    API_KEY = os.environ.get("HOPSWORKS_API_KEY")
    PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME", "id2223_lab1_G22")

# Stop early if we don't have an API key (otherwise login will fail later).
if not API_KEY:
    raise RuntimeError("Missing HOPSWORKS_API_KEY")


# -----------------------------
# Helpers
# -----------------------------
def save_confusion_matrix_png(y_true, y_pred, labels, out_path: str, title: str):
    """
    Create and save a confusion matrix figure as a PNG.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    # Write the count inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def train_model(min_count=10):
    """
    Training pipeline:
    1) Read data from Hopsworks Feature Group
    2) Filter rare classes (min_count) to make training stable
    3) Train a Pipeline: OneHotEncode(city/day) + RandomForest
    4) Evaluate and save artifacts (model + confusion matrix + meta.json)
    5) Upload to Hopsworks Model Registry
    """
    # Connect to Hopsworks
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("📥 Loading Feature Group...")
    fg = fs.get_feature_group(name="police_events", version=1)

    # Read data from the feature group and keep only required columns
    df = fg.read()
    needed = ["city", "day_of_week", "hour", "precipitation", "type"]
    df = df[needed].dropna()

    # Remove very small classes to avoid unstable training / split problems
    vc = df["type"].value_counts()
    keep_labels = vc[vc >= min_count].index
    df = df[df["type"].isin(keep_labels)].copy()
    print(f"✅ Kept {df['type'].nunique()} classes (min_count={min_count})")

    # Define features and label
    X = df[["city", "day_of_week", "hour", "precipitation"]]
    y = df["type"].astype(str)

    # Train/test split (stratify keeps label distribution similar in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # Preprocess: OneHotEncode categorical features and pass through numeric ones
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city", "day_of_week"]),
            ("num", "passthrough", ["hour", "precipitation"]),
        ],
        remainder="drop",
    )

    # RandomForest classifier (balanced_subsample helps with class imbalance)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    # Full sklearn Pipeline: preprocessing + model
    model = Pipeline(steps=[("preprocess", pre), ("model", clf)])

    print("🧠 Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ Balanced Accuracy: {bacc:.3f}")
    print(classification_report(y_test, y_pred))

    # Save artifacts locally
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save the full pipeline (this is enough for inference)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))

    # Save confusion matrix image
    labels_sorted = sorted(list(keep_labels))
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    save_confusion_matrix_png(
        y_test, y_pred, labels_sorted, cm_path,
        title="Police Type Confusion Matrix (Test Set)"
    )

    # Save small metadata
    meta = {
        "min_count": int(min_count),
        "classes": labels_sorted,
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
    }
    with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Upload to Hopsworks Model Registry
    print("📤 Uploading model to Hopsworks Model Registry...")

    input_schema = Schema(X_train)
    # Hopsworks expects a dataframe-like output schema
    output_schema = Schema(pd.DataFrame(y_train))
    model_schema = ModelSchema(input_schema, output_schema)

    hs_model = mr.python.create_model(
        name="police_crime_model",
        metrics={"accuracy": acc, "balanced_accuracy": bacc},
        model_schema=model_schema,
        description="Predict police event type from city/day/hour + precipitation. Includes confusion_matrix.png."
    )
    hs_model.save(model_dir)

    print("🎉 Training pipeline finished successfully!")


if __name__ == "__main__":
    train_model(min_count=10)
