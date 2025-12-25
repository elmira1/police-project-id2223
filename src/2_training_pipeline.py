import os
import json
import joblib
import warnings
from dataclasses import dataclass
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import hopsworks

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.utils.class_weight import compute_sample_weight

# XGBoost
from xgboost import XGBClassifier

# Hopsworks model schema (usually available when hopsworks is installed)
from hsml.schema import Schema
from hsml.model_schema import ModelSchema


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


# -----------------------------
# Helpers
# -----------------------------
def save_confusion_matrix_png(y_true, y_pred, labels: List[str], out_path: str, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = range(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def safe_drop_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing, errors="ignore")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # detect columns
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    cat_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    num_pipe = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )
    return pre


@dataclass
class PoliceXGBModel:
    """
    A small wrapper that:
    - preprocesses with ColumnTransformer (OneHot on categoricals)
    - label-encodes y to 0..K-1 for XGBoost
    - returns predictions as ORIGINAL string labels
    """
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    model: XGBClassifier
    classes_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        self.preprocessor.fit(X)
        X_tr = self.preprocessor.transform(X)

        self.label_encoder.fit(y.astype(str))
        y_enc = self.label_encoder.transform(y.astype(str))

        # XGBoost wants numeric labels for multiclass
        self.model.fit(X_tr, y_enc, sample_weight=sample_weight)
        self.classes_ = list(self.label_encoder.classes_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_tr = self.preprocessor.transform(X)
        pred_enc = self.model.predict(X_tr)
        pred_enc = np.asarray(pred_enc).astype(int)
        return self.label_encoder.inverse_transform(pred_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_tr = self.preprocessor.transform(X)
        return self.model.predict_proba(X_tr)


def get_or_create_feature_view(fs, fg, label_col: str, fv_name: str = "police_events_fv", fv_version: int = 1):
    """
    Creates a Feature View in Hopsworks so it appears in UI.
    Feature View doesn't need "updating" - it always points to latest FG data.
    """
    try:
        fv = fs.get_feature_view(name=fv_name, version=fv_version)
        print(f"✅ Feature View exists: {fv_name} v{fv_version}")
        return fv
    except Exception:
        print(f"🧩 Creating Feature View: {fv_name} v{fv_version} ...")
        query = fg.select_all()
        fv = fs.create_feature_view(
            name=fv_name,
            version=fv_version,
            query=query,
            labels=[label_col],
            description="Feature View for police events (created by training pipeline).",
        )
        print(f"✅ Feature View created: {fv_name} v{fv_version}")
        return fv


# -----------------------------
# Main training pipeline
# -----------------------------
def train_model(
    min_count: int = 10,
    test_size: float = 0.2,
    model_name: str = "police_crime_model",
    fv_name: str = "police_events_fv",
    fv_version: int = 1,
):
    """
    1) Read data from Hopsworks Feature Group
    2) Create/ensure Feature View exists
    3) Filter rare classes
    4) Preprocess (OHE) + XGBoost multiclass
    5) Evaluate + save artifacts
    6) Upload to Hopsworks Model Registry
    """

    # Reduce noise from version warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

    print("🔌 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    mr = project.get_model_registry()

    print("📥 Loading Feature Group...")
    fg = fs.get_feature_group(name="police_events", version=1)

    # Create / ensure feature view
    get_or_create_feature_view(fs, fg, label_col="type", fv_name=fv_name, fv_version=fv_version)

    df = fg.read()

    # Basic sanity
    if "type" not in df.columns:
        raise RuntimeError(f"Label column 'type' not found. Available columns: {list(df.columns)}")

    # Keep a minimal set if exists, otherwise fallback to auto selection
    # (This avoids pulling text columns that are not useful for ML)
    preferred_cols = ["city", "day_of_week", "hour", "precipitation", "latitude", "longitude", "month", "type"]
    use_cols = [c for c in preferred_cols if c in df.columns]
    if len(use_cols) >= 3 and "type" in use_cols:
        df = df[use_cols].copy()
    else:
        # fallback: remove common non-feature columns if present
        df = safe_drop_columns(df, ["summary", "description", "name", "url", "link", "datetime", "date", "time"])
        # keep label + others
        df = df.copy()

    df = df.dropna(subset=["type"]).copy()

    # Remove very small classes
    vc = df["type"].astype(str).value_counts()
    keep_labels = vc[vc >= min_count].index.tolist()
    df = df[df["type"].astype(str).isin(keep_labels)].copy()

    print(f"✅ Kept {df['type'].nunique()} classes (min_count={min_count})")
    print(f"✅ Rows after filtering: {len(df)}")

    # Split
    y = df["type"].astype(str)
    X = df.drop(columns=["type"])

    # Extra safe: drop an id column if it exists (often harms learning)
    X = safe_drop_columns(X, ["id", "event_id", "case_id"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"✅ Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # Preprocessor + Model
    pre = build_preprocessor(X_train)

    # Sample weights to reduce imbalance
    # NOTE: compute_sample_weight accepts string labels fine.
    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

    # XGBoost parameters (safe defaults)
    n_classes = y_train.nunique()
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    wrapper = PoliceXGBModel(
        preprocessor=pre,
        label_encoder=LabelEncoder(),
        model=xgb,
    )

    print("🧠 Training model (XGBoost)...")
    wrapper.fit(X_train, y_train, sample_weight=sample_w)

    # Evaluate
    y_pred = wrapper.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.3f}")
    print(f"✅ Balanced Accuracy: {bacc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save artifacts
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save one pickle with everything needed for inference
    joblib.dump(wrapper, os.path.join(model_dir, "model.pkl"))

    # Confusion matrix
    labels_sorted = sorted(keep_labels)
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    save_confusion_matrix_png(
        y_test, y_pred, labels_sorted, cm_path,
        title="Police Type Confusion Matrix (Test Set)"
    )

    meta = {
        "model_type": "xgboost",
        "min_count": int(min_count),
        "rows_total_after_filter": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "classes": labels_sorted,
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "feature_columns_used": list(X.columns),
        "project": PROJECT_NAME,
    }
    with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Upload to Hopsworks Model Registry
    print("📤 Uploading model to Hopsworks Model Registry...")

    input_schema = Schema(X_train)
    output_schema = Schema(pd.DataFrame(y_train, columns=["type"]))
    model_schema = ModelSchema(input_schema, output_schema)

    hs_model = mr.python.create_model(
        name=model_name,
        metrics={"accuracy": acc, "balanced_accuracy": bacc},
        model_schema=model_schema,
        description="XGBoost model to predict police event type from tabular features. Includes confusion_matrix.png and meta.json.",
    )
    hs_model.save(model_dir)

    print("🎉 Training pipeline finished successfully!")


if __name__ == "__main__":
    # You can change these if you want:
    # - min_count: higher => fewer classes, more stable
    # - test_size: e.g. 0.2
    train_model(min_count=10, test_size=0.2)
