import os
import hopsworks
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

# --- 1 & 2. Connection Settings (Same as before) ---
try:
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except (ImportError, AttributeError):
    api_key = os.environ.get('HOPSWORKS_API_KEY')
    project_name = os.environ.get('HOPSWORKS_PROJECT_NAME', 'id2223_lab1_G22')

project = hopsworks.login(api_key_value=api_key, project=project_name)
fs = project.get_feature_store()

def train_model():
    # --- 3. Get Feature Group ---
    print("📥 Loading data from Feature Group...")
    police_fg = fs.get_feature_group(name="police_events", version=1)

    # NEW: Added 'precipitation' to the query
    query = police_fg.select(["city", "hour", "day_of_week", "type", "precipitation"])
    
    # --- 4. Create Feature View ---
    print("✨ Creating/Retrieving Feature View (Version 2 for Weather)...")
    feature_view = fs.get_or_create_feature_view(
        name="police_crime_fv",
        version=2, # NEW: Increment version to 2 because we added a new feature
        description="Feature view for police crime classification with weather",
        labels=["type"],
        query=query
    )

    # --- 5 & 6. Read Data and Encoding ---
    print("📥 Reading data into Pandas...")
    df = police_fg.read()
    
    print("⚙️ Encoding text features...")
    le_city, le_day, le_type = LabelEncoder(), LabelEncoder(), LabelEncoder()
    
    df['city_encoded'] = le_city.fit_transform(df['city'])
    df['day_of_week_encoded'] = le_day.fit_transform(df['day_of_week'])
    df['type_encoded'] = le_type.fit_transform(df['type'])
    
    # NEW: Added 'precipitation' to the feature list X
    X = df[['city_encoded', 'hour', 'day_of_week_encoded', 'precipitation']]
    y = df['type_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 7. Train the Model ---
    print("🧠 Training Smart Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # --- 8. Evaluation ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
    
    # --- 9. Save Artifacts ---
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, f"{model_dir}/police_model.pkl")
    joblib.dump(le_city, f"{model_dir}/city_encoder.pkl")
    joblib.dump(le_day, f"{model_dir}/day_encoder.pkl")
    joblib.dump(le_type, f"{model_dir}/type_encoder.pkl")
    
    # --- 10. Model Registry ---
    print("📤 Uploading model to Hopsworks Registry...")
    mr = project.get_model_registry()
    
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)
    
    python_model = mr.python.create_model(
        name="police_crime_model", 
        metrics={"accuracy": accuracy},
        model_schema=model_schema,
        feature_view=feature_view,
        description="Predicts crime type based on city, hour, day, and precipitation."
    )
    
    python_model.save(model_dir)
    print("🎉 Success! Training Pipeline with Weather is finished.")

if __name__ == "__main__":
    train_model()