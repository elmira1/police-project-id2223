import os
import hopsworks
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Login Settings ---
api_key = None
project_name = "id2223_lab1_G22"

try:
    import config
    api_key = config.HOPSWORKS_API_KEY
    project_name = config.HOPSWORKS_PROJECT_NAME
except ImportError:
    try:
        from src import config
        api_key = config.HOPSWORKS_API_KEY
        project_name = config.HOPSWORKS_PROJECT_NAME
    except ImportError:
        api_key = os.environ.get('HOPSWORKS_API_KEY')

if api_key is None:
    raise Exception("API Key not found!")

def train_model():
    # 1. Connect to Hopsworks
    project = hopsworks.login(api_key_value=api_key, project=project_name)
    fs = project.get_feature_store()
    
    # 2. Get data
    print("📥 Downloading data...")
    try:
        police_fg = fs.get_feature_group(name="police_events", version=1)
        query = police_fg.select(["city", "hour", "day_of_week", "type"])
        df = query.read()
    except:
        print("❌ Error: Feature group not found. Run pipeline 1 first.")
        return

    print(f"Rows loaded: {len(df)}")
    
    # 3. Convert text to numbers (Encoding)
    le_city = LabelEncoder()
    le_day = LabelEncoder()
    le_type = LabelEncoder()
    
    df['city_encoded'] = le_city.fit_transform(df['city'])
    df['day_encoded'] = le_day.fit_transform(df['day_of_week'])
    df['type_encoded'] = le_type.fit_transform(df['type'])
    
    X = df[['city_encoded', 'hour', 'day_encoded']]
    y = df['type_encoded']
    
    # 4. Split data (Train vs Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train the Brain (Model)
    print("🧠 Training model...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # 6. Check score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    
    # 7. Save files locally
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/police_model.pkl")
    joblib.dump(le_city, "models/city_encoder.pkl")
    joblib.dump(le_day, "models/day_encoder.pkl")
    joblib.dump(le_type, "models/type_encoder.pkl")
    
    # 8. Upload model to cloud
    mr = project.get_model_registry()
    
    python_model = mr.python.create_model(
        name="police_crime_model", 
        metrics={"accuracy": accuracy},
        description="Predicts crime type."
    )
    
    print("📤 Uploading model...")
    python_model.save("models")
    print("🎉 Success! Model saved.")

if __name__ == "__main__":
    train_model()