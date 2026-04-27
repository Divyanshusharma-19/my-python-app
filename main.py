import os
import numpy as np
import pandas as pd
import warnings
from flask import Flask, request, jsonify

# Suppress warnings for clean logs
warnings.filterwarnings("ignore")

# --- 1. CORE AI LOGIC (Condensed for main.py) ---

class DisasterGuardEnsemble:
    """
    The brain of the app. Handles training and real-time prediction.
    """
    ALERT_THRESHOLD = 0.65
    WARNING_THRESHOLD = 0.40

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.is_trained = False

    def train(self, df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        try:
            from xgboost import XGBClassifier
        except ImportError:
            XGBClassifier = RandomForestClassifier # Fallback

        # Preprocessing
        feature_cols = [
            "rainfall_mm", "river_level_m", "soil_moisture", "temperature_c", 
            "humidity_pct", "wind_speed_kmh", "elevation_m", "pressure_hpa", 
            "prev_rainfall", "cyclone_dist_km"
        ]
        X = df[feature_cols].values
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train simplified models for the app
        self.models["flood"] = RandomForestClassifier(n_estimators=100).fit(X_scaled, df["flood_label"])
        self.models["cyclone"] = RandomForestClassifier(n_estimators=100).fit(X_scaled, df["cyclone_label"])
        self.models["heatwave"] = RandomForestClassifier(n_estimators=100).fit(X_scaled, df["heatwave_label"])
        self.models["landslide"] = RandomForestClassifier(n_estimators=100).fit(X_scaled, df["landslide_label"])
        
        self.is_trained = True

    def predict(self, sensor_data):
        feature_order = [
            "rainfall_mm", "river_level_m", "soil_moisture", "temperature_c", 
            "humidity_pct", "wind_speed_kmh", "elevation_m", "pressure_hpa", 
            "prev_rainfall", "cyclone_dist_km"
        ]
        X = np.array([[sensor_data.get(f, 0) for f in feature_order]])
        X_scaled = self.scaler.transform(X)

        results = {}
        for disaster, model in self.models.items():
            prob = float(model.predict_proba(X_scaled)[0][1])
            results[disaster] = {
                "probability": f"{round(prob * 100, 1)}%",
                "status": "🔴 HIGH ALERT" if prob >= self.ALERT_THRESHOLD else 
                          "🟡 WARNING" if prob >= self.WARNING_THRESHOLD else "🟢 NORMAL"
            }
        return results

# --- 2. APP INITIALIZATION ---

app = Flask(__name__)
dg_ai = DisasterGuardEnsemble()

def bootstrap_app():
    """Generates synthetic data and trains the model on startup."""
    print("⏳ Training DisasterGuard AI... please wait.")
    # Simple synthetic data generator for demo purposes
    N = 2000
    data = pd.DataFrame(np.random.rand(N, 10), columns=[
        "rainfall_mm", "river_level_m", "soil_moisture", "temperature_c", 
        "humidity_pct", "wind_speed_kmh", "elevation_m", "pressure_hpa", 
        "prev_rainfall", "cyclone_dist_km"
    ])
    # Create dummy labels
    for label in ["flood_label", "cyclone_label", "heatwave_label", "landslide_label"]:
        data[label] = np.random.randint(0, 2, N)
    
    dg_ai.train(data)
    print("✅ Model trained and ready.")

# Run training before the first request
bootstrap_app()

# --- 3. ROUTES ---

@app.route('/')
def home():
    return {
        "app": "DisasterGuard AI",
        "version": "1.0",
        "status": "Online",
        "endpoints": ["/predict (POST)", "/health (GET)"]
    }

@app.route('/health')
def health():
    return jsonify({"ready": dg_ai.is_trained})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON input:
    {
        "location": "Patna",
        "rainfall_mm": 80,
        ... (rest of features)
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    location = data.pop("location", "Unknown")
    try:
        prediction = dg_ai.predict(data)
        return jsonify({
            "location": location,
            "results": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 4. START SERVER ---

if __name__ == "__main__":
    # Railway/Heroku dynamic port binding
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
