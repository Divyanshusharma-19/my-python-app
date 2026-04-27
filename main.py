import os
import threading
import warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Suppress warnings for clean deployment logs
warnings.filterwarnings("ignore")

# --- 1. CORE AI ENSEMBLE CLASS ---
class DisasterGuardEnsemble:
    """
    Ensemble model combining RF + XGBoost predictions.
    """
    ALERT_THRESHOLD = 0.65
    WARNING_THRESHOLD = 0.40
    FEATURES = [
        "rainfall_mm", "river_level_m", "soil_moisture",
        "temperature_c", "humidity_pct", "wind_speed_kmh",
        "elevation_m", "pressure_hpa", "prev_rainfall", "cyclone_dist_km"
    ]

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.is_trained = False

    def train(self):
        """Standard training logic for RF / XGBoost"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        print("⏳ Generating synthetic data and training models...")
        # Simulating data for initialization
        N = 2000
        data = pd.DataFrame(np.random.rand(N, len(self.FEATURES)), columns=self.FEATURES)
        
        # Simulated Labels based on your project logic
        f_label = (data["rainfall_mm"] > 0.7) & (data["river_level_m"] > 0.6)
        c_label = (data["wind_speed_kmh"] > 0.8) & (data["pressure_hpa"] < 0.3)
        h_label = (data["temperature_c"] > 0.85) & (data["humidity_pct"] < 0.2)
        l_label = (data["rainfall_mm"] > 0.75) & (data["elevation_m"] > 0.7)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(data)

        self.models["flood"] = RandomForestClassifier(n_estimators=50).fit(X_scaled, f_label)
        self.models["cyclone"] = RandomForestClassifier(n_estimators=50).fit(X_scaled, c_label)
        self.models["heatwave"] = RandomForestClassifier(n_estimators=50).fit(X_scaled, h_label)
        self.models["landslide"] = RandomForestClassifier(n_estimators=50).fit(X_scaled, l_label)
        
        self.is_trained = True
        print("✅ DisasterGuard AI Training Complete.")

    def predict(self, sensor_data):
        if not self.is_trained: return None
        
        # Build feature vector
        X = np.array([[sensor_data.get(f, 0.5) for f in self.FEATURES]])
        X_scaled = self.scaler.transform(X)
        
        results = {}
        for d_name, model in self.models.items():
            prob = float(model.predict_proba(X_scaled)[0][1])
            
            if prob >= self.ALERT_THRESHOLD:
                level, advisory = "🔴 HIGH ALERT", f"IMMEDIATE {d_name.upper()} RISK"
            elif prob >= self.WARNING_THRESHOLD:
                level, advisory = "🟡 WARNING", f"MONITORING {d_name.upper()}"
            else:
                level, advisory = "🟢 NORMAL", "STABLE"

            results[d_name] = {
                "probability": f"{round(prob * 100, 1)}%",
                "alert_level": level,
                "advisory": advisory
            }
        return results

# --- 2. APP INITIALIZATION ---

app = Flask(__name__)
dg_ai = DisasterGuardEnsemble()

# Global training before start
dg_ai.train()

@app.route('/')
def home():
    return {
        "status": "Online",
        "system": "DisasterGuard AI v1.0",
        "trained": dg_ai.is_trained
    }

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
    else:
        # Default high-risk sample for browser testing
        data = {
            "location": "Sample Station",
            "rainfall_mm": 0.9, "river_level_m": 0.85, "soil_moisture": 0.8,
            "temperature_c": 0.4, "humidity_pct": 0.6, "wind_speed_kmh": 0.2,
            "elevation_m": 0.2, "pressure_hpa": 0.5, "prev_rainfall": 0.7,
            "cyclone_dist_km": 0.9
        }

    if not data: return jsonify({"error": "No data"}), 400
    
    location = data.pop("location", "Unknown")
    res = dg_ai.predict(data)
    return jsonify({"location": location, "forecast": res})

# --- 3. THE ONLY START BLOCK YOU NEED ---

if __name__ == "__main__":
    # Binding to dynamic port for Railway
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
@app.route('/')
def home():
    # This is the "face" of your app
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_api():
    # This is the "brain" of your app (the JSON data)
    # ... keep your existing prediction logic here ...
