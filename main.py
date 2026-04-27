
       import os
import numpy as np
import pandas as pd
import warnings
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. THE AI BRAIN ---
class DisasterGuardAI:
    def __init__(self):
        self.features = [
            "rainfall_mm", "river_level_m", "soil_moisture", "temperature_c", 
            "humidity_pct", "wind_speed_kmh", "elevation_m", "pressure_hpa", 
            "prev_rainfall", "cyclone_dist_km"
        ]
        self.is_trained = True 

    def predict(self, sensor_data):
        # Simulated logic based on your project requirements
        rain = float(sensor_data.get("rainfall_mm", 0.5))
        river = float(sensor_data.get("river_level_m", 0.5))
        wind = float(sensor_data.get("wind_speed_kmh", 0.5))
        
        # Weighted probability formulas
        flood_prob = (rain * 0.6) + (river * 0.4)
        cyclone_prob = (wind * 0.7) + (rain * 0.3)
        
        results = {}
        for name, prob in [("flood", flood_prob), ("cyclone", cyclone_prob)]:
            if prob >= 0.65:
                level, advisory = "🔴 HIGH ALERT", "Immediate Evacuation Advised"
            elif prob >= 0.40:
                level, advisory = "🟡 WARNING", "Monitor Conditions Closely"
            else:
                level, advisory = "🟢 NORMAL", "Conditions Stable"
            
            results[name] = {
                "probability": f"{round(prob * 100, 1)}%",
                "alert_level": level,
                "advisory": advisory
            }
        return results

ai_engine = DisasterGuardAI()

# --- 2. THE ROUTES ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # FIXED: Capturing the location from the search bar
    user_location = "Global Station"
    
    if request.method == 'POST':
        data = request.get_json()
        if data and "location" in data:
            user_location = data["location"]
    
    # Static high-risk data for demonstration
    test_data = {
        "rainfall_mm": 0.85, "river_level_m": 0.8, "soil_moisture": 0.9,
        "temperature_c": 0.3, "humidity_pct": 0.9, "wind_speed_kmh": 0.1,
        "elevation_m": 0.1, "pressure_hpa": 0.5, "prev_rainfall": 0.7,
        "cyclone_dist_km": 0.9
    }
        
    analysis = ai_engine.predict(test_data)
    
    # FIXED: Sending the location back to the HTML
    return jsonify({
        "location": user_location,
        "forecast": analysis
    })

if __name__ == "__main__":
    # Railway looks for the PORT environment variable
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
