
   import os
import warnings
from flask import Flask, request, jsonify, render_template

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- 1. CORE PREDICTION LOGIC ---
class DisasterGuardAI:
    def __init__(self):
        self.is_trained = True 

    def calculate_risk(self, sensors):
        # Math-based weighted ensemble logic
        rain = float(sensors.get("rainfall_mm", 0.5))
        river = float(sensors.get("river_level_m", 0.5))
        wind = float(sensors.get("wind_speed_kmh", 0.5))
        
        # Risk formulas (0.0 to 1.0)
        flood_score = (rain * 0.6) + (river * 0.4)
        cyclone_score = (wind * 0.7) + (rain * 0.3)
        
        forecast = {}
        for name, score in [("flood", flood_score), ("cyclone", cyclone_score)]:
            if score >= 0.65:
                status, msg = "🔴 HIGH ALERT", "Immediate Evacuation Advised"
            elif score >= 0.40:
                status, msg = "🟡 WARNING", "Monitor Conditions Closely"
            else:
                status, msg = "🟢 NORMAL", "Conditions Stable"
            
            forecast[name] = {
                "probability": f"{round(score * 100, 1)}%",
                "alert_level": status,
                "advisory": msg
            }
        return forecast

ai_engine = DisasterGuardAI()

# --- 2. WEB ROUTES ---

@app.route('/')
def home():
    # Serves the interface from templates/index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Default location and data
    location = "Global Station"
    test_sensors = {"rainfall_mm": 0.82, "river_level_m": 0.75, "wind_speed_kmh": 0.2}

    if request.method == 'POST':
        user_input = request.get_json()
        if user_input and "location" in user_input:
            location = user_input["location"]
    
    # Run the prediction
    results = ai_engine.calculate_risk(test_sensors)
    
    return jsonify({
        "location": location,
        "forecast": results
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
