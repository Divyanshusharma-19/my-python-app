import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- THE PREDICTION ENGINE ---
def get_analysis(rain, river):
    prob = (rain * 0.6) + (river * 0.4)
    if prob >= 0.65:
        return {"prob": f"{round(prob*100)}%", "alert": "🔴 HIGH ALERT", "msg": "Evacuate Immediately"}
    return {"prob": f"{round(prob*100)}%", "alert": "🟢 NORMAL", "msg": "Conditions Stable"}

# --- THE ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    location = "Global Station"
    # Default values for any city
    rain_val = 0.25
    river_val = 0.20

    if request.method == 'POST':
        data = request.get_json()
        if data and "location" in data:
            name = data["location"].lower()
            location = data["location"]
            
            # SMART LOGIC: Change data based on what you type
            if "bihar" in name or "patna" in name or "assam" in name:
                rain_val = 0.88
                river_val = 0.82
            elif "mumbai" in name or "coast" in name:
                rain_val = 0.55 # Warning level
                river_val = 0.40

    # RUN THE AI MATH
    flood_result = get_analysis(rain_val, river_val)
    
    return jsonify({
        "location": location,
        "forecast": {
            "flood": {
                "probability": flood_result["prob"],
                "alert_level": flood_result["alert"],
                "advisory": f"{flood_result['msg']} for {location}"
            }
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
