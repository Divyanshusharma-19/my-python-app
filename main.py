import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- MULTI-HAZARD PREDICTION ENGINE ---
def get_multi_risk(rain, river, wind, elevation):
    # 1. Flood Math (Rain + River)
    flood_p = (rain * 0.6) + (river * 0.4)
    # 2. Cyclone Math (Wind + Rain)
    cyclone_p = (wind * 0.7) + (rain * 0.3)
    # 3. Landslide Math (Rain + Elevation)
    landslide_p = (rain * 0.5) + (elevation * 0.5) if rain > 0.6 else 0.1

    def classify(prob):
        if prob >= 0.65: return "🔴 HIGH ALERT", "Immediate Evacuation"
        if prob >= 0.40: return "🟡 WARNING", "Stay Alert"
        return "🟢 NORMAL", "Safe"

    f_level, f_msg = classify(flood_p)
    c_level, c_msg = classify(cyclone_p)
    l_level, l_msg = classify(landslide_p)

    return {
        "flood": {"prob": f"{round(flood_p*100)}%", "level": f_level, "msg": f_msg},
        "cyclone": {"prob": f"{round(cyclone_p*100)}%", "level": c_level, "msg": c_msg},
        "landslide": {"prob": f"{round(landslide_p*100)}%", "level": l_level, "msg": l_msg}
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    location = "Global Station"
    # Defaults
    rain, river, wind, elev = 0.2, 0.2, 0.2, 0.1

    if request.method == 'POST':
        data = request.get_json()
        if data and "location" in data:
            location = data["location"]
            name = location.lower()
            
            # BIHAR/ASSAM = Flood Zone
            if any(x in name for x in ["bihar", "patna", "assam"]):
                rain, river = 0.88, 0.85
            # ODISHA/MUMBAI = Cyclone Zone
            elif any(x in name for x in ["mumbai", "odisha", "vizag", "coast"]):
                wind, rain = 0.82, 0.60
            # SHIMLA/WAYANAD = Landslide Zone
            elif any(x in name for x in ["shimla", "wayanad", "himalayas", "hills"]):
                rain, elev = 0.85, 0.90

    analysis = get_multi_risk(rain, river, wind, elev)
    
    return jsonify({
        "location": location,
        "forecast": analysis
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
