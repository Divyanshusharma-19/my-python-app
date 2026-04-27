import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    location = "Global Station"
    # Default: Everything is Safe (Green)
    f_prob, f_lvl, f_msg = "22%", "🟢 NORMAL", "Conditions Stable"
    c_prob, c_lvl, c_msg = "15%", "🟢 NORMAL", "Low wind activity"

    if request.method == 'POST':
        data = request.get_json()
        if data and "location" in data:
            location = data["location"]
            name = location.lower()
            
            # If user types Punjab, Bihar, or Patna -> High Flood Risk
            if any(x in name for x in ["punjab", "bihar", "patna", "assam"]):
                f_prob, f_lvl, f_msg = "88%", "🔴 HIGH ALERT", "Heavy monsoon flooding predicted"
            
            # If user types Mumbai or Coast -> High Cyclone Risk
            elif any(x in name for x in ["mumbai", "coast", "odisha"]):
                c_prob, c_lvl, c_msg = "79%", "🔴 HIGH ALERT", "High speed winds detected"

    # CRITICAL: These keys (prob, level, msg) MUST match the HTML exactly
    return jsonify({
        "location": location,
        "forecast": [
            {"type": "FLOOD", "prob": f_prob, "level": f_lvl, "msg": f_msg},
            {"type": "CYCLONE", "prob": c_prob, "level": c_lvl, "msg": c_msg}
        ]
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
