import os
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    location = "Global Station"
    if request.method == 'POST':
        data = request.get_json()
        if data and "location" in data:
            location = data["location"]
    
    # Simple prediction math
    # High rainfall (0.9) + High river level (0.8)
    risk_score = 0.85 
    
    return jsonify({
        "location": location,
        "forecast": {
            "flood": {
                "probability": "85%",
                "alert_level": "🔴 HIGH ALERT",
                "advisory": "Immediate evacuation advised for " + location
            },
            "cyclone": {
                "probability": "12%",
                "alert_level": "🟢 NORMAL",
                "advisory": "Conditions stable"
            }
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
  
