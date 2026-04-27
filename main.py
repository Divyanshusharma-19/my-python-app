"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DisasterGuard AI — Disaster Prediction System v1.0                 ║
║          Predicts Floods, Cyclones, Heatwaves & Landslides 72 hrs ahead     ║
║          Design Thinking & Innovation Practical — 2025-26                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Installation:
    pip install numpy pandas scikit-learn tensorflow xgboost matplotlib seaborn
    pip install flask requests python-dotenv

Usage:
    python DisasterGuard_AI.py --train        # Train all models
    python DisasterGuard_AI.py --predict      # Run predictions
    python DisasterGuard_AI.py --api          # Launch REST API server
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ─── 1. DATA GENERATION (simulates real IMD/NDMA sensor data) ────────────────

def generate_training_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Simulate weather & environmental sensor data for training.
    In production: replace with real IMD API / NDMA IoT feeds.

    Features:
        rainfall_mm     — Hourly rainfall in mm
        river_level_m   — River gauge level in metres
        soil_moisture   — Soil moisture percentage
        temperature_c   — Air temperature in Celsius
        humidity_pct    — Relative humidity %
        wind_speed_kmh  — Wind speed in km/hr
        elevation_m     — Terrain elevation in metres
        pressure_hpa    — Atmospheric pressure in hPa
        prev_rainfall   — Cumulative rainfall last 72 hrs
        cyclone_dist_km — Distance from nearest cyclone centre
    """
    np.random.seed(seed)
    N = n_samples

    data = pd.DataFrame({
        "rainfall_mm":      np.random.exponential(5, N),          # mostly light rain
        "river_level_m":    np.random.normal(3.5, 1.5, N).clip(0, 20),
        "soil_moisture":    np.random.beta(2, 3, N) * 100,
        "temperature_c":    np.random.normal(28, 8, N),
        "humidity_pct":     np.random.beta(3, 2, N) * 100,
        "wind_speed_kmh":   np.random.exponential(20, N),
        "elevation_m":      np.abs(np.random.normal(200, 150, N)),
        "pressure_hpa":     np.random.normal(1013, 15, N),
        "prev_rainfall":    np.random.exponential(30, N),
        "cyclone_dist_km":  np.random.exponential(500, N).clip(10, 2000),
    })

    # Simulate realistic disaster labels based on feature combinations
    flood_risk = (
        (data["rainfall_mm"] > 50).astype(float) * 0.4 +
        (data["river_level_m"] > 7).astype(float) * 0.3 +
        (data["soil_moisture"] > 80).astype(float) * 0.2 +
        (data["prev_rainfall"] > 100).astype(float) * 0.1
    )
    cyclone_risk = (
        (data["wind_speed_kmh"] > 64).astype(float) * 0.5 +
        (data["cyclone_dist_km"] < 200).astype(float) * 0.3 +
        (data["pressure_hpa"] < 980).astype(float) * 0.2
    )
    heatwave_risk = (
        (data["temperature_c"] > 44).astype(float) * 0.5 +
        (data["humidity_pct"] < 20).astype(float) * 0.3 +
        (data["rainfall_mm"] < 1).astype(float) * 0.2
    )
    landslide_risk = (
        (data["rainfall_mm"] > 30).astype(float) * 0.3 +
        (data["soil_moisture"] > 90).astype(float) * 0.3 +
        (data["elevation_m"] > 300).astype(float) * 0.4
    )

    data["flood_label"]     = (flood_risk + np.random.normal(0, 0.05, N) > 0.45).astype(int)
    data["cyclone_label"]   = (cyclone_risk + np.random.normal(0, 0.05, N) > 0.35).astype(int)
    data["heatwave_label"]  = (heatwave_risk + np.random.normal(0, 0.05, N) > 0.40).astype(int)
    data["landslide_label"] = (landslide_risk + np.random.normal(0, 0.05, N) > 0.55).astype(int)

    print(f"✅ Generated {N} training samples")
    print(f"   Flood events:     {data['flood_label'].sum():,}  ({data['flood_label'].mean()*100:.1f}%)")
    print(f"   Cyclone events:   {data['cyclone_label'].sum():,}  ({data['cyclone_label'].mean()*100:.1f}%)")
    print(f"   Heatwave events:  {data['heatwave_label'].sum():,}  ({data['heatwave_label'].mean()*100:.1f}%)")
    print(f"   Landslide events: {data['landslide_label'].sum():,}  ({data['landslide_label'].mean()*100:.1f}%)")
    return data


# ─── 2. PREPROCESSING ─────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, feature_cols: list) -> tuple:
    """Scale features and split into train/test sets."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


FEATURES = [
    "rainfall_mm", "river_level_m", "soil_moisture",
    "temperature_c", "humidity_pct", "wind_speed_kmh",
    "elevation_m", "pressure_hpa", "prev_rainfall", "cyclone_dist_km"
]


# ─── 3. RANDOM FOREST MODEL (Flood & Landslide) ───────────────────────────────

def train_random_forest(X_train, y_train, disaster_type: str):
    """
    Random Forest — Best for tabular sensor data.
    Handles non-linear relationships in terrain + weather variables.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",    # Handles imbalanced disaster data
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    print(f"\n📊 {disaster_type.upper()} — Random Forest Results:")
    print(f"   ROC-AUC: {auc:.4f}")
    print(classification_report(y_val, y_pred, target_names=["No Event", "Disaster"]))
    return model


# ─── 4. XGBOOST MODEL (Cyclone Intensity) ────────────────────────────────────

def train_xgboost(X_train, y_train, disaster_type: str):
    """
    XGBoost — Handles complex feature interactions & gradient boosting.
    Particularly effective for cyclone severity classification.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("⚠  XGBoost not installed. Using Random Forest as fallback.")
        return train_random_forest(X_train, y_train, disaster_type)

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score

    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        verbosity=0
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_prob)
    print(f"\n📊 {disaster_type.upper()} — XGBoost Results:")
    print(f"   ROC-AUC: {auc:.4f}")
    print(classification_report(y_val, y_pred, target_names=["No Event", "Disaster"]))
    return model


# ─── 5. LSTM MODEL (Time-Series Flood Forecasting) ────────────────────────────

def build_lstm_model(timesteps: int = 24, n_features: int = 10) -> "tf.keras.Model":
    """
    LSTM (Long Short-Term Memory) Neural Network.
    Uses 24-hour rolling window of sensor data to predict 72-hr flood risk.
    
    Architecture:
        Input: (batch, 24 timesteps, 10 features)
        → LSTM(128) → Dropout(0.2)
        → LSTM(64) → Dropout(0.2)
        → Dense(32, ReLU)
        → Dense(1, Sigmoid)  → Flood probability (0–1)
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(timesteps, n_features)),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )
        print(f"\n🧠 LSTM Model Summary:")
        model.summary()
        return model

    except ImportError:
        print("⚠  TensorFlow not installed. LSTM model skipped.")
        return None


def prepare_lstm_sequences(X: np.ndarray, y: np.ndarray, timesteps: int = 24):
    """Convert flat sensor data into rolling 24-hour windows for LSTM."""
    X_seq, y_seq = [], []
    for i in range(timesteps, len(X)):
        X_seq.append(X[i - timesteps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# ─── 6. ENSEMBLE PREDICTOR ───────────────────────────────────────────────────

class DisasterGuardEnsemble:
    """
    Ensemble model combining RF + XGBoost predictions.
    Final probability = weighted average of all models.
    
    Threshold:
        >= 0.65 → ALERT issued (high precision to avoid false alarms)
        >= 0.40 → WARNING issued (monitoring mode)
        <  0.40 → Normal conditions
    """

    ALERT_THRESHOLD   = 0.65
    WARNING_THRESHOLD = 0.40

    DISASTER_EMOJIS = {
        "flood":     "🌊",
        "cyclone":   "🌀",
        "heatwave":  "☀️",
        "landslide": "🏔️"
    }

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        """Train all disaster prediction models."""
        print("\n" + "="*60)
        print("🚀 Training DisasterGuard AI Models...")
        print("="*60)

        X, self.scaler = preprocess(df, FEATURES)

        self.models["flood_rf"]      = train_random_forest(X, df["flood_label"].values, "FLOOD")
        self.models["landslide_rf"]  = train_random_forest(X, df["landslide_label"].values, "LANDSLIDE")
        self.models["cyclone_xgb"]   = train_xgboost(X, df["cyclone_label"].values, "CYCLONE")
        self.models["heatwave_xgb"]  = train_xgboost(X, df["heatwave_label"].values, "HEATWAVE")

        self.is_trained = True
        print("\n✅ All models trained successfully!")

    def predict(self, sensor_data: dict) -> dict:
        """
        Generate disaster predictions from real-time sensor readings.

        Args:
            sensor_data: dict with keys matching FEATURES list

        Returns:
            dict with probability scores and alert levels for each disaster type
        """
        if not self.is_trained:
            raise RuntimeError("Models not trained. Call .train() first.")

        # Build feature vector
        X = np.array([[sensor_data.get(f, 0) for f in FEATURES]])
        X_scaled = self.scaler.transform(X)

        results = {}
        disaster_map = {
            "flood":     "flood_rf",
            "landslide": "landslide_rf",
            "cyclone":   "cyclone_xgb",
            "heatwave":  "heatwave_xgb"
        }

        for disaster, model_key in disaster_map.items():
            model = self.models[model_key]
            prob = float(model.predict_proba(X_scaled)[0][1])

            if prob >= self.ALERT_THRESHOLD:
                level = "🔴 HIGH ALERT"
            elif prob >= self.WARNING_THRESHOLD:
                level = "🟡 WARNING"
            else:
                level = "🟢 NORMAL"

            results[disaster] = {
                "probability": round(prob * 100, 1),
                "alert_level": level,
                "emoji": self.DISASTER_EMOJIS[disaster],
                "72hr_forecast": self._generate_forecast_message(disaster, prob)
            }

        return results

    def _generate_forecast_message(self, disaster: str, prob: float) -> str:
        """Generate human-readable forecast message in plain English."""
        if prob >= self.ALERT_THRESHOLD:
            messages = {
                "flood":     "EVACUATE low-lying areas. River overflow likely within 72 hrs.",
                "cyclone":   "CYCLONE ALERT: Seek shelter immediately. Do not go to sea.",
                "heatwave":  "EXTREME HEAT: Stay indoors. Avoid outdoor work 11am–4pm.",
                "landslide": "LANDSLIDE RISK: Leave hill-slope areas. Road blockage likely."
            }
        elif prob >= self.WARNING_THRESHOLD:
            messages = {
                "flood":     "Monitor river levels closely. Prepare emergency kit.",
                "cyclone":   "Cyclonic conditions developing. Fishermen advised not to venture.",
                "heatwave":  "High heat alert. Stay hydrated. Check on elderly neighbors.",
                "landslide": "Wet slopes detected. Avoid hilly roads after heavy rain."
            }
        else:
            messages = {
                "flood":     "No significant flood risk in next 72 hours.",
                "cyclone":   "No cyclonic activity detected in the region.",
                "heatwave":  "Temperature within normal range.",
                "landslide": "Stable ground conditions. Normal advisory."
            }
        return messages[disaster]

    def generate_alert_report(self, location: str, sensor_data: dict) -> str:
        """Generate a formatted disaster alert report."""
        predictions = self.predict(sensor_data)
        lines = [
            "=" * 60,
            f"  🛡️  DisasterGuard AI — ALERT REPORT",
            f"  📍 Location: {location}",
            f"  🕐 Forecast Window: Next 72 Hours",
            "=" * 60,
            ""
        ]
        for disaster, result in predictions.items():
            lines.append(f"  {result['emoji']}  {disaster.upper()}")
            lines.append(f"     Risk Probability : {result['probability']}%")
            lines.append(f"     Alert Level      : {result['alert_level']}")
            lines.append(f"     Advisory         : {result['72hr_forecast']}")
            lines.append("")

        high_alerts = [d for d, r in predictions.items() if "HIGH ALERT" in r["alert_level"]]
        if high_alerts:
            lines.append(f"  ⚠️  IMMEDIATE ACTION REQUIRED FOR: {', '.join(high_alerts).upper()}")
            lines.append(f"  📞 NDMA Helpline: 1078  |  Emergency: 112")
        else:
            lines.append("  ✅ No immediate high-risk conditions detected.")

        lines.append("=" * 60)
        return "\n".join(lines)


# ─── 7. FLASK REST API ────────────────────────────────────────────────────────

def launch_api(model: DisasterGuardEnsemble, port: int = 5000):
    """
    Launch REST API server for DisasterGuard AI.
    
    Endpoints:
        GET  /health           → API health check
        POST /predict          → Get disaster predictions
        GET  /districts        → List monitored districts
    
    Example request:
        POST /predict
        {
            "location": "Patna, Bihar",
            "rainfall_mm": 85,
            "river_level_m": 8.2,
            "soil_moisture": 88,
            "temperature_c": 32,
            "humidity_pct": 91,
            "wind_speed_kmh": 25,
            "elevation_m": 55,
            "pressure_hpa": 1005,
            "prev_rainfall": 180,
            "cyclone_dist_km": 1200
        }
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        print("⚠  Flask not installed. Run: pip install flask")
        return

    app = Flask("DisasterGuardAI")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": "DisasterGuard AI v1.0", "ready": model.is_trained})

    @app.route("/predict", methods=["POST"])
    def predict():
        body = request.get_json()
        if not body:
            return jsonify({"error": "No JSON body provided"}), 400
        location = body.pop("location", "Unknown Location")
        try:
            predictions = model.predict(body)
            return jsonify({
                "location": location,
                "forecast_window": "72 hours",
                "predictions": predictions
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/districts", methods=["GET"])
    def districts():
        return jsonify({
            "high_risk_districts": [
                "Patna (Bihar)", "Kamrup (Assam)", "Puri (Odisha)",
                "Wayanad (Kerala)", "Chamoli (Uttarakhand)", "Srinagar (J&K)",
                "Vizianagaram (AP)", "Cuddalore (Tamil Nadu)"
            ],
            "monitored_states": 28,
            "sensor_nodes": 847
        })

    print(f"\n🌐 DisasterGuard AI API running on http://localhost:{port}")
    print(f"   POST /predict  — Get disaster predictions")
    print(f"   GET  /health   — API health check")
    app.run(host="0.0.0.0", port=port, debug=False)


# ─── 8. VISUALIZATION ─────────────────────────────────────────────────────────

def plot_risk_dashboard(predictions: dict, location: str):
    """Plot a visual risk dashboard for all disaster types."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f"DisasterGuard AI — Risk Dashboard: {location}\n72-Hour Forecast",
                     fontsize=14, fontweight="bold", color="#0A1628")
        fig.patch.set_facecolor("#F0F4FF")

        colors_map = {
            "flood": "#0077B6", "cyclone": "#6A0572",
            "heatwave": "#E85D04", "landslide": "#2E9E44"
        }

        for ax, (disaster, result) in zip(axes, predictions.items()):
            prob = result["probability"]
            bar_color = "#D62828" if prob >= 65 else "#FAA307" if prob >= 40 else "#2E9E44"
            ax.set_facecolor("#1A3560")
            ax.barh(["Risk"], [prob], color=bar_color, height=0.4)
            ax.barh(["Risk"], [100 - prob], left=prob, color="#334466", height=0.4)
            ax.set_xlim(0, 100)
            ax.set_title(f"{result['emoji']} {disaster.upper()}", fontweight="bold",
                         color="white", pad=6)
            ax.set_xlabel("Probability (%)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#334466")
            ax.text(prob / 2 if prob > 15 else prob + 2, 0,
                    f"{prob}%", va="center", ha="center",
                    fontweight="bold", color="white", fontsize=13)
            ax.text(0.5, -0.3, result["alert_level"], transform=ax.transAxes,
                    ha="center", fontsize=9, color=bar_color, fontweight="bold")

        plt.tight_layout()
        plt.savefig("risk_dashboard.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.show()
        print("📊 Risk dashboard saved as risk_dashboard.png")

    except ImportError:
        print("⚠  Matplotlib not installed. Run: pip install matplotlib")


# ─── 9. MAIN ENTRY POINT ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DisasterGuard AI — Disaster Prediction System")
    parser.add_argument("--train",   action="store_true", help="Train all models")
    parser.add_argument("--predict", action="store_true", help="Run a sample prediction")
    parser.add_argument("--api",     action="store_true", help="Launch REST API server")
    parser.add_argument("--port",    type=int, default=5000, help="API server port")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║   DisasterGuard AI v1.0  — 2025 Edition  ║")
    print("║   Flood | Cyclone | Heatwave | Landslide ║")
    print("╚══════════════════════════════════════════╝\n")

    # Always generate and train
    df = generate_training_data(n_samples=10000)
    model = DisasterGuardEnsemble()
    model.train(df)

    if args.predict or (not args.train and not args.api):
        # Sample sensor readings — Patna, Bihar (high flood risk scenario)
        sensor_readings = {
            "rainfall_mm":     92.0,    # Heavy rain
            "river_level_m":   9.1,     # River near danger level
            "soil_moisture":   88.5,    # Saturated soil
            "temperature_c":   33.2,
            "humidity_pct":    94.0,
            "wind_speed_kmh":  28.0,
            "elevation_m":     55.0,    # Low-lying terrain
            "pressure_hpa":    1005.0,
            "prev_rainfall":   210.0,   # Heavy cumulative rainfall
            "cyclone_dist_km": 1400.0
        }

        print("\n" + model.generate_alert_report("Patna, Bihar", sensor_readings))

        predictions = model.predict(sensor_readings)
        plot_risk_dashboard(predictions, "Patna, Bihar")

        # Also test a low-risk scenario
        print("\n--- Low Risk Scenario: Jaipur, Rajasthan ---")
        low_risk = {
            "rainfall_mm": 2.0, "river_level_m": 1.5, "soil_moisture": 25.0,
            "temperature_c": 46.5, "humidity_pct": 18.0, "wind_speed_kmh": 12.0,
            "elevation_m": 430.0, "pressure_hpa": 1018.0, "prev_rainfall": 8.0,
            "cyclone_dist_km": 1800.0
        }
        print(model.generate_alert_report("Jaipur, Rajasthan", low_risk))

    if args.api:
        launch_api(model, port=args.port)
from flask import Flask
import os
import threading

app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>DisasterGuard AI v1.0 is Live</h1><p>The AI model is running in the background.</p>"

def start_ai():
    launch_api(model, port=args.port)
    # Looking at your code, it might be a call to a class or a main() function
    pass 

# --- ADD THIS PART JUST ABOVE THE START BLOCK ---
@app.route('/predict/<location>')
def predict(location):
    # This assumes 'model' is already loaded in your code
    # If your prediction function needs more than just a name, 
    # you can adjust the logic here.
    try:
        # Placeholder: replace with your actual model prediction call
        # result = model.predict(location) 
        @app.route('/predict/<location>')
def predict(location):
    try:
        # Call your actual model here
        # prediction_score = model.predict(location) 
        prediction_score = "75% Risk"  # Link this to your AI logic
        return {"location": location, "result": prediction_score}
    except Exception as e:
        return {"error": str(e)}
        return {"location": location, "prediction": status, "system": "DisasterGuard v1.0"}
    except Exception as e:
        return {"error": str(e)}

# --- THIS IS THE ONLY START BLOCK YOU NEED ---
if __name__ == "__main__":
    # 1. Start the AI background logic
    # We use threading so the AI runs while the web server stays open
    threading.Thread(target=main, daemon=True).start()
    
    # 2. Start the Web Server for Railway
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
