from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("model", "temperature_model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Run train_model.py first to create the model.")

model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # Convert and store inputs ONCE (important fix)
        sensor_id = float(request.form["sensor_id"])
        lat = float(request.form["lat"])
        lon = float(request.form["lon"])
        pressure = float(request.form["pressure"])
        humidity = float(request.form["humidity"])

        # Create DataFrame for prediction
        input_df = pd.DataFrame([{
            "sensor_id": sensor_id,
            "lat": lat,
            "lon": lon,
            "pressure": pressure,
            "humidity": humidity
        }])

        # Predict temperature
        prediction = model.predict(input_df)[0]

        # Send SAME processed values to result page
        return render_template(
            "result.html",
            prediction=prediction,
            sensor_id=sensor_id,
            lat=lat,
            lon=lon,
            pressure=pressure,
            humidity=humidity
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
