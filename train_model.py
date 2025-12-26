import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load dataset
data = pd.read_csv("dataset/temperature.csv")

# Select features safely
if "temperature" not in data.columns:
    raise Exception("Dataset must contain a 'temperature' column")

X = data.drop(columns=["temperature"])
y = data["temperature"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/temperature_model.pkl")

print("✅ Model trained and saved successfully!")
