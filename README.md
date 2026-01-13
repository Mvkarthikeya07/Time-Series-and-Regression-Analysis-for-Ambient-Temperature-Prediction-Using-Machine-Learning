🌡 Time-Series and Regression Analysis for Ambient Temperature Prediction Using Machine Learning

A professional, end-to-end Machine Learning web application that predicts ambient temperature using environmental and sensor data.
The system demonstrates the complete ML lifecycle — from data processing and model training to real-time inference via a clean Flask-based web interface.

📌 Project Overview

Accurate temperature prediction is a critical component of climate monitoring, smart city infrastructure, and environmental analytics.

This project focuses on building a statistical, learning-based prediction system, rather than a rule-based or lookup approach.
The model learns underlying patterns from historical sensor data and generalizes them to predict temperature for unseen inputs.

Key Design Goals

Clean and modular architecture

Reliable end-to-end ML workflow

Real-world data handling

Professional deployment-ready structure

🧠 Machine Learning Details

Problem Type: Supervised Learning (Regression)

Model Used: Linear Regression

Target Variable: Temperature (°C)

The model learns relationships between atmospheric features and temperature rather than memorizing individual records.
Minor deviations between predicted and actual values are expected and desirable, indicating healthy model generalization.

🗂 Dataset Information

The dataset consists of real-world environmental and sensor measurements.

Features Used

sensor_id

lat — Latitude

lon — Longitude

pressure — Atmospheric pressure (Pa)

humidity — Relative humidity (%)

temperature — Target variable (°C)

⚠️ Note:
The dataset is stored in a compressed format to comply with GitHub file size limitations while maintaining full project reproducibility.

🏗 Project Structure
TEMPERATURE_PREDICTION/
│
├── dataset/
│   └── temperature.csv          # Compressed dataset
│
├── model/
│   └── temperature_model.pkl    # Trained ML model
│
├── templates/
│   ├── index.html               # User input interface
│   └── result.html              # Prediction results page
│
├── app.py                       # Flask web application
├── train_model.py               # Model training script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation

🖼 Application Screenshots
🔹 Home Page – User Input Interface

<img width="1366" height="768" alt="Screenshot (41)" src="https://github.com/user-attachments/assets/0ddf1f32-4e7d-4767-aa44-854b678fdc09" />

<img width="1366" height="768" alt="Screenshot (40)" src="https://github.com/user-attachments/assets/8bd19266-e9cf-414d-a7c7-e931a411f842" />

The home page allows users to enter sensor and atmospheric parameters through a clean and intuitive form.

🔹 Prediction Result Page

<img width="1366" height="768" alt="Screenshot (42)" src="https://github.com/user-attachments/assets/04669a85-1497-442f-80b7-ce14d5086ec2" />

After submitting the inputs, the trained machine learning model predicts the ambient temperature and displays the result along with an input summary.

🔹 Workflow Visualization (Optional)

Illustrates the end-to-end flow from user input to model inference and result display.

🚀 How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/your-username/smart-climate-temperature-prediction.git
cd smart-climate-temperature-prediction

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model
python train_model.py


This generates:

model/temperature_model.pkl

4️⃣ Run the Web Application
python app.py


Open your browser and navigate to:

http://127.0.0.1:5000

🖥 Application Workflow

User enters sensor and atmospheric parameters

Inputs are validated and converted into numerical format

The trained ML model performs temperature prediction

Results are displayed on a dedicated output page with input summary

🎯 Key Features

End-to-end Machine Learning pipeline

Clean and professional Flask-based UI

Clear separation of training and inference logic

Robust handling of real-world numerical inputs

Industry-standard project structure

Resume, interview, and academic-ready design

📊 Model Interpretation

Predictions may not exactly match existing dataset values

This behavior is expected and correct in ML systems

A small prediction error (≈ ±1°C) indicates effective learning and generalization

🛠 Technologies Used

Python

Flask

Pandas

Scikit-learn

Joblib

HTML & CSS

📈 Future Enhancements

Feature scaling and normalization

Advanced regression models (Random Forest, XGBoost)

Model evaluation metrics (MAE, RMSE, R²)

Interactive data visualizations

Cloud deployment (AWS / Azure / GCP)

👤 Author

M V Karthikeya

Skills:
Python • Machine Learning • Flask • Data Analysis

📜 License

This project is licensed under the MIT License.

⭐ If you find this project useful, consider starring the repository.
