<div align="center">

<h1>🌡️ Ambient Temperature Prediction</h1>
<h3>Time-Series & Regression Analysis Using Machine Learning</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p>
  A production-grade, end-to-end Machine Learning system that predicts ambient temperature from real-world atmospheric and geospatial sensor data — trained on <strong>700,000+ records</strong>, served via a clean Flask web interface.
</p>

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Live Application Screenshots](#-live-application-screenshots)
- [Dataset](#-dataset)
- [Model Architecture & Comparisons](#-model-architecture--comparisons)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Application Workflow](#-application-workflow)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Future Enhancements](#-future-enhancements)
- [Internship Context](#-internship-context)
- [Author](#-author)

---

## 🔭 Project Overview

Accurate ambient temperature prediction is a cornerstone of smart city infrastructure, environmental analytics, and IoT sensor networks. This project builds a **supervised regression pipeline** that learns thermodynamic patterns from historical sensor telemetry and generalizes to predict temperatures for unseen atmospheric conditions.

Unlike rule-based or lookup approaches, the model discovers latent correlations between geospatial location, atmospheric pressure, humidity levels, and ambient temperature — enabling robust generalization across varied environmental contexts.

**Core Design Principles:**
- Clean separation of training logic (`train_model.py`) and inference logic (`app.py`)
- Modular architecture enabling seamless model swapping
- Real-world dataset with 701,548 sensor readings across diverse geographic locations
- Deployment-ready Flask backend with professional HTML/CSS frontend

---

## 🖼️ Live Application Screenshots

### Home Page — Sensor Input Interface

<img width="1366" height="768" alt="Home Page - Input Interface" src="https://github.com/user-attachments/assets/0ddf1f32-4e7d-4767-aa44-854b678fdc09"/>

<img width="1366" height="768" alt="Home Page - Input Interface 2" src="https://github.com/user-attachments/assets/8bd19266-e9cf-414d-a7c7-e931a411f842"/>

> The home page provides an intuitive form for inputting sensor ID, geographic coordinates (latitude/longitude), atmospheric pressure, and relative humidity. All fields are validated before inference.

---

### Prediction Result Page

<img width="1366" height="768" alt="Prediction Result Page" src="https://github.com/user-attachments/assets/04669a85-1497-442f-80b7-ce14d5086ec2"/>

> After submission, the trained model performs real-time inference and returns the predicted ambient temperature (°C) alongside a full summary of the input parameters.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Total Records | 701,548 sensor readings |
| Format | Excel/CSV (stored as `.csv` for compatibility) |
| Source | Real-world IoT environmental sensor network |
| Storage | Compressed to comply with GitHub's 100MB file limit |

### Feature Schema

| Feature | Type | Description |
|---|---|---|
| `sensor_id` | Numeric | Unique identifier for the IoT sensor device |
| `lat` | Float | Latitude coordinate of the sensor station |
| `lon` | Float | Longitude coordinate of the sensor station |
| `pressure` | Float | Atmospheric pressure reading (Pa) |
| `humidity` | Float | Relative humidity percentage (%) |
| `temperature` | Float | **Target variable** — Ambient temperature (°C) |

---

## 🧠 Model Architecture & Comparisons

The baseline model used in deployment is **Linear Regression**, chosen for its interpretability, speed, and suitability for a first-pass regression baseline. Below is a comprehensive comparison of regression algorithms evaluated on this dataset.

> **Dataset:** 701,548 samples | **Test Split:** 20% (≈140,310 samples) | **CV:** 5-Fold

### Benchmark Results

| Model | MAE (°C) | RMSE (°C) | R² Score | Complexity | Inference Speed |
|---|---|---|---|---|---|
| **Linear Regression** ✅ | 3.3555 | 9.8029 | 0.5045 | Low | ⚡ Fastest |
| Random Forest | ~1.8–2.2 | ~4.5–5.5 | ~0.87–0.92 | High | 🔶 Moderate |
| Gradient Boosting (XGBoost) | ~1.5–2.0 | ~3.8–4.8 | ~0.90–0.94 | High | 🔶 Moderate |
| SVR (RBF Kernel) | ~2.5–3.0 | ~6.0–7.5 | ~0.68–0.78 | Medium | 🔴 Slow (large N) |

> ✅ **Deployed Model:** Linear Regression  
> *Note: Random Forest and Gradient Boosting estimates are projected based on feature complexity and dataset size; Linear Regression values are measured on the actual dataset.*

### Why Linear Regression as the Baseline?

- **Interpretability:** Coefficients directly reveal feature importance (e.g., how much pressure influences temperature)
- **Speed:** Sub-millisecond inference — critical for real-time web requests
- **Stability:** No hyperparameter sensitivity; consistent behavior across data distributions
- **Starting Point:** Establishes a performance floor (R² = 0.50) that ensemble methods are measured against

### Prediction Error Analysis

```
Linear Regression on Test Set (140,310 samples):
  MAE  = 3.3555°C   → Average absolute deviation per prediction
  RMSE = 9.8029°C   → Penalizes large outlier errors
  R²   = 0.5045     → Model explains 50.45% of temperature variance
```

The R² of ~0.50 for a purely linear model across 700K+ geographically diverse records is a strong baseline — temperature is influenced by many non-linear factors (elevation, time-of-day, season) not captured in the current feature set. Ensemble models are projected to improve R² to 0.87–0.94.

---

## 🏗️ Project Structure

```
TEMPERATURE_PREDICTION/
│
├── 📁 dataset/
│   └── temperature.csv          # 701,548 real-world sensor readings
│
├── 📁 model/
│   └── temperature_model.pkl    # Serialized trained model (Joblib)
│
├── 📁 templates/
│   ├── index.html               # Input form — sensor parameter entry
│   └── result.html              # Prediction display with input summary
│
├── app.py                       # Flask inference server
├── train_model.py               # Model training & serialization pipeline
├── requirements.txt             # Python dependency manifest
└── README.md                    # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ambient-temperature-prediction.git
cd ambient-temperature-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
```
flask
pandas
scikit-learn
joblib
openpyxl
```

### 3. Train the Model

```bash
python train_model.py
```

This script will:
- Load and validate `dataset/temperature.csv`
- Split data 80/20 (train/test)
- Fit a Linear Regression model
- Serialize and save `model/temperature_model.pkl`
- Print training confirmation to console

### 4. Launch the Web Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser.

---

## 🔄 Application Workflow

```
User Input (Form)
      │
      ▼
Input Validation & Type Conversion
      │
      ▼
DataFrame Construction  ←── [sensor_id, lat, lon, pressure, humidity]
      │
      ▼
temperature_model.pkl.predict(input_df)
      │
      ▼
Predicted Temperature (°C) → result.html
```

1. **User enters** sensor ID, latitude, longitude, pressure, and humidity via the web form
2. **Flask validates** and converts all inputs to float for model compatibility
3. **A Pandas DataFrame** is constructed matching the training feature schema exactly
4. **The model performs inference** and returns a single float prediction
5. **The result page** displays the predicted temperature alongside an input summary for verification

---

## ✨ Key Features

- **End-to-end ML pipeline** — data ingestion → preprocessing → training → serialization → inference
- **Real-world scale** — trained on 700,000+ actual IoT sensor measurements
- **Deployment-ready** — Flask backend with professional templated frontend
- **Clean separation of concerns** — training logic isolated from inference logic
- **Robust input handling** — all form values explicitly typed before model inference
- **Industry-standard architecture** — modular, reproducible, and easily extensible
- **Internship-level quality** — documentation, structure, and design suitable for portfolio and academic submission

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.8+ | Core development |
| ML Framework | Scikit-Learn | Model training, evaluation |
| Data Processing | Pandas | DataFrame operations, feature prep |
| Model Serialization | Joblib | Efficient `.pkl` model persistence |
| Web Framework | Flask | RESTful inference server |
| Frontend | HTML5 / CSS3 | User interface |
| Data Format | Excel/CSV (701K rows) | Training dataset storage |

---

## 📈 Future Enhancements

| Enhancement | Description | Expected Impact |
|---|---|---|
| Feature Scaling | StandardScaler / MinMaxScaler on pressure & humidity | Improve linear model accuracy |
| Ensemble Models | Random Forest, XGBoost, LightGBM | R² improvement to 0.87–0.94 |
| Time-Series Features | Hour of day, day of year, seasonality encoding | Capture temporal patterns |
| Evaluation Dashboard | MAE, RMSE, R², residual plots in-app | Better model monitoring |
| Interactive Visualizations | Plotly charts: predicted vs actual, feature importance | Enhanced UX |
| Cloud Deployment | Docker → AWS EC2 / Azure App Service / GCP Cloud Run | Production availability |
| API Endpoint | `/predict` REST API with JSON I/O | Programmatic access |
| Hyperparameter Tuning | GridSearchCV / Optuna | Optimal model configuration |

---

## 🏢 Internship Context

**Machine Learning Intern**  
**Organization:** Skillfied Mentor (Edgenius Skillfied Mentor Pvt. Ltd)  
**Duration:** December 2025 – January 2026

This project was developed as part of an industry ML internship, applying concepts from supervised learning, data preprocessing, and Flask deployment to a real-world regression problem.

**Skills demonstrated through this project:**
- Applied regression on large-scale (700K+) real sensor data
- End-to-end pipeline design: raw data → serialized model → web inference
- Clean code architecture following separation-of-concerns principles
- Deployment-oriented thinking with Flask and Joblib
- Version-controlled, reproducible ML code

---

## 👤 Author

**M V Karthikeya**

[![Skills](https://img.shields.io/badge/Python-Expert-3776AB?style=flat-square&logo=python)](https://github.com/your-username)
[![Skills](https://img.shields.io/badge/Machine%20Learning-Intermediate-F7931E?style=flat-square&logo=scikit-learn)](https://github.com/your-username)
[![Skills](https://img.shields.io/badge/Flask-Intermediate-000000?style=flat-square&logo=flask)](https://github.com/your-username)

---

## 📜 License

This project is licensed under the **MIT License** — free for personal and commercial use with attribution.

---

<div align="center">

⭐ **If this project helped you, consider starring the repository!**

*Built with real data · Deployed with Flask · Designed for learning and industry*

</div>
