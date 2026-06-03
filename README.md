# 🌡️ Ambient Temperature Prediction — Time-Series & Regression Analysis

> **End-to-end Machine Learning system for real-time ambient temperature inference using atmospheric sensor data.**  
> Built with Python · Scikit-learn · Flask · Pandas | 701,548 real-world records

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)  
2. [Live Demo & Screenshots](#-live-demo--screenshots)  
3. [Dataset](#-dataset)  
4. [Machine Learning Pipeline](#-machine-learning-pipeline)  
5. [Model Comparison & Benchmarks](#-model-comparison--benchmarks)  
6. [Project Structure](#-project-structure)  
7. [Getting Started](#-getting-started)  
8. [Application Workflow](#-application-workflow)  
9. [Technologies Used](#-technologies-used)  
10. [Future Roadmap](#-future-roadmap)  
11. [Author](#-author)  
12. [License](#-license)

---

## 🎯 Project Overview

Accurate ambient temperature prediction is foundational to **climate monitoring**, **smart city infrastructure**, **IoT sensor networks**, and **environmental analytics**. This project delivers a production-grade, end-to-end Machine Learning system that learns the underlying statistical relationships between atmospheric features and temperature, rather than relying on rule-based or lookup approaches.

### Core Objectives

| Goal | Approach |
|------|----------|
| Learn from real-world sensor data | Supervised Regression on 701K+ records |
| Generalize beyond training data | Train/Test split with random state control |
| Serve predictions in real-time | REST-style Flask web application |
| Maintain clean, reproducible code | Modular separation of training & inference |

### Problem Statement

**Type:** Supervised Learning — Regression  
**Target Variable:** `temperature` (°C)  
**Input Features:** `sensor_id`, `lat`, `lon`, `pressure`, `humidity`

The model generalizes from historical patterns — minor deviations between prediction and ground truth (≈ ±1–2°C) indicate healthy generalization, not failure.

---

## 🖥️ Live Demo & Screenshots

### 🔹 Home Page — Sensor Input Interface

> Users enter atmospheric sensor parameters through a clean, professional web form.

![Home Page Input Interface](https://github.com/user-attachments/assets/0ddf1f32-4e7d-4767-aa44-854b678fdc09)

![Alternate Home View](https://github.com/user-attachments/assets/8bd19266-e9cf-414d-a7c7-e931a411f842)

---

### 🔹 Prediction Result Page

> After form submission, the trained ML model returns the predicted ambient temperature alongside a full input summary.

![Prediction Result Page](https://github.com/user-attachments/assets/04669a85-1497-442f-80b7-ce14d5086ec2)

---

### 🔹 End-to-End Workflow

```
 User Input (Web Form)
        │
        ▼
 Input Validation & Type Conversion
        │
        ▼
 Pandas DataFrame Construction
        │
        ▼
 Trained Model Inference (temperature_model.pkl)
        │
        ▼
 Predicted Temperature → Result Page
```

---

## 📊 Dataset

The dataset comprises real-world environmental sensor measurements collected across a geographic grid.

| Attribute | Value |
|-----------|-------|
| **Total Records** | 701,548 |
| **Features** | 6 (5 input + 1 target) |
| **Format** | Excel-compressed CSV |
| **Source** | Real-world IoT sensor network |

### Feature Description

| Feature | Type | Description |
|---------|------|-------------|
| `sensor_id` | Numerical | Unique sensor identifier (1764 – 4661) |
| `lat` | Float | Latitude coordinate (42.62 – 42.74) |
| `lon` | Float | Longitude coordinate |
| `pressure` | Float | Atmospheric pressure in Pascals |
| `humidity` | Float | Relative humidity (%) |
| `temperature` | Float | **Target** — Ambient temperature (°C) |

### Statistical Summary

| Statistic | Temperature (°C) | Humidity (%) | Pressure (Pa) |
|-----------|-----------------|--------------|----------------|
| Mean | 24.75 | 48.35 | ~94,500 |
| Std Dev | 14.01 | 20.91 | — |
| Min | −145.12 | 0.00 | — |
| Median (50%) | 24.78 | 48.35 | — |
| Max | 61.17 | 100.00 | — |

> ⚠️ **Note:** The dataset is stored in compressed format to comply with GitHub file size limits while maintaining full reproducibility.

---

## 🧠 Machine Learning Pipeline

```
Raw Dataset (temperature.csv)
        │
        ▼
  Feature Selection
  [sensor_id, lat, lon, pressure, humidity]
        │
        ▼
  Train / Test Split (80% / 20%, random_state=42)
        │
        ▼
  Model Training (Linear Regression)
        │
        ▼
  Model Serialization (joblib → temperature_model.pkl)
        │
        ▼
  Flask API Inference (app.py)
        │
        ▼
  Predicted Temperature (°C)
```

### Training Details

- **Algorithm:** Ordinary Least Squares Linear Regression
- **Train Split:** 80% (561,238 samples)
- **Test Split:** 20% (140,310 samples)
- **Random State:** 42 (fully reproducible)
- **Serialization:** `joblib` (fast binary pickle)

---

## 📈 Model Comparison & Benchmarks

The following benchmarks were computed on the **held-out test set (140,310 samples)** from the full 701,548-record dataset. All models use identical train/test splits for a fair comparison.

| Model | MAE (°C) ↓ | RMSE (°C) ↓ | R² Score ↑ | Complexity |
|-------|-----------|------------|-----------|------------|
| **Linear Regression** *(current)* | 3.3555 | 9.8029 | 0.5045 | Low |
| Ridge Regression (α=1.0) | 3.3555 | 9.8029 | 0.5045 | Low |
| Lasso Regression (α=0.1) | 3.7787 | 9.9772 | 0.4867 | Low |
| Decision Tree (depth=10) | 1.8003 | 2.3909 | **0.9705** | Medium |
| **Random Forest** *(recommended)* | **1.7635** | **2.3544** | **0.9714** | High |
| Gradient Boosting (depth=5) | 1.8210 | 2.4070 | 0.9701 | High |

> **Metrics Explained:**  
> — **MAE** (Mean Absolute Error): Average prediction error in °C — lower is better  
> — **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily — lower is better  
> — **R² Score**: Proportion of variance explained (1.0 = perfect) — higher is better

### Key Findings

- **Linear Regression** (current implementation) establishes a solid baseline with R² = 0.50, indicating the features have a partially non-linear relationship with temperature.
- **Random Forest** achieves the strongest performance across all three metrics, with R² = **0.9714** and MAE of only **1.76°C** — a 47% improvement in MAE over Linear Regression.
- **Decision Tree and Gradient Boosting** both achieve comparable ensemble-level performance (R² ≈ 0.97).
- Linear and Ridge Regression produce nearly identical results, confirming that regularization has minimal effect given the feature scale.

### 📉 Performance Visualization

```
R² Score Comparison (higher = better)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Random Forest     ████████████████████████████████  0.9714
Decision Tree     ████████████████████████████████  0.9705
Gradient Boosting ████████████████████████████████  0.9701
Linear Regression ████████████████                  0.5045
Ridge Regression  ████████████████                  0.5045
Lasso Regression  ███████████████                   0.4867
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAE (lower = better)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Random Forest     ██                                1.76°C
Decision Tree     ██                                1.80°C
Gradient Boosting ██                                1.82°C
Linear Regression ████                              3.36°C
Ridge Regression  ████                              3.36°C
Lasso Regression  █████                             3.78°C
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> 🚀 **Upgrade Path:** Replacing Linear Regression with Random Forest is the single highest-impact improvement. See [Future Roadmap](#-future-roadmap) for implementation guidance.

---

## 🗂️ Project Structure

```
TEMPERATURE_PREDICTION/
│
├── dataset/
│   └── temperature.csv          # Real-world sensor dataset (701K records, compressed)
│
├── model/
│   └── temperature_model.pkl    # Serialized trained ML model (joblib)
│
├── templates/
│   ├── index.html               # User input interface (Flask template)
│   └── result.html              # Prediction results page (Flask template)
│
├── app.py                       # Flask web application & inference logic
├── train_model.py               # Model training & serialization script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ambient-temperature-prediction.git
cd ambient-temperature-prediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

```
flask
pandas
scikit-learn
joblib
```

### 3️⃣ Train the Model

```bash
python train_model.py
```

**Output:**
```
✅ Model trained and saved successfully!
```

This generates `model/temperature_model.pkl` — the serialized inference artifact.

### 4️⃣ Launch the Web Application

```bash
python app.py
```

Navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## ⚙️ Application Workflow

```
Step 1 │ User fills the web form with:
       │   sensor_id · lat · lon · pressure · humidity
       │
Step 2 │ Flask receives POST request
       │ Inputs are validated and cast to float
       │
Step 3 │ Pandas DataFrame is constructed from inputs
       │
Step 4 │ Pre-trained model (temperature_model.pkl)
       │ performs inference via model.predict()
       │
Step 5 │ Predicted temperature (°C) + input summary
       │ is rendered on the result page
```

---

## 🛠️ Technologies Used

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.x | Core development |
| ML Framework | Scikit-learn | Model training & inference |
| Data Processing | Pandas | Feature engineering & input handling |
| Serialization | Joblib | Model persistence |
| Web Framework | Flask | REST-style inference server |
| Frontend | HTML5 & CSS3 | User interface |

---

## 🔮 Future Roadmap

| Enhancement | Priority | Expected Impact |
|-------------|----------|-----------------|
| 🌲 Replace Linear Regression with Random Forest | **High** | R² from 0.50 → 0.97 |
| ⚖️ Feature scaling with StandardScaler | Medium | Improves linear model stability |
| 📊 Model evaluation dashboard (MAE, RMSE, R²) | Medium | Better production monitoring |
| 📈 Interactive data visualizations (Plotly/Altair) | Medium | Explainability & UX |
| ⏱️ Time-series forecasting (LSTM / Prophet) | Medium | Temporal pattern capture |
| ☁️ Cloud deployment (AWS / Azure / GCP) | Low | Production scalability |
| 🔁 CI/CD pipeline for model retraining | Low | Automated model lifecycle |

---

## 🏢 Internship Background

**Machine Learning Intern**  
**Organization:** Skillfied Mentor — Edgenius Skillfied Mentor Pvt. Ltd  
**Duration:** December 2025 – January 2026

This project was developed during and following an industry ML internship, incorporating professional discipline in:

- End-to-end ML pipeline design (data → training → inference)
- Clean project architecture and separation of concerns
- Deployment-oriented thinking with Flask
- Reproducible ML code and documentation standards

---

## 👤 Author

**M V Karthikeya**

`Python` · `Machine Learning` · `Flask` · `Data Analysis` · `Scikit-learn`

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

> ⭐ **If this project helped you, consider starring the repository.**  
> Pull requests for model improvements are welcome!
