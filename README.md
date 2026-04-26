# 🏎️ F1 Race Strategy Optimizer
### Ferrari-Themed Predictive Racing Intelligence System

![Python](https://img.shields.io/badge/Python-3.10+-red?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-F1%3A0.82-darkred?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.99-cc0000?style=flat-square)
![Backtest](https://img.shields.io/badge/Backtest%20Accuracy-80.7%25-red?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-Deployed-009688?style=flat-square&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?style=flat-square&logo=kaggle)

---

> *"To finish first, first you must finish."* — Enzo Ferrari

An end-to-end machine learning pipeline that predicts optimal F1 pit stop windows and tyre degradation patterns using 5 seasons of telemetry data. Built with a full ML stack — from raw data to a deployed FastAPI + Streamlit interface — with a Ferrari-themed UI.

---

## 📌 Project Overview

Modern F1 racing is won and lost in the pit lane. This project builds a **data-driven race strategy system** that:

- **Classifies optimal pit stop windows** using XGBoost on lap-by-lap telemetry features
- **Forecasts tyre degradation** using an LSTM model trained on compound performance sequences
- **Backtests strategy decisions** across 5 historical seasons to validate model reliability
- **Deploys a real-time dashboard** via FastAPI + Streamlit for live race simulation

---

## 🗂️ Repository Structure

```
f1-race-strategy-optimizer/
│
├── notebooks/
│   ├── NB1_EDA_Feature_Engineering.ipynb       # Data loading, cleaning, feature construction
│   ├── NB2_XGBoost_Pit_Stop_Classifier.ipynb   # Pit stop window classification model
│   ├── NB3_Backtesting_Engine.ipynb            # Historical strategy validation
│   ├── NB4_LSTM_Tyre_Degradation.ipynb         # Sequence model for tyre wear forecasting
│   └── NB5_FastAPI_Streamlit_Dashboard.ipynb   # API backend + Ferrari-themed UI
│
├── assets/
│   └── ferrari_dashboard_preview.png           # UI screenshot
│
├── requirements.txt
└── README.md
```

---

## 🔬 Pipeline Breakdown

### NB1 — EDA & Feature Engineering
- Loaded and merged multi-season F1 telemetry data from Kaggle
- Engineered features: lap delta, tyre age, compound encoding, stint position, track temperature, safety car flags
- Visualized pit stop distributions, tyre degradation curves, and lap time trends

### NB2 — XGBoost Pit Stop Classifier
- Framed pit stop prediction as a **binary classification** problem (pit now vs stay out)
- Trained XGBoost with hyperparameter tuning via GridSearchCV
- **Results:**
  - F1-Score: **0.82**
  - ROC-AUC: **0.99**
- SHAP analysis for feature importance and model interpretability

### NB3 — Backtesting Engine
- Simulated model-driven strategy decisions across **5 historical F1 seasons**
- Compared model decisions against actual race outcomes
- **Backtest Accuracy: 80.7%**
- Evaluated undercut/overcut detection, safety car window exploitation

### NB4 — LSTM Tyre Degradation Forecasting
- Modelled lap-by-lap compound performance as a **time series**
- LSTM architecture trained on stint sequences per compound type (Soft / Medium / Hard)
- Predicts degradation slope and optimal stint length per compound
- Deployed on Kaggle T4 GPU infrastructure

### NB5 — FastAPI + Streamlit Dashboard
- FastAPI backend exposing `/predict` and `/degrade` inference endpoints
- Ferrari-themed Streamlit UI for real-time race strategy simulation
- Input: current lap, tyre compound, tyre age, track conditions
- Output: pit recommendation, predicted degradation curve, strategy confidence score

---

## 📊 Key Results

| Metric | Value |
|---|---|
| XGBoost F1-Score | 0.82 |
| XGBoost ROC-AUC | 0.99 |
| Backtest Accuracy (5 seasons) | 80.7% |
| LSTM Input Sequence Length | Lap-by-lap per stint |
| Deployment | FastAPI + Streamlit on Kaggle T4 |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & EDA | Pandas, NumPy, Matplotlib, Seaborn |
| Classical ML | XGBoost, Scikit-learn, SHAP |
| Deep Learning | PyTorch / Keras LSTM |
| Backtesting | Custom Python engine |
| API | FastAPI |
| Frontend | Streamlit |
| Platform | Kaggle (T4 GPU) |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/sharmaaditya2003/F1-Race-Strategy-Optimizer.git
cd F1-Race-Strategy-Optimizer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset
This project uses the [Formula 1 World Championship Dataset](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) available on Kaggle.

Download and place in a `/data` folder, or run directly on Kaggle to access the dataset natively.

### 4. Run notebooks in order
```
NB1 → NB2 → NB3 → NB4 → NB5
```

### 5. Launch the dashboard (NB5)
```bash
uvicorn app:app --reload        # Start FastAPI
streamlit run dashboard.py      # Launch Streamlit UI
```

---

## 🏁 Motivation

Formula 1 strategy is one of the most high-stakes real-time decision problems in sport. A single wrong pit call can cost a race win — or a championship. This project applies modern ML to that problem, combining classical tree-based models with deep sequence learning, validated against real race history.

Built as a Ferrari fan. Engineered as a data scientist.

---

## 👤 Author

**Aditya Sharma**
MTech Data Science (Business Analytics) — NMIMS University
[LinkedIn](https://linkedin.com/in/aditya-sharma-097057250) · [Kaggle](https://www.kaggle.com/adityasharma100203) · [GitHub](https://github.com/sharmaaditya2003)

---

*Forza Ferrari. 🔴*
