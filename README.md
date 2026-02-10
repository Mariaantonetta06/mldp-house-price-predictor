# MLDP House Price Predictor

This project was developed as part of the **Machine Learning Development Project (MLDP)** at **Temasek Polytechnic**.

It demonstrates the end-to-end machine learning workflow, including data preprocessing, model training and evaluation, and deployment using a Streamlit web application.

---

## 📊 Dataset
- Source: Kaggle – Ames Housing Dataset
- Files used:
  - `train.csv`
  - `test.csv`
- The dataset contains residential housing data with numerical and categorical features.

---

## 🧠 Model Overview
- Model: Random Forest Regressor
- Target variable: `SalePrice` (log-transformed during training)
- Evaluation metric: R² Score
- Final test R²: **~0.88**
- Predictions are converted back using `expm1` for interpretability.

A full **scikit-learn Pipeline** (preprocessing + model) is saved as:

---

## 🚀 Installation & Running Locally

### Requirements
- Python 3.8+
- Dependencies: see `requirements.txt`

### Setup
```bash
git clone https://github.com/yourusername/mldp-house-price-predictor
cd mldp-house-price-predictor
pip install -r requirements.txt
```

### Run Locally
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

---

## 🌐 Live Deployment

**Deployed on Streamlit Cloud:**
(https://mldp-house-price-predictor-9c2bxzwctajhqbqgkuan6j.streamlit.app/)

---

## 📊 Project Structure