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
