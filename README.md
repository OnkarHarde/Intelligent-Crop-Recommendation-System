# 🌾 Intelligent Crop Recommendation System

This Streamlit web application predicts the **most suitable crop** to grow based on soil and weather conditions using **machine learning**.  
It helps farmers and agricultural experts make **data-driven decisions** to improve yield and sustainability.
Streamlit app URL:https://intelligent-crop-recommendation-system-liqmnjqojau6l8gdricfep.streamlit.app/
---

## 🚀 Features

- 📊 **Automated Model Training & Comparison**
  - Trains Logistic Regression, KNN, SVM, and Random Forest
  - Displays accuracy comparison chart

- 🌿 **Two Prediction Modes**
  1. **Manual Input** – Enter soil & climate data directly  
  2. **CSV Upload** – Upload multiple records for batch prediction

- 📈 **Visual Analytics**
  - Crop distribution chart
  - Model performance visualization

- 📥 **Download Predictions**
  - Save predicted crops as a CSV file

---

## 🧠 Dataset Information

**Dataset File:** `Crop_recommendation.csv`  

| Feature | Description |
|----------|-------------|
| `N` | Nitrogen content in soil (mg/kg) |
| `P` | Phosphorus content in soil (mg/kg) |
| `K` | Potassium content in soil (mg/kg) |
| `temperature` | Average temperature (°C) |
| `humidity` | Average relative humidity (%) |
| `ph` | Soil pH value |
| `rainfall` | Rainfall (mm) |
| `label` | Most suitable crop (Target variable) |

---

