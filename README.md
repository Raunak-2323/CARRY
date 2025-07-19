
# AutoVal AI – A Machine Learning Based Car Price Prediction System

This project predicts the **selling price of used cars** using real-world data and machine learning models.  
It includes a complete ML pipeline — from **data cleaning** to **model building**, **evaluation**, and **saving the trained model** for future use.

---

## 🔍 Project Overview

- **Goal**: Predict used car prices based on features like year, fuel type, transmission, mileage, power, and brand.
- **Dataset**: `Cardetails.csv` (includes car names, specs, and prices)
- **Tech Stack**: Python, Pandas, NumPy, scikit-learn, Random Forest, Linear Regression, Pickle

---

## 🧠 Machine Learning Workflow

### 1. **Data Cleaning & Preprocessing**
- Dropped irrelevant columns like `torque`, `index`
- Removed missing values and duplicates
- Extracted `brand name` from the `name` column
- Cleaned textual numeric values like `"100 bhp"` → `100.0`
- Encoded categorical columns (`Manual` = 1, `Automatic` = 2)

### 2. **Feature Engineering**
- Defined `input_data` and `output_data` (`selling_price`)
- Converted all input features to numerical format

### 3. **Train-Test Split**
- Used `train_test_split()` to split data: 80% training, 20% testing

### 4. **Model Building**
- Built and compared two models:
  - ✅ **Linear Regression** (Baseline) → R² = 0.61
  - 🌳 **Random Forest Regressor** → R² = 0.93

### 5. **Model Evaluation**
- Evaluated using:
  - `R² Score`: Measures model accuracy
  - `RMSE`: Root Mean Squared Error

### 6. **Model Saving**
- Saved the trained Random Forest model using `pickle`:
  ```python
  with open('car_price_model.pkl', 'wb') as file:
      pk.dump(rf_model, file)
