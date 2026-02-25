import joblib
import pandas as pd
import numpy as np
import warnings

# Warnings thagginchadaniki
warnings.filterwarnings("ignore")

# 1. Load the saved tools
model = joblib.load('xgboost_diabetes_model.pkl')
encoder = joblib.load('ordinal_encoder.pkl')
scaler = joblib.load('standard_scaler.pkl')

# 2. Test Patient Data (Exact same columns as training)
data = pd.DataFrame([{
    'race': 'Caucasian', 'gender': 'Male', 'time_in_hospital': 8,
    'num_lab_procedures': 60, 'num_procedures': 3, 'num_medications': 20,
    'number_outpatient': 0, 'number_emergency': 0, 'number_inpatient': 4,
    'diag_1': 'Circulatory', 'diag_2': 'Diabetes', 'diag_3': 'Other',
    'metformin': 'No', 'insulin': 'No', 'change': 'No', 'diabetesMed': 'Yes',
    'total_visits': 4, 'age_group': 'Elderly'
}])

# Training lo vaadina feature order ide:
feature_order = ['race', 'gender', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 
                 'diag_1', 'diag_2', 'diag_3', 'metformin', 'insulin', 'change', 
                 'diabetesMed', 'total_visits', 'age_group']

# Reorder columns to match training
data = data[feature_order]

# 3. Process the data
cat_cols = ['race', 'gender', 'diag_1', 'diag_2', 'diag_3', 'metformin', 'insulin', 'change', 'diabetesMed', 'age_group']

# Encode Categorical columns
data[cat_cols] = encoder.transform(data[cat_cols])

# Scale the ENTIRE row (not just num_cols) because scaler was fit on 18 features
data_scaled = scaler.transform(data)

# 4. Predict
prob = model.predict_proba(data_scaled)[0, 1]
result = "High Risk" if prob >= 0.5 else "Low Risk"

print("\n" + "="*30)
print(f"Patient Readmission Analysis")
print("-" * 30)
print(f"Probability: {prob:.2%}")
print(f"Prediction:  {result}")
print("="*30 + "\n")