import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime

app = Flask(__name__, template_folder='xgtemplates')

# Load Models
model = joblib.load('xgboost_diabetes_model.pkl')
encoder = joblib.load('ordinal_encoder.pkl')
scaler = joblib.load('standard_scaler.pkl')

FEATURE_ORDER = ['race', 'gender', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 
                 'diag_1', 'diag_2', 'diag_3', 'metformin', 'insulin', 'change', 
                 'diabetesMed', 'total_visits', 'age_group']

def save_to_excel(patient_data, probability, result):
    file_name = "readmission_records.xlsx"
    patient_data['Prediction_Time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_data['Readmission_Probability'] = f"{probability:.2f}%"
    patient_data['Final_Result'] = result
    
    if os.path.exists(file_name):
        existing_df = pd.read_excel(file_name)
        updated_df = pd.concat([existing_df, patient_data], ignore_index=True)
        updated_df.to_excel(file_name, index=False)
    else:
        patient_data.to_excel(file_name, index=False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_page')
def predict_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        raw_form = request.form.to_dict()
        
        # Numeric conversion
        input_data = raw_form.copy()
        numeric_fields = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                          'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient']
        
        for col in numeric_fields:
            input_data[col] = float(input_data[col])
            
        input_data['total_visits'] = input_data['number_outpatient'] + input_data['number_emergency'] + input_data['number_inpatient']
        
        df_input = pd.DataFrame([input_data])
        raw_data_for_excel = df_input.copy() 
        
        # Preprocessing & ML Features
        df_ml = df_input[FEATURE_ORDER].copy()
        cat_cols = ['race', 'gender', 'diag_1', 'diag_2', 'diag_3', 'metformin', 'insulin', 'change', 'diabetesMed', 'age_group']
        
        # Asian Race handling
        try:
            df_ml[cat_cols] = encoder.transform(df_ml[cat_cols])
        except ValueError:
            df_ml['race'] = df_ml['race'].replace(['Asian', 'Hispanic'], 'Other')
            df_ml[cat_cols] = encoder.transform(df_ml[cat_cols])

        df_input_scaled = scaler.transform(df_ml)
        
        # Prediction
        prob_raw = model.predict_proba(df_input_scaled)[0, 1]
        prob = float(prob_raw)
        result = "High Risk" if prob >= 0.5 else "Low Risk"
        
        # Save to Excel
        save_to_excel(raw_data_for_excel, prob * 100, result)
        
        return jsonify({
            'probability': round(prob * 100, 2),
            'result': result,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/records')
def view_records():
    file_name = "readmission_records.xlsx"
    if os.path.exists(file_name):
        try:
            df = pd.read_excel(file_name)
            # Nan values handle chesi latest records mundu chupinchu
            df = df.fillna("N/A")
            patients = df.iloc[::-1].to_dict(orient='records')
        except Exception as e:
            return f"Error reading records: {str(e)}"
    else:
        patients = []
    return render_template('records.html', patients=patients)

if __name__ == '__main__':
    app.run(debug=True)