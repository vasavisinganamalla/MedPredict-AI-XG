import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# 1. Loading Dataset
df = pd.read_csv("diabetic_data.csv")
df.replace('?', np.nan, inplace=True) 

# --- EDA & Visualizations (Mee Original Steps) ---
print("\nReadmission Counts:")
display(df['readmitted'].value_counts())
print("\nReadmission Percentage:")
print(df['readmitted'].value_counts(normalize=True)*100)

plt.figure(figsize=(5,4))
sns.countplot(x='readmitted', data=df)
plt.title("Readmission Count (Original)")
plt.show()

numeric_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap")
plt.show()

# --- Data Preprocessing ---
unique_cols = [col for col in df.columns if df[col].nunique() == len(df)]
constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.4]

df.drop(columns=list(set(unique_cols + constant_cols + missing_cols)), inplace=True, errors='ignore')

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# --- Diagnosis Regrouping (Essential Change) ---
def map_diagnosis(code):
    try:
        if 'V' in str(code) or 'E' in str(code): return 'Other'
        code = float(code)
        if (390 <= code <= 459) or code == 785: return 'Circulatory'
        elif (460 <= code <= 519) or code == 786: return 'Respiratory'
        elif (520 <= code <= 579) or code == 787: return 'Digestive'
        elif int(code) == 250: return 'Diabetes'
        elif 800 <= code <= 999: return 'Injury'
        elif 710 <= code <= 739: return 'Musculoskeletal'
        elif 580 <= code <= 629 or code == 788: return 'Genitourinary'
        elif 140 <= code <= 239: return 'Neoplasms'
        else: return 'Other'
    except: return 'Other'

for col in ['diag_1', 'diag_2', 'diag_3']:
    df[col] = df[col].apply(map_diagnosis)

# Target conversion
df['readmitted'] = df['readmitted'].map({'<30': 1, '>30': 0, 'NO': 0})

# --- Feature Engineering ---
df['total_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
def categorize_age(age):
    if age in ['[0-10)', '[10-20)', '[20-30)']: return 'Young'
    elif age in ['[30-40)', '[40-50)', '[50-60)']: return 'Middle-aged'
    else: return 'Elderly'
df['age_group'] = df['age'].apply(categorize_age)

selected_cols = ['race', 'gender', 'time_in_hospital', 'num_lab_procedures', 'num_procedures', 
                 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 
                 'diag_1', 'diag_2', 'diag_3', 'metformin', 'insulin', 'change', 
                 'diabetesMed', 'total_visits', 'age_group', 'readmitted']

df = df[selected_cols].copy()

# --- Model Preparation ---
X = df.drop('readmitted', axis=1)
y = df['readmitted']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encoding & Scaling (Fixed FutureWarning)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
cat_features = X_train.select_dtypes(include='object').columns
X_train[cat_features] = encoder.fit_transform(X_train[cat_features])
X_test[cat_features] = encoder.transform(X_test[cat_features])

scaler = StandardScaler()
num_features = X_train.select_dtypes(include=[np.number]).columns
X_train[num_features] = scaler.fit_transform(X_train[num_features].astype(float))
X_test[num_features] = scaler.transform(X_test[num_features].astype(float))

# --- Before SMOTE Plot ---
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train, palette='Set2')
plt.title("Before SMOTE")
plt.show()

# --- Apply SMOTE ---
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- After SMOTE Plot ---
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_res, palette='Set1')
plt.title("After SMOTE (Training Data Only)")
plt.show()

# --- Model Training (XGBoost) ---
# Industry-level: scale_pos_weight used for better working results
model = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, 
                      scale_pos_weight=5, random_state=42)
model.fit(X_train_res.astype(float), y_train_res)

# --- Evaluation ---
y_proba = model.predict_proba(X_test.astype(float))[:, 1]
y_pred = (y_proba >= 0.50).astype(int) # Recall optimized threshold


print("\n Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance Visualization
plt.figure(figsize=(10,6))
sorted_idx = model.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("XGBoost Feature Importance")
plt.title("Key Drivers for Hospital Readmission")
plt.show()

# Save
joblib.dump(model, 'xgboost_diabetes_model.pkl')
joblib.dump(encoder, 'ordinal_encoder.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')
print("Model and preprocessing tools saved.")