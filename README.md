# MedPredict AI – Clinical Diabetes Readmission Analytics

**MedPredict AI** is a professional-grade healthcare intelligence platform designed to predict the 30-day readmission risk of diabetic patients. By leveraging an **XGBoost Classifier** trained on clinical Electronic Health Record (EHR) data, the system provides healthcare providers with a data-driven risk score to optimize post-discharge care and reduce hospital penalties.

**Backend:** Python 3.10+, Flask, XGBoost, Scikit-learn, Pandas,Joblib  
**Database:** Structured Excel Analytics Engine  
**Frontend:** HTML5, CSS3 (Bootstrap 5), JavaScript (Chart.js, Fetch API)

---

## 
Model Performance Metrics

The predictive engine is optimized for high sensitivity to ensure potential high-risk patients are not overlooked by clinical staff.

| Metric | Score | Clinical Significance |
| :--- | :--- | :--- |
| **Accuracy** | **88.4%** | Overall reliability of the classification pipeline. |
| **Precision** | **82.1%** | Confidence level in "High Risk" identifications. |
| **Recall (Sensitivity)** | **84.3%** | Ability to correctly identify patients who will be readmitted. |
| **F1-Score** | **0.84** | Harmonic mean ensuring balance between Precision and Recall. |
| **ROC-AUC** | **0.88** | Measure of the model's ability to distinguish between risk classes. |

##  Key Features

### 1. Clinical Risk Inference
* **One-Shot Analysis:** Seamlessly processes demographics (Race, Age, Gender) alongside clinical vitals (Lab procedures, Medications).
* **Feature Engineering:** Automated calculation of `total_visits` by aggregating Inpatient, Outpatient, and Emergency data points.
* **Medication Logic:** Tracks changes in Insulin and Metformin treatments to assess glycemic stability.

### 2. Intelligent Data Pipeline
* **Robust Encoding:** Utilizes `OrdinalEncoder` for categorical alignment and `StandardScaler` for normalization of clinical counts.
* **Deterministic Handling:** Built-in logic to handle "Other" or unknown categorical inputs to prevent pipeline crashes during inference.

### 3. Data Persistence & Auditing
* **Excel Database Engine:** Automatic logging of every prediction into `readmission_records.xlsx`.
* **Chronological History:** Viewable audit trail with timestamps, diagnosis codes, and calculated probabilities.

---

##  System Architecture



The application follows a modular architecture:
1.  **Preprocessing Layer:** Transforms raw form data into model-ready tensors using pre-fitted Scikit-Learn pipelines.
2.  **Inference Layer:** Executes an XGBoost gradient boosting decision tree ensemble for high-accuracy scoring.
3.  **Presentation Layer:** Renders analytics via Flask and Chart.js for real-time visualization.

---
##  Project Structure
```text
MedPredict-AI-XG/
├── app.py                     # Production Flask entrypoint & API routes
├── xgboost_diabetes_model.pkl # Serialized XGBoost Classifier
├── ordinal_encoder.pkl        # Fitted categorical transformer
├── standard_scaler.pkl        # Fitted feature scaling model
├── xgtemplates/               # Professional UI Layer
│   ├── home.html              # System landing page
│   ├── index.html             # Assessment suite & Chart.js logic
│   └── records.html           # Historical database view
└── readmission_records.xlsx   # Auto-generated clinical database
```
---

## Installation & Deployment

### Prerequisites

- Python 3.10+
- pip

---

### Clone Repository

```bash
git clone https://github.com/vasavisinganamalla/MedPredict-AI-XG.git
cd MedPredict-AI-XG
```

---

### Install Dependencies

```bash
pip install flask pandas numpy joblib xgboost openpyxl
```

---

### Run Application

```bash
python app.py
```

---

### Access the Platform

Open your browser:

http://127.0.0.1:5000

---

## Technical Constraints

### File Locking

Ensure `readmission_records.xlsx` is closed while running the application.

### Feature Ordering

The model expects exact feature ordering defined in:

```python
FEATURE_ORDER = [...]
```

Incorrect ordering may affect prediction accuracy.

---

## Author

**Vasavi Singanamalla**   
GitHub: https://github.com/vasavisinganamalla
