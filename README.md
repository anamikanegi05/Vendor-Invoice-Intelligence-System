# Vendor Invoice Intelligence System
**Freight Cost Prediction & Invoice Risk Flagging**

## 📌 Table of Contents
- <a href="#project-overview">Project Overview</a>
- <a href="#business-objectives">Business Objectives</a>
- <a href="#data-sources">Data Sources</a>
- <a href="#eda">Exploratory Data Analysis</a>
- <a href="#models-used">Models Used</a>
- <a href="#metrics">Evaluation Metrics</a>
- <a href="#application">Application</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#author--contact">Author & Contact</a>

---

<h2><a class="anchor" id="project-overview"></a>📌 Project Overview</h2>

This project implements an **end-to-end machine learning system** designed to support finance teams by:

1. **Predicting expected freight cost** for vendor invoices.
2. **Flagging high-risk invoices** that require manual review due to abnormal cost, freight, or operational patterns.

---

<h2><a class="anchor" id="business-objectives"></a> Business Objectives</h2>

### 1. Freight Cost Prediction (Regression)

The objective is to predict the expected freight cost for vendor invoices based on invoice quantity, invoice dollars, and other relevant features.

**Models Evaluated:**
- Linear Regression (Baseline Model)
- Random Forest Regressor (Final Model)

### 2. Invoice Risk Flagging (Classification)

The objective is to identify potentially abnormal vendor invoices using historical invoice patterns and financial features.

**Models Evaluated:**
- Logistic Regression (Baseline Model)
- Decision Tree Classifier
- Random Forest Classifier

---

## Data Sources

The project uses a relational SQLite database (`inventory.db`) containing vendor invoice and purchase-related information.

SQL queries are used to extract, aggregate, and transform raw data into invoice-level features for machine learning models.

### Features Used

- Invoice Quantity
- Invoice Dollars
- Freight
- Days PO to Invoice
- Days to Pay
- Total Brands
- Total Item Quantity
- Total Item Dollars
- Average Receiving Delay

---

## Exploratory Data Analysis (EDA)

EDA is performed to understand invoice patterns and identify factors influencing freight cost and invoice risk.

Key analysis includes:

- Freight cost distribution
- Relationship between quantity and freight
- Invoice financial exposure
- Comparison of flagged and normal invoices
- Statistical analysis using t-tests

---

## Models Used

### Freight Cost Prediction

- Linear Regression
- Random Forest Regressor

### Invoice Risk Flagging

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

Feature engineering and preprocessing are performed before model training.

---

## Evaluation Metrics

### Freight Cost Prediction (Regression)

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Invoice Risk Flagging (Classification)

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## End-to-End Application

The project includes a Streamlit-based Vendor Invoice Intelligence Portal that integrates the trained machine learning models.

### Application Features

- Interactive invoice input
- Freight cost prediction
- Invoice risk flag prediction
- Invoice quantity and financial details
- Tabular output for invoice analysis
- Real-time model inference

The application helps users analyze vendor invoices and supports data-driven financial decision-making.

---

## Project Structure

```text
Vendor-Invoice-Intelligence-System/
│
├── inventory-invoice-analytics/
│   └── data/
│       └── inventory.db
│
├── freight_cost_prediction/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── invoice_flagging/
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   └── train.py
│
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
├── models/
│   ├── predict_freight_model.pkl
│   ├── scaler.pkl
│   └── predict_flag_invoice.pkl
│
├── notebooks/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/ayushimishra28/Vendor-Invoice-Intelligence-System.git
```

### 2. Navigate to the Project Directory

```bash
cd Vendor-Invoice-Intelligence-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Models

```bash
python freight_cost_prediction/train.py
```

```bash
python invoice_flagging/train.py
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SQLite
- SQL
- Streamlit
- Plotly
- Joblib

---

## Author & Contact

**Anamika Negi**

AI/ML Engineer | Computer Science Engineering Student

GitHub: [Anamika Negi](https://github.com/anamikanegi05)
