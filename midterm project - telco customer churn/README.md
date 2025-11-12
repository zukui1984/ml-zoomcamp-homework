# Customer Churn Prediction - ML Project

A machine learning project that predicts customer churn for a telecom company using XGBoost.

## 📊 What This Project Does

This project helps telecom companies identify customers who are likely to stop using their services. By predicting churn early, companies can take action to keep their customers and reduce losses.

## 📁 Dataset

**Telco Customer Churn Dataset**

- **7,043 customers** with detailed information
- **21 features** including customer demographics, services, and billing
- **Target Variable:** Churn (26.5%) vs No Churn (73.5%)

![Churn Distribution](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/1-pic-churn.jpg)

## 📈 Data Exploration

### Numerical Features Distribution

The dataset includes three key numerical features:

- **Tenure:** How long the customer has been with the company (0-72 months)
- **Monthly Charges:** Amount charged per month ($18-$119)
- **Total Charges:** Total amount paid ($0-$8,685)

![Numerical Features](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/2-pic-distributio%20numerical%20features.jpg)

### Feature Correlations

**Key Finding:** Tenure shows the strongest correlation - customers with longer tenure are less likely to churn.

![Feature Correlation](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/3-pic-feature_correlation.jpg)

## 🤖 Model Comparison

I tested four different machine learning models:

| Model | Validation AUC Score |
|-------|---------------------|
| Logistic Regression | 0.8398 ⭐ |
| Random Forest | 0.8380 |
| Decision Tree | 0.8207 |
| XGBoost | 0.8192 |

![Model Performance](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/4-pic-model_perfomance_comparison.jpg)

**Winner:** XGBoost was selected after hyperparameter tuning.

## 🎯 Most Important Features

Top features that influence churn predictions:

1. Contract type (month-to-month vs long-term)
2. Tenure (how long they've been a customer)
3. Internet service type
4. Payment method

![Feature Importance](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/6-pic-most-important-features.jpg)

## 📊 Final Model Performance

### Performance Metrics

- **AUC Score: 0.8497** - Excellent ability to distinguish churners
- **Accuracy: 77.6%** - Correctly predicts most customers
- **Precision: 56.1%** - When predicting churn, correct 56% of the time
- **Recall: 71.7%** - Catches 72% of customers who actually churn
- **F1-Score: 0.6291** - Balanced measure
- **Optimal Threshold: 0.34**

### Confusion Matrix

![Confusion Matrix](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/5-pic-confusion-matrix-test-set.jpg)

**Results:**

- **938 customers:** Correctly predicted as "Not Churn"
- **198 customers:** Correctly predicted as "Churn"
- **97 customers:** False positives
- **176 customers:** False negatives

The model is especially good at catching customers who will churn (71.7% recall).

## 🛠 Technologies Used

- **Python** - Programming language
- **Pandas & NumPy** - Data manipulation
- **Matplotlib & Seaborn** - Visualization
- **Scikit-learn** - Machine learning
- **XGBoost** - Final model
- **Jupyter Notebook** - Development

## 📦 Installation & Setup

### Prerequisites

