# Telco Customer Churn Prediction - ML Project

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

### 1. Navigate to project
cd telco-churn

### 2. Setup environment
pipenv install
pipenv install --dev
pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask gunicorn (VS Code)

### 3. Open in VS Code
code .

### 4. Run notebook in VS Code
  - Open notebook.ipynb
  - Select Pipenv kernel
  - Run all cells (Ctrl+Shift+Enter)
  - Verify model.json and dv.pkl are created

### 5. Test training script
pipenv run python train.py
<img width="945" height="565" alt="image" src="https://github.com/user-attachments/assets/91300926-0910-4d08-a4d2-a0eaaee382cf" />

### 6. Start prediction service
pipenv run python predict.py
#### Keep this terminal open!
<img width="945" height="440" alt="image" src="https://github.com/user-attachments/assets/fc365b3f-2048-4d30-9b29-67dede491f9e" />

### 7. Open new terminal (Ctrl+Shift+`)
#### Test prediction
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d "{\"gender\":\"female\",\"seniorcitizen\":0,\"partner\":\"yes\",\"dependents\":\"no\",\"tenure\":12,\"phoneservice\":\"yes\",\"internetservice\":\"fiber optic\",\"contract\":\"month-to-month\",\"monthlycharges\":85.0,\"totalcharges\":1020.0,\"multiplelines\":\"no\",\"onlinesecurity\":\"no\",\"onlinebackup\":\"no\",\"deviceprotection\":\"no\",\"techsupport\":\"no\",\"streamingtv\":\"yes\",\"streamingmovies\":\"yes\",\"paperlessbilling\":\"yes\",\"paymentmethod\":\"electronic check\"}"

### 8. Build Docker
docker build -t churn-prediction .
<img width="945" height="464" alt="image" src="https://github.com/user-attachments/assets/764aa4b3-e4fa-4d70-adce-b1e28508231d" />

### 9. Run Docker container
docker run -p 9696:9696 churn-prediction

### 10. Test Docker (new terminal)
http://localhost:9696/

<img width="373" height="296" alt="image" src="https://github.com/user-attachments/assets/e978e330-a7b2-4165-919a-156ce535900a" />

http://localhost:9696/predict

<img width="574" height="263" alt="image" src="https://github.com/user-attachments/assets/7ed812ce-237d-48b0-bc88-99efd130ab38" />

http://localhost:9696/health

<img width="103" height="65" alt="image" src="https://github.com/user-attachments/assets/30cddc3f-668c-4d1e-868e-3ebda9694f8b" />

## 💡 Key Insights

1. **Month-to-month contracts** have 42.7% churn rate - highest risk
2. **Longer tenure** = lower churn risk
3. **Fiber optic users** churn more than DSL users
4. **Electronic check** payment correlates with higher churn

## 📝 Model Optimization

- Feature engineering (6 new features created)
- Hyperparameter tuning (depth and learning rate)
- Multiple model comparison
