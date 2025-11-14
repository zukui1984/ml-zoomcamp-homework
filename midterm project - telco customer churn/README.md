# Telco Customer Churn Prediction 

A machine learning project to predict customer churn probability for telecom companies, enabling proactive retention strategies.

## Problem Description

Customer churn represents a critical business challenge for telecom providers. When customers cancel their contracts and switch to competitors, companies lose both immediate revenue and long-term customer lifetime value. 

**Business Value:**
- Identify high-risk customers before they leave
- Enable targeted retention campaigns with personalized offers
- Optimize marketing spend by focusing on at-risk segments
- Reduce overall churn rate and improve customer lifetime value

## Dataset

**Source:** [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 

- **7,043 customers** with detailed account information
- **20 features** including demographics, services, and billing
- **Target distribution:** 26.5% Churn, 73.5% No Churn

![Churn Distribution](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/1-pic-churn.jpg)

### Feature Groups

**Account Information:**
- `tenure` - Months with the company
- `Contract` - Month-to-month, one year, or two year
- `PaymentMethod` - Payment type
- `MonthlyCharges` - Current monthly bill
- `TotalCharges` - Total amount paid

**Services:**
- `InternetService` - DSL, Fiber optic, or No
- `PhoneService`, `MultipleLines`
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`
- `TechSupport`, `StreamingTV`, `StreamingMovies`

**Demographics:**
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`

## Exploratory Data Analysis

### Numerical Features

The three key numerical features show distinct patterns:

![Numerical Features Distribution](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/2-pic-distributio%20numerical%20features.jpg)

- **Tenure:** Ranges from 0-72 months - newer customers churn more frequently
- **Monthly Charges:** $18-$419 - higher charges correlate with fiber optic service
- **Total Charges:** $0-$8,685 - directly related to tenure

### Feature Importance for Churn

![Feature Correlation](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/3-pic-feature_correlation.jpg)

**Key Findings:**
- Contract type shows strongest relationship with churn
- Longer tenure significantly reduces churn probability
- Fiber optic internet users churn more than DSL users
- Electronic check payment method associates with higher churn

## Model Development

### Models Evaluated

| Model | Validation AUC |
|-------|---------------|
| Logistic Regression | 0.8398 ⭐ |
| Random Forest | 0.8380 |
| XGBoost | 0.8192 |
| Decision Tree | 0.8207 |

![Model Performance Comparison](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/4-pic-model_perfomance_comparison.jpg)

After hyperparameter tuning, **Logistic Regression** achieved the best validation performance and was selected as the final model. 

### Feature Importance

![Top Features](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/6-pic-most-important-features.jpg)

The most influential features for predictions:
1. Contract type (month-to-month vs long-term)
2. Customer tenure
3. Internet service type
4. Payment method

## Final Model Performance

### Test Set Metrics

- **AUC:** 0.8497 - Strong discrimination between churners and non-churners
- **Accuracy:** 80.6%
- **Precision:** 67.1% - Of predicted churners, 56% actually churn
- **Recall:** 52.9% - Model catches 72% of actual churners

![Confusion Matrix](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/5-pic-confusion-matrix-test-set.jpg)

**Test Set Results:**
- True Negatives: 938 (correctly predicted to stay)
- True Positives: 198 (correctly predicted to churn)
- False Positives: 97 (predicted churn but stayed)
- False Negatives: 176 (predicted stay but churned)

The high recall (71.7%) ensures most at-risk customers are identified for retention campaigns.

## Project Structure
```
midterm-project-telco-customer-churn/
│
├── data/
│ └── telco-churn.csv # Dataset
│
├── img/ # Visualization outputs
│
├── telco-churn.ipynb # EDA and model experimentation
├── train.py # Model training script
├── predict.py # Flask prediction service
├── model.json # Trained XGBoost model
├── dv.pkl # DictVectorizer for preprocessing
├── requirements.txt # Python dependencies
├── Dockerfile # Container configuration
└── README.md # Documentation
```

## 🛠 Technologies Used
- **Python** - Programming language
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Preprocessing and baseline models
- **XGBoost** - Final gradient boosting model
- **Flask** - REST API framework
- **Docker** - Containerization
- **Jupyter** - Interactive analysis

## 📦 Installation & Setup

### 1. Navigate to project
```
cd telco-churn
```
### 2. Setup environment
```
pipenv install
pipenv install --dev
pip install pandas numpy scikit-learn matplotlib seaborn xgboost flask gunicorn (VS Code)
```
### 3. Open in VS Code
code .

### 4. Run notebook in VS Code
  - Open notebook.ipynb
  - Select Pipenv kernel
  - Run all cells (Ctrl+Shift+Enter)
  - Verify model.json and dv.pkl are created

### 5. Test training script
```
pipenv run python train.py
```
<img width="945" height="565" alt="image" src="https://github.com/user-attachments/assets/91300926-0910-4d08-a4d2-a0eaaee382cf" />

### 6. Start prediction service
```
pipenv run python predict.py
```
#### Keep this terminal open!
<img width="945" height="440" alt="image" src="https://github.com/user-attachments/assets/fc365b3f-2048-4d30-9b29-67dede491f9e" />

### 7. Open new terminal (Ctrl+Shift+`)
#### Test prediction
```
curl -X POST http://localhost:9696/predict
-H "Content-Type: application/json"
-d '{
"gender": "female",
"seniorcitizen": 0,
"partner": "yes",
"dependents": "no",
"tenure": 12,
"phoneservice": "yes",
"internetservice": "fiber optic",
"contract": "month-to-month",
"monthlycharges": 85.0,
"totalcharges": 1020.0,
"multiplelines": "no",
"onlinesecurity": "no",
"onlinebackup": "no",
"deviceprotection": "no",
"techsupport": "no",
"streamingtv": "yes",
"streamingmovies": "yes",
"paperlessbilling": "yes",
"paymentmethod": "electronic check"
}'
```
### 8. Build Docker
docker build -t churn-prediction .
<img width="945" height="464" alt="image" src="https://github.com/user-attachments/assets/764aa4b3-e4fa-4d70-adce-b1e28508231d" />

### 9. Run Docker container
```Docker
docker run -p 9696:9696 churn-prediction
```
### 10. Test Docker (new terminal)
```localhost
http://localhost:9696/
```
<img width="373" height="296" alt="image" src="https://github.com/user-attachments/assets/e978e330-a7b2-4165-919a-156ce535900a" />

```localhost
http://localhost:9696/predict
```

<img width="574" height="263" alt="image" src="https://github.com/user-attachments/assets/7ed812ce-237d-48b0-bc88-99efd130ab38" />

```localhost
http://localhost:9696/health
```

<img width="103" height="65" alt="image" src="https://github.com/user-attachments/assets/30cddc3f-668c-4d1e-868e-3ebda9694f8b" />

## 💡 Key Insights

Analysis of the dataset and model predictions reveals:

1. **Month-to-month contracts** have significantly higher churn rates compared to annual contracts
2. **Customer tenure** is the strongest protective factor - established customers rarely leave
3. **Fiber optic internet** subscribers churn more frequently than DSL users, possibly due to higher prices
4. **Electronic check** payment correlates with elevated churn risk compared to automatic payment methods

## 🎯 Business Recommendations

Based on model insights, telecom companies should:

- Offer contract upgrade incentives to month-to-month customers
- Implement early engagement programs for customers in their first 12 months
- Review pricing strategy for fiber optic services
- Encourage migration from electronic check to automatic payment methods
- Focus retention campaigns on customers with low tenure, month-to-month contracts, and fiber optic service
