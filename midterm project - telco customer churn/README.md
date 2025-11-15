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
- **Precision:** 67.1% - Of predicted churners, 67% actually churn
- **Recall:** 52.9% - Model catches 53% of actual churners

![Confusion Matrix](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/5-pic-confusion-matrix-test-set.jpg)

**Test Set Results:**
- True Negatives: 938 (correctly predicted to stay)
- True Positives: 198 (correctly predicted to churn)
- False Positives: 97 (predicted churn but stayed)
- False Negatives: 176 (predicted stay but churned)

The model achieves good precision (67.1%), meaning when it predicts churn, it's correct two-thirds of the time, making retention campaigns cost-effective.

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
- **XGBoost** - Gradient boosting 
- **Flask** - REST API framework
- **Docker** - Containerization
- **Jupyter Notebook** - Interactive analysis

## 📦 Installation & Setup

### 1. Clone Repository
```
git clone https://github.com/your-username/telco-customer-churn.git
cd telco-customer-churn
```
### 2. Install dependecies
```
pip install -r requirements.txt
jupyter lab
```
This installs all required packages:
 - pandas, numpy, scikit-learn, xgboost, flask, gunicorn
 - matplotlib, seaborn, jupyter notebook

### 3. Run Jupyter Notebook: EDA + Model
```
jupyter lab
```

### 4. Test training script - [CODE](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/predict.py)
```
python train.py
```
<img width="400" height="200" alt="image" src="https://github.com/user-attachments/assets/91300926-0910-4d08-a4d2-a0eaaee382cf" />

### 5. Start prediction service - [CODE](https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/predict.py)
```
python predict.py
```
#### Test prediction (locally PORT 5000) - Keep this terminal open!
<img width="400" height="200" alt="image" src="https://github.com/zukui1984/ml-zoomcamp-homework/blob/master/midterm%20project%20-%20telco%20customer%20churn/img/7-pic-flask-localhost.jpg" />

Click http://localhost:5000/predict

```
{
  "endpoints": {
    "/": "Service information (GET)",
    "/health": "Health check (GET)",
    "/predict": "Predict customer churn (GET for test, POST for production)"
  },
  "production_example": {
    "body": {
      "deviceprotection": "No",
      "internetservice": "Fiber optic",
      "monthlycharges": 70.0,
      "onlinebackup": "Yes",
      "onlinesecurity": "No",
      "phoneservice": "Yes",
      "streamingmovies": "No",
      "streamingtv": "Yes",
      "techsupport": "No",
      "tenure": 12,
      "totalcharges": 840.0
    },
    "method": "POST",
    "url": "http://localhost:5000/predict"
  },
  "service": "Churn Prediction API",
  "version": "1.0"
}
```
### 7. Build Docker
docker build -t churn-prediction .

<img width="643" height="164" alt="image" src="https://github.com/user-attachments/assets/8673406d-cc96-4a75-8009-76681838febb" />

### 8. Run Docker container 
```Docker
docker run -p 9696:9696 churn-prediction
```
### 9. Test Docker (new terminal PORT 9696)
```localhost
http://localhost:9696/predict
```
<img width="270" height="352" alt="image" src="https://github.com/user-attachments/assets/22b306cc-492f-4e06-879f-0c8ff32a0787" />

```localhost
http://localhost:9696/health
```

<img width="230" height="149" alt="image" src="https://github.com/user-attachments/assets/8feb96cc-7a21-4257-9d3e-e3a0f3c1c2c3" />

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
