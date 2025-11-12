# Telco Customer Churn Prediction

Predict customer churn for telecom companies using machine learning to enable proactive retention strategies.

## Problem Description

Customer churn (when customers stop using a service) costs the telecom industry billions annually. This project builds a machine learning model to predict which customers are likely to churn, allowing companies to:

- Identify at-risk customers before they leave
- Offer targeted retention incentives
- Reduce customer acquisition costs
- Improve customer lifetime value

## Installation & Setup

### Prerequisites
- Python 3.9+: sklearn, pandas, numpy, seaborn
- pipenv (or pip)
- Docker (optional, for containerization)

## Dataset

- **Source**: IBM Telco Customer Churn Dataset from Kaggle - [LINK](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: 7,043 customers
- **Features**: 21 features including:
  - Demographics: gender, senior citizen status, partner, dependents
  - Services: phone, internet, streaming, security
  - Account: contract type, payment method, charges
- **Target**: Binary classification (Churn: Yes/No)

## Project Structure
1. **Install dependencies (VS Code)**
   - pip install pipenv
   - pipenv install
     <img width="945" height="146" alt="image" src="https://github.com/user-attachments/assets/d2d1e616-b296-4c2c-b693-90a4690efd50" />
   - pipenv install --dev
     <img width="945" height="146" alt="image" src="https://github.com/user-attachments/assets/9a457aa1-1a93-4a3e-85d9-8262ade8f3d8" />
2. **Create & work with jupyter notebook** & run pipenv run jupyter lab 
   *Content on notebook* 
    - Data cleaning and preparation
    - Extensive EDA with visualizations
    - Feature engineering
    - Model comparison (Logistic Regression, Decision Tree, Random Forest, XGBoost)
    - Hyperparameter tuning
    - Final model evaluation
3. Train Model
   pipenv run python train.py
   <img width="945" height="565" alt="image" src="https://github.com/user-attachments/assets/51d8949c-d912-4a2a-8628-41899352dd5b" />

5. Run web service (locally)
   pipenv run python predict.py
   
  <img width="435" height="238" alt="image" src="https://github.com/user-attachments/assets/a2d128ce-d447-425d-acbb-60f49681a9f8" />

  after that and test "run localhost:9696"
  
  <img width="398" height="284" alt="image" src="https://github.com/user-attachments/assets/b4d9934d-c6b2-4bd0-bee7-e772c58e5dd0" />

  run localhost:9696/predict
  
  <img width="582" height="266" alt="image" src="https://github.com/user-attachments/assets/a9613a9f-a7f8-4dad-a298-7e9ad100c350" />

  run localhost:9696/health
  <img width="110" height="72" alt="image" src="https://github.com/user-attachments/assets/987108e9-45a4-43d8-a15c-2d635ed40a95" />

6. Docker:
   a. Build image: docker build -t churn-prediction .
   <img width="945" height="464" alt="image" src="https://github.com/user-attachments/assets/dba84564-b1e5-488c-95a1-2174129dd480" />

   b. Run container: docker run -p 9696:9696 churn-prediction
   
8. 

