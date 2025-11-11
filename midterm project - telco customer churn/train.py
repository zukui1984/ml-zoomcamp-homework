import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def load_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    return df

def engineer_features(df):
    df['tenure_group'] = pd.cut(df['tenure'], 
                                bins=[0, 12, 24, 48, 72], 
                                labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
    df['tenure_group'] = df['tenure_group'].astype(str)
    
    df['avg_monthly_charge'] = df['totalcharges'] / (df['tenure'] + 1)
    df['has_internet'] = (df['internetservice'] != 'No').astype(int)
    df['has_phone'] = (df['phoneservice'] == 'Yes').astype(int)
    
    services = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 
                'techsupport', 'streamingtv', 'streamingmovies']
    df['total_services'] = df[services].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    return df

def prepare_data(df):
    df_clean = df.drop('customerid', axis=1)
    X = df_clean.drop('churn', axis=1)
    y = df_clean['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

def encode_features(X_train, X_test):
    train_dicts = X_train.fillna('Missing').to_dict(orient='records')
    test_dicts = X_test.fillna('Missing').to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train_enc = dv.fit_transform(train_dicts)
    X_test_enc = dv.transform(test_dicts)
    
    return X_train_enc, X_test_enc, dv

def train_model(X_train, y_train):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.3,
        'seed': 42,
        'eval_metric': 'auc'
    }
    
    model = xgb.train(params, dtrain, num_boost_round=50)
    return model

def main():
    # Configuration
    DATA_FILE = 'telco-customer-churn.csv'
    MODEL_FILE = 'model.json'
    DV_FILE = 'dv.pkl'
    
    print()
    print("TRAINING START")
        
    # Step 1: Load data
    print("\n1. Loading data...")
    df = load_clean_data(DATA_FILE)
    print(f"   Loaded {len(df):,} records")
    
    # Step 2: Feature engineering
    print("\n2. Engineering features...")
    df = engineer_features(df)
    
    # Step 3: Prepare data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Step 4: Encode features
    print("\n4. Encoding features...")
    X_train_enc, X_test_enc, dv = encode_features(X_train, X_test)
    print(f"   Encoded features: {X_train_enc.shape[1]}")
    
    # Step 5: Train model
    print("\n5. Training model...")
    model = train_model(X_train_enc, y_train)
    
    # Step 6: Evaluate
    print("\n6. Evaluating...")
    dtest = xgb.DMatrix(X_test_enc)
    y_pred = model.predict(dtest)
    auc = roc_auc_score(y_test, y_pred)
    print(f"   Test AUC: {auc:.4f}")
    
    # Step 7: Save
    print("\n7. Saving model...")
    model.save_model(MODEL_FILE)
    with open(DV_FILE, 'wb') as f:
        pickle.dump(dv, f)
    print(f"Saved {MODEL_FILE}")
    print(f"Saved {DV_FILE}")
    
    print()
    print("TRAINING COMPLETE")


if __name__ == '__main__':
    main()