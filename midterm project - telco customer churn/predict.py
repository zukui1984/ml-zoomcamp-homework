import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

# Load model and encoder
print("Loading model")
model = xgb.Booster()
model.load_model('model.json')

print("Loading DictVectorizer")
with open('dv.pkl', 'rb') as f:
    dv = pickle.load(f)

app = Flask('churn-prediction')

def prepare_customer(customer):
    # Add engineered features
    tenure = customer.get('tenure', 0)
    if tenure <= 12:
        customer['tenure_group'] = '0-1yr'
    elif tenure <= 24:
        customer['tenure_group'] = '1-2yr'
    elif tenure <= 48:
        customer['tenure_group'] = '2-4yr'
    else:
        customer['tenure_group'] = '4+yr'
    
    customer['avg_monthly_charge'] = customer.get('totalcharges', 0) / (tenure + 1)
    customer['has_internet'] = int(customer.get('internetservice', 'No') != 'No')
    customer['has_phone'] = int(customer.get('phoneservice', 'No') == 'Yes')
    
    services = ['onlinesecurity', 'onlinebackup', 'deviceprotection', 
                'techsupport', 'streamingtv', 'streamingmovies']
    customer['total_services'] = sum(1 for s in services 
                                    if customer.get(s, 'No') == 'Yes')
    
    return customer

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()
    
    customer = prepare_customer(customer)
    
    # Encode
    X = dv.transform([customer])
    dmatrix = xgb.DMatrix(X)
    
    # Predict
    churn_prob = float(model.predict(dmatrix)[0])
    churn = churn_prob >= 0.5
    
    result = {
        'churn_probability': round(churn_prob, 3),
        'churn': bool(churn),
        'recommendation': 'Offer retention incentive' if churn else 'Regular service'
    }
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'xgboost'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000)) 
    
    print(f"API running at: http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  - GET  /         : API information")
    print(f"  - POST /predict  : Predict churn")
    print(f"  - GET  /health   : Health check")
    
    app.run(debug=True, host='0.0.0.0', port=port)

    
