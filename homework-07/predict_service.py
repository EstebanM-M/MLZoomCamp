import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask('fraud_prediction')

with open('fraud_model.bin', 'rb') as f:
    model,scaler,thr,columns = pickle.load(f)

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json()
    print('Preparing data for prediction')
    data['cero_transaction'] = (data['Amount']==0)
    data['day'] = (data['Time']/(3600))//24
    data['hour'] = (data['Time']//(3600))%24

    df = pd.DataFrame([data])[columns]
    print('data selection and scalation')
    X_val = scaler.transform(df)   

    print('Starting prediction')
    y_pred = model.predict_proba(X_val)[0,1]
    output = (y_pred>thr)
    print('prediction finalized, sending response')
    return jsonify({'probability': float(y_pred), 'is_fraud': bool(output)})

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 9696)