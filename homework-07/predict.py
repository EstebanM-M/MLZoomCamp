import pickle
import pandas as pd
import sys

file_name = sys.argv[1]

print('reading file')
df = pd.read_csv(file_name)

print('loading model')
with open('fraud_model.bin', 'rb') as f:
    model, scaler, thr, columns = pickle.load(f)

print('model loaded')
print('Tranforming data for inference')
# Data cleaned: without nulls and duplicates
df_cleaned = df.copy()

df_cleaned['cero_transaction'] = df_cleaned.Amount==0

df_cleaned['day'] = (df_cleaned['Time']/(3600))//24
df_cleaned['hour'] = (df_cleaned['Time']//(3600))%24 

print('data selection and scalation')
X_val = scaler.transform(df_cleaned[columns])
print('data prepared')
print('starting predictions')
y_pred = model.predict_proba(X_val)[:,1]
output = (y_pred>thr)
print('predictions completed, see in output_file.csv')
pd.DataFrame({'probability': y_pred, 'is_fraud': output}).to_csv('output_file.csv', index=False)