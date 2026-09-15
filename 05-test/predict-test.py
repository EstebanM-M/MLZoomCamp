import requests

host = 'churn-serving-env.eba-kebqmkk2.us-east-1.elasticbeanstalk.com'
url = f'http://{host}/predict'

# %%
customer={'gender': 'male',
 'partner': 'yes',
 'dependents': 'yes',
 'phoneservice': 'yes',
 'multiplelines': 'no',
 'internetservice': 'fiber_optic',
 'onlinesecurity': 'no',
 'onlinebackup': 'yes',
 'deviceprotection': 'no',
 'techsupport': 'no',
 'streamingtv': 'yes',
 'streamingmovies': 'yes',
 'contract': 'month-to-month',
 'paperlessbilling': 'yes',
 'paymentmethod': 'mailed_check',
 'seniorcitizen': 1,
 'tenure': 32,
 'monthlycharges': 93.95,
 'totalcharges': 2861.45}

# %%
response = requests.post(url, json=customer).json()
response

# %%
print(f'churn:{response['churn']}, prob:{response['churn_probabilty']}')



