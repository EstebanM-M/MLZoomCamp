#Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import pickle

# Data preparation

df = pd.read_csv('data-week-3.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes=='str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn=='yes').astype(int)

print(df.churn.unique())

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


df.dtypes[df.dtypes=='str'].index

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = ['gender','seniorcitizen', 'partner', 'dependents', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod']


# Training

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical+numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    model = LogisticRegression(max_iter=1000, C=C)
    model.fit(X_train, y_train)

    return model, dv

def predict(df, dv, model):
    dicts = df[categorical+numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]

    return y_pred  

# Parameters
C=1.0
n_splits=5


#Validation
print(f'doing validation with C={C}')
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores=[]

fold=0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train=df_train.churn.values
    y_val = df_val.churn.values

    model, dv = train(df_train,y_train, C=C)
    y_pred = predict(df_val,dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print('auc on fold {fold} is {auc}')
    fold+=1

print('validation results')
print('C=%s %.3f +- %.3f'%(C, np.mean(scores), np.std(scores)))

print('training the final model')
model,cv = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)

print('auc {auc}')
# #### Save the model
print("Saving the model")
output_file = f'model_C={C}.bin'
output_file

with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)

print(f'the model is saved to {output_file}')
# **IMPORTANT: CLOSE FILE**