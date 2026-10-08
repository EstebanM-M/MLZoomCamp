import kagglehub
from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import pickle

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
file_path = list(Path(path).iterdir())
print('reading file')
df = pd.read_csv(file_path[0])

print('Cleaning file')
# Data cleaned: without nulls and duplicates
df_cleaned = df.copy()
df_cleaned = df_cleaned.drop_duplicates()

df_cleaned['cero_transaction'] = df_cleaned.Amount==0

df_cleaned['day'] = (df_cleaned['Time']/(3600))//24
df_cleaned['hour'] = (df_cleaned['Time']//(3600))%24 

y_cleaned = df_cleaned['Class']
del df_cleaned['Class']

print('preparing information for training')
X_full_train, X_test, y_full_train, y_test = train_test_split(df_cleaned, y_cleaned, test_size = 0.2, shuffle = True, stratify=y_cleaned, random_state=42)

xgbc = XGBClassifier(eta=0.03, max_depth=6, min_child_weight=1, random_state = 42)
scaler = StandardScaler()

X_train_ = scaler.fit_transform(X_full_train)
X_test_ = scaler.transform(X_test)

print('Starting the traning')
xgbc.fit(X_train_, y_full_train)

y_pred = xgbc.predict_proba(X_test_)[:,1]
pr_c = precision_recall_curve(y_test,y_pred)
idx = min(np.argmin(abs(pr_c[1]-0.8)), len(pr_c[2])-1)
print('The average precision score is:',average_precision_score(y_test, y_pred))
print('Training  finalized - proceeding to save the model')
print('The threshold for a recall of 0.8 is:', pr_c[2][idx])
#model, scaler and threshold save:
with open('fraud_model.bin', 'wb') as f:
    pickle.dump((xgbc, scaler, pr_c[2][idx], X_full_train.columns.tolist()),f)

print('File is saved')