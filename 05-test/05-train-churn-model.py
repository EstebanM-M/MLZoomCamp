# %%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# %%
df = pd.read_csv('data-week-3.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes=='str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn=='yes').astype(int)

# %%
df.head()

# %%
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# %%
df.dtypes[df.dtypes=='str'].index

# %%
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = ['gender','seniorcitizen', 'partner', 'dependents', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod']


# %%
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


# %%
from tqdm.auto import tqdm

# %%
C=1.0
n_splits=5

# %%
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores=[]

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train=df_train.churn.values
    y_val = df_val.churn.values

    model, dv = train(df_train,y_train, C=C)
    y_pred = predict(df_val,dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f'%(C, np.mean(scores), np.std(scores)))

# %%
scores

# %%
model,cv = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)
auc

# %% [markdown]
# #### Save the model

# %%
import pickle

# %%
output_file = f'model_C={C}.bin'
output_file

# %%
f_out = open(output_file, 'wb') #write and file is going to be binary wb
pickle.dump((dv,model), f_out)
f_out.close()

# %% [markdown]
# **IMPORTANT: CLOSE FILE**

# %%
#Alternative so it's not necessary to use file.close()
with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)
    #do tuff

#do other stuff

# %% [markdown]
# #### Load the model

# %%
import pickle

# %%
model_file = 'model_C=1.0.bin'

# %%
with open(model_file, 'rb') as f_in:
    (dv,model) = pickle.load(f_in)

# %%
dv,model

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
X = dv.transform([customer])
X

# %%
y = model.predict_proba(X)[:,1]

# %% [markdown]
# #### Making requests 


