# Credit Card Fraud Detection
 
A web service that takes a credit card transaction and returns the probability that it is
fraudulent, along with a binary decision based on a threshold calibrated to catch 80% of
frauds. The model is an XGBoost classifier trained on the Université Libre de Bruxelles
dataset of European card transactions, and the service ships as a Docker container.
 
## Problem description
 
Credit card fraud is the operational channel for several crimes: identity theft, money
laundering, use of stolen cards. Detecting it automatically is necessary because transaction
volume makes manual review impossible, and because fraud methods change faster than fixed
rules can be written.
 
The problem has three characteristics that shape every technical decision in this project:
 
**Extreme class imbalance.** In the dataset, 492 out of 284,807 transactions are fraud:
0.17%. A model that labelled everything "normal" would be right 99.83% of the time and
would be useless. That is why accuracy is not used here, and the primary metric is average
precision (the area under the precision-recall curve), which measures performance on the
minority class.
 
**Asymmetric costs.** A false negative is a fraud that goes through: money lost and a
customer harmed. A false positive is a legitimate transaction flagged: it costs an analyst's
time to review. The second is far cheaper, so the model is calibrated to favour recall over
precision.
 
**Anonymised features.** The variables are PCA components of the original data, which is not
published for confidentiality reasons. This limits what is possible: there is no
interpretability, and no way to build new features from domain knowledge.
 
The model does not decide on its own: it produces a probability, and the threshold that turns
that probability into an action is a business decision. At this project's threshold, the
service catches 76 of every 95 frauds and raises 21 false alarms — those 21 are transactions
someone has to review. Lowering the threshold catches more fraud and increases that workload;
raising it reduces the workload and lets more fraud through. The model provides the curve;
the business picks the point on it.
 
## Data
 
Credit Card Fraud Detection, collected by the Machine Learning Group at the Université Libre
de Bruxelles and published on Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
 
284,807 card transactions made by European cardholders over two days in September 2013, of
which 492 are frauds.
 
The columns are:
 
- **`V1`–`V28`** — principal components obtained by PCA on the original variables. The
  untransformed data is not published for confidentiality reasons, so what each component
  represents is unknown.
- **`Time`** — seconds elapsed since the first transaction in the record.
- **`Amount`** — the transaction amount.
- **`Class`** — the target variable: 1 for fraud, 0 for a normal transaction.
### Getting the dataset
 
The file is around 144MB and exceeds GitHub's limit, so it is not in the repository.
`train.py` downloads it automatically using kagglehub, which caches it locally and does not
download it again on later runs:
 
```python
import kagglehub
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
```
 
This requires a Kaggle account and an API token stored at `~/.kaggle/kaggle.json`, which is
generated in Kaggle under Settings → API → Create New Token. Without the token the download
fails with an authentication error.
 
Alternatively, the CSV can be downloaded manually from the link above.
 
## Project structure
 
```
.
├── notebook.ipynb         # EDA, model comparison, tuning and final evaluation
├── train.py               # trains the final model and exports the artifact
├── predict_service.py     # Flask service exposing /predict
├── fraud_model.bin        # model, scaler, threshold and column order
├── sample_fraud.json      # example transaction for testing the endpoint
├── Dockerfile
├── .dockerignore
├── pyproject.toml         # dependencies
├── uv.lock                # pinned versions
└── README.md
```
 
## Results
 
The primary metric is **average precision** (the area under the precision-recall curve). All
values in the table are averages from 5-fold stratified cross-validation on the training set.
 
| Model | AP (default parameters) | AP (tuned) |
|---|---|---|
| Baseline (random prediction) | 0.0017 | — |
| Decision Tree | 0.575 | — |
| Logistic Regression | 0.757 | — |
| XGBoost | 0.803 | 0.840 |
| Random Forest | 0.837 | 0.841 |
 
The Decision Tree fell well behind due to overfitting: with no depth limit it reaches pure
leaves and does not generalise. Logistic Regression falls short because the relationship
between the features and the target is not linear.
 
Random Forest and XGBoost ended up statistically indistinguishable — the 0.001 difference
between the tuned models is far below the deviation across folds (±0.024). With performance
tied, the deciding criterion became cost: XGBoost's final configuration uses a lower maximum
depth, which makes training cheaper. **The final model is XGBoost.**
 
### What hyperparameter tuning contributed
 
Of nine configurations explored across the two models, only one produced a measurable
improvement: lowering `eta` in XGBoost (0.803 → 0.840, roughly 1.5 deviations). Every other
change fell within the noise across folds.
 
The most interesting result was a negative one: `scale_pos_weight`, the XGBoost parameter
designed specifically for imbalanced classes, **made the model clearly worse** (−0.113, about
3.4 deviations). It is the only unambiguously significant effect in the whole tuning exercise,
and it runs against what intuition would suggest for this dataset.
 
### Final evaluation
 
The tuned model was retrained on the full training set and evaluated once on the test set,
which was not used at any earlier stage:
 
- **Average precision on test: 0.808** — roughly 484 times the random baseline (0.0017).
This is 0.031 lower than the cross-validation estimate (0.840), which is expected: picking
the configuration that scored best across folds introduces an optimistic bias, because part
of that "best" is favourable noise.
 
### Operating point
 
From the precision-recall curve, the threshold chosen is the one reaching a recall close to
0.8: **0.0603**, far below the default of 0.5. At that threshold, across the 56,746 test
transactions:
 
|  | Predicted normal | Predicted fraud |
|---|---|---|
| **Normal** | 56,630 | 21 |
| **Fraud** | 19 | 76 |
 
In other words: of 95 real frauds the model catches 76 and misses 19, at the cost of 21 false
alarms. Precision 0.776, recall 0.800.
 
## How to run
 
### Setup
 
```bash
uv sync
```
 
Installs the exact versions pinned in `uv.lock`.
 
### Training (optional)
 
The trained artifact (`fraud_model.bin`) is in the repository, so training is not required to
run the service. To regenerate it:
 
```bash
uv run python train.py
```
 
The script downloads the dataset, trains the model and prints the metrics obtained:
 
```
The average precision score is: 0.8084548302680767
The threshold for a recall of 0.8 is: 0.060319927
```
 
### Running the service locally
 
```bash
uv run gunicorn --bind=0.0.0.0:9696 predict_service:app
```
 
Gunicorn does not run on Windows (it depends on `fcntl`). On Windows, use Docker or WSL.
 
### Running with Docker
 
```bash
docker build -t fraud-service .
docker run -it --rm -p 9696:9696 fraud-service
```
 
### Testing the endpoint
 
With the service running, from another terminal:
 
```python
import json, requests
 
with open('sample_fraud.json') as f:
    sample = json.load(f)
 
r = requests.post('http://localhost:9696/predict', json=sample)
print(r.json())
```
 
`sample_fraud.json` contains a real fraudulent transaction from the test set. The expected
response is:
 
```python
{'is_fraud': True, 'probability': 0.9301382303237915}
```
 
## Deployment

The service was deployed to AWS Elastic Beanstalk as a Docker container, in a single-instance
environment.

**Deployment demo:** https://github.com/user-attachments/assets/dbc2e392-927c-432c-9684-1c2fc1ce3d30

The video shows the running environment in the AWS console, a request sent to its public URL
from a notebook, and the response matching the probability computed locally for the same
transaction — confirming the deployed service reproduces the model exactly.

The environment was terminated after recording to avoid running costs, so the public URL is
no longer active. The service can be reproduced locally with the commands in the previous
section.

## Limitations
 
**No interpretability.** The features are PCA components, so the model cannot point to what
makes a transaction suspicious. In a real system this is an operational problem: an analyst
receiving an alert needs to know what to check, and a cardholder whose purchase is blocked is
entitled to an explanation.
 
**The missed frauds will not be fixed by more tuning.** The 19 cases the model lets through
most likely require information this dataset does not contain: cardholder history, device
used, transaction geolocation. More tuning over the same 30 variables will not recover them.
 
**The threshold was chosen on the test set.** The precision of 0.776 and the recall of 0.800
are measured on the same data used to pick the threshold, so they are an optimistic ceiling
rather than a clean estimate. The correct approach would have been to hold out a validation
set for threshold selection and reserve the test set for reporting only.
 
**The `day` feature does not generalise.** It is derived from `Time` as a monotonic index, and
in the dataset it only takes the values 0 and 1 because the record spans two days. A real
transaction would produce a value outside the range seen in training. `hour` does generalise,
because it is cyclical.
 
**The data is from 2013.** Fraud methods change continuously, so a model trained on
transactions from over a decade ago does not reflect current patterns. In production this
requires periodic retraining and drift monitoring, not a one-off deployment.
