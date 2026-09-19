import pickle

client = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}


with open('./cohort-documents/pipeline.bin', 'rb') as pipeline:
    res = pickle.load(pipeline)
    y_pred = res.predict_proba([client])[0,1]
    print(round(y_pred, 3))
    #the output was: $ python script_pipeline.py ---> 0.5329944186432507