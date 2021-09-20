import pandas as pd
import pickle
from collections import defaultdict
import numpy as np
from flask import abort



#format output
# {
#     "model":"german-credit-risk",
#     "version": "1.0.0",
#     "score_proba":{result}
#
# }
raw_input = {'person_age': 23,
 'person_income': 21000,
 'person_home_ownership': 'RENT',
 'person_emp_length': 2.0,
 'loan_intent': 'DEBTCONSOLIDATION',
 'loan_grade': 'C',
 'loan_amnt': 3000,
 'loan_int_rate': 12.68,
 'loan_status': 0,
 'loan_percent_income': 0.14,
 'cb_person_default_on_file': 'Y',
 'cb_person_cred_hist_length': 2}

test_input = {'person_age': 22,
 'person_income': 70000,
 'person_home_ownership': 'MORTGAGE',
 'person_emp_length': 6.0,
 'loan_intent': 'VENTURE',
 'loan_grade': 'B',
 'loan_amnt': 6000,
 'loan_int_rate': 11.49,
 'loan_percent_income': 0.09,
 'cb_person_default_on_file': 'N',
 'cb_person_cred_hist_length': 2,
 'loan_int_rate_nan': 11.49,
 'person_home_ownership_MORTGAGE': 1,
 'person_home_ownership_OTHER': 0,
 'person_home_ownership_OWN': 0,
 'person_home_ownership_RENT': 0,
 'loan_intent_DEBTCONSOLIDATION': 0,
 'loan_intent_EDUCATION': 0,
 'loan_intent_HOMEIMPROVEMENT': 0,
 'loan_intent_MEDICAL': 0,
 'loan_intent_PERSONAL': 0,
 'loan_intent_VENTURE': 1,
 'loan_grade_A': 0,
 'loan_grade_B': 1,
 'loan_grade_C': 0,
 'loan_grade_D': 0,
 'loan_grade_E': 0,
 'loan_grade_F': 0,
 'loan_grade_G': 0,
 'cb_person_default_on_file_N': 1,
 'cb_person_default_on_file_Y': 0,
 'loan_status': 0,
 'score_proba': 0.09694193831738365,
 'prediction': 0}

# One Hot Encoder
with open("OneHotEncoder-1.0.0.pkl", "rb") as f:
  encoder = pickle.load(f)

# Model
with open("Model-1.0.0.pkl", "rb") as f:
  model = pickle.load(f)

def formatting_data(raw_input):
    # error handling
    # Missing Column (Key) Handling as np.nan
    required_columns = ['person_age', 'person_income', 'person_home_ownership', 'person_emp_length', 'loan_intent', 'loan_grade', 
    'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']
    for col in required_columns:
        try:
            raw_input[col]
        except KeyError as err:
            # If the categorical column missing: reject request
            if (col == 'person_home_ownership' or col == 'loan_intent' or col == 'loan_grade' or col == 'cb_person_default_on_file'):
                abort(400, {'error': 'categorical column is missing'})
            else:
                abort(400, {'error': 'categorical column is missing'})
  
    categorical_columns = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    val = {
        "person_home_ownership": ['RENT','MORTGAGE','OWN','OTHER'],
        "loan_intent": ['EDUCATION', 'MEDICAL', 'VENTURE','PERSONAL','HOMEIMPROVEMENT','DEBTCONSOLIDATION'],
        "loan_grade": ['A','B','C','D','E','F','G'],
        "cb_person_default_on_file": ['Y','N']
    }
    for col in categorical_columns:
        if(raw_input[col] not in val[col]):
            abort(400, {'error': 'categorical column not match with the exact value'})


    # pandas DataFrame
    mapper_replace = {
      "null": np.nan,
      "": np.nan,
      None: np.nan
  }

    # Transform into DataFrame. Turn the dict to list first
    raw_input = pd.DataFrame([raw_input]).replace(mapper_replace)

    #raw_input = pd.DataFrame.from_dict(raw_input, orient = 'index').T.replace({None: np.nan, "null": np.nan, "": np.nan})
    # urutan column sama
    return raw_input

def preprocess(data):
    nominal_features = ['person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file']
    data_transformed = encoder.transform(data[nominal_features]).toarray()

    # Get the OHE column name
    column_name = encoder.get_feature_names(nominal_features)
    # Format into DF
    data_one_hot_encoded =  pd.DataFrame(data_transformed, columns= column_name, index=data[nominal_features].index).astype(int)
    # concat the data
    data = pd.concat([data,data_one_hot_encoded], axis=1).reset_index(drop=True)
    #print(data)
    return data

def predict(data):
    selected_features = ['person_age', 'person_income', 'person_emp_length', 
                            'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
                            'cb_person_default_on_file_N',
                            'cb_person_default_on_file_Y']
    pred_proba = model.predict_proba(data[selected_features])[:, 1]
    threshold = 0.6
    prediction = (pred_proba > threshold).astype(int)

    return { "data": [ { "pred_proba": float(pred_proba[0]), "prediction": int(prediction[0])} ] }
    # load model
    # input dict
    # preprocess preprocess_data
    # return final predictions
    #return final_prediction
    

def makeprediction(raw_input):
    data = formatting_data(raw_input)
    data = preprocess(data)
    result = predict(data)
    return result
    
if __name__ == "__main__":
    #result = predict(test_input)
    #result = formatting_data(raw_input)
    #result = preprocess(result)
    #result = predict(result)
    result = predict(pd.DataFrame([test_input]))
    print(result)

