###################################### create scoring script ###############################
%%writefile score.py

import json
import numpy as np
import os
from sklearn.externals import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path('keras-housing-model)
    model = joblib.load(model_path)



input_sample = pd.DataFrame(data=[{
              "longitude": -122.23,         
              "latitude": 37.88,    
              "housing_median_age": 41.0,            
              "total_rooms": 888.0,
              "total_bedrooms": 129.0,
              "population": 322.0,
              "households": 126.0,
              "median_income": 8.3252,
              "ocp_<1H OCEAN": 0,
              "ocp_INLAND"; 0,
              "ocp_ISLAND": 0,
              "ocp_NEAR BAY": 1,
              "ocp_NEAR OCEAN": 0
            }])

def run(data):
    try:
        data = input_sample.values #np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
