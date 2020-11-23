import json
import numpy as np

import joblib
from azureml.core.model import Model
import azureml.train.automl

def init():
    global model
    model_path = Model.get_model_path('automl_housing')
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    y_hat = model.predict(data)
    return y_hat.tolist()


