from sklearn.ensemble import GradientBoostingRegressor
import argparse
import os
import numpy as np
#from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd
from azureml.core.run import Run



run = Run.get_context()

########################## get data ###############################################################################
from azureml.core.dataset import Dataset
import azureml
from azureml.core import Workspace
import pandas ad pd

def get_dataset():
    ws = Workspace.from_config()
    key="housing"
    dataset = ws.datasets[key] 
    data = dataset.to_pandas_dataframe()
    return data


############ Data Preparation #####################################################################################

def prepare_data(data):
    encoded_column=pd.get_dummies(data['ocean_proximity'], prefix='ocp')
    data=data.join(encoded_column)
    data=data.drop("ocean_proximity", axis=1)

    target="median_house_value"
    y=data[target]
    X=data.drop(target, axis=1)
    
    return x,y



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=1.0, help="learning rate shrinks the contribution of each tree by learning_rate")
    parser.add_argument('--n_estimators', type=int, default=100, help="The number of boosting stages to perform.")

    args = parser.parse_args()

    run.log("Learning Rate:", np.float(args.learning_rate))
    run.log("Number of estimators:", np.int(args.n_estimators))
    
    data = get_dataset()
    x, y = prepare_data(data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
   

    model = GradientBoostingRegressor(learning_rate=args.learning_rate, n_estimators=args.n_estimators).fit(x_train, y_train)
    
  
    pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, pred)
    run.log("Mean_Absolute_Error", np.float(mae))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib') 

if __name__ == '__main__':
    main()

