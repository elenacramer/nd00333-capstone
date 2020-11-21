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
