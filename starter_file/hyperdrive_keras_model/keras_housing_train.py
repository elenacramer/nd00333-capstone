import numpy as np
import argparse
import os
import glob

import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback

import tensorflow as tf

from azureml.core import Run


print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)



parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
parser.add_argument('--first-layer-neurons', type=int, dest='n_hidden_1', default=100,
                    help='# of neurons in the first layer')
parser.add_argument('--second-layer-neurons', type=int, dest='n_hidden_2', default=100,
                    help='# of neurons in the second layer')
parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')

args = parser.parse_args()


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

data = get_dataset()

############ Data Preparation #####################################################################################

def prepare_data(data):
    encoded_column=pd.get_dummies(data['ocean_proximity'], prefix='ocp')
    data=data.join(encoded_column)
    data=data.drop("ocean_proximity", axis=1)

    target="median_house_value"
    y=data[target]
    X=data.drop(target, axis=1)
    
    return x,y

def split_scale(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # returns Dataframes and Series
    columns=x_train.columns.to_list()
    scaler_x = MinMaxScaler().fit(x_train[columns])
    scaled_x_train = scaler_x.transform(x_train[columns]) # numpy array
    scaled_x_test = scaler_x.transform(x_test[columns])

    max=y_train.max()
    min=y_train.min()
    scaled_y_train = (y_train-y_min)/(y_max-y_min) # pandas Series
    scaled_y_test=(y_test-y_min)/(y_max-y_min)
    

    return scaled_x_train, scaled_y_train.values, scaled_x_test, scaled_y_test.values


x, y = prepare_data(data)
x_train, y_train, x_test, y_test = split_scale(x,y)


#################### Build a simple MLP model #####################################

n_h1 = args.n_hidden_1
n_h2 = args.n_hidden_2
n_epochs = 1
batch_size = args.batch_size
learning_rate = args.learning_rate


#  MLP model with 2 hidden layers
model = Sequential()
model.add(Dense(n_h1, activation='relu', input_shape=[x_train.shape[1]]))
model.add(Dense(n_h2, activation='relu'))
# output layer
model.add(Dense(1))

model.summary()

opt = Adam(learning_rate=learning_rate)
model.compile(loss='mse',
              optimizer=opt,
              metrics=['mae'])




#class LogRunMetrics(Callback):
#    # callback at the end of every epoch
#    def on_epoch_end(self, epoch, log):
#        # log a value repeated which creates a list
#        run.log('Loss', log['val_loss'])
#        run.log('MAE', log['val_mean_absolute_error']) 


history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=n_epochs,
                    validation_data=(x_test, y_test),
                    verbose=1)

score = model.evaluate(x_test, y_test, verbose=0) # returns the loss value & metrics values (here mae) for the model in test mode

# log a single value
print('Test loss:', score[0])

print('Test mae:', score[1])

#lo metrics
try:
    run = Run.get_context()
    run.log('Test loss', score[0])
    run.log('MAE', score[1])
except:
    print("Running locally")

# create a ./outputs/model folder in the compute target
# files saved in the "./outputs" folder are automatically uploaded into run history
os.makedirs('./outputs/model', exist_ok=True)

# serialize NN architecture to JSON
model_json = model.to_json()
# save model JSON
with open('./outputs/model/model.json', 'w') as f:
    f.write(model_json)
# save model weights
model.save_weights('./outputs/model/model.h5')
print("model saved in ./outputs/model folder")
