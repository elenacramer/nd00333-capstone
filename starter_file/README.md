*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

## Table of Contents
- [Problem Statement](##problem)
- [Data Set](##dataset)
    - [Task](###task)
    - [Access](###access)
- [Set Up and Installation](##setup)
- [Automated ML](##automl)
    - [Result](##automl_result)
- [Hyperparameter Tuning](##hyperdrive)
    - [Result](##hyperdrive_result)
 - [Model Deployment](##deployment)
 - [Recreen Recording](##recording) 
 - [Standout Suggestions](##standout)

## Problem Statement <a name="problem"></a>
In this project we will consider a regression problem, i.e. a process where a model learns to predict a continuous value output for a given input data). We first apply AutoML where multiple models are trained to fit the training data. We then choose and save the best model, that is, the model with the best score. Secondly, we build a simple neural network consisting of two hidden layers. In particular, a keras model where we tune hyperparameters using HyperDrive.  


## Dataset  <a name="dataset"></a>
In this project we consider the *California housing* data set from [kaggle](https://www.kaggle.com/camnugent/california-housing-prices). The data contains information from the 1990 California census. So although it may not help us with predicting current housing prices, we chose the data set because it requires rudimentary data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome. This enables us to focus on all required configuration to work with Azure ML. 

### Task  <a name="task"></a>
Our objective is to build prediction models that predict the housing prices from the set of given house features. Thus our task is a regression problem (a process where a model learns to predict a continuous value output for a given input data). 

### Access <a name="access"></a>
We downloaded the *housing.csv* from kaggle localy to the virtual machine and uploaded the csv file to the Azure ML platform.

## General Set Up <a name="setup"></a>
Before we either apply Automated ML or tune hyperparamteres for a keras model, we need to the following steps:
- Import all needed dependencies
- Set up a Workspace and initialize an experiment
- Create or attach a compute resource (VM with cpu for automated ML and VM with gpu for hyperdrive)
- Load csv file and register data set in the *Dataset*
- Initialize AutoMLConfig / HyperDriveConfig object
- Submit experiment 
- Save best model
- Deploy and consume best model (either for the automated ML or hyperdrive run)


## Automated ML <a name="automl"></a>
To configure the Automated ML run we need to specify what kind of a task we are dealing with, the primary metric, train and validation data sets (which are in *TabularDataset*  form) and the target column name. Featurization is set to "auto", meaning that the featurization step should be done automatically. To avoid overfitting we enable early stopping. 
```
automl_setting={
    "featurization": "auto",
    "experiment_timeout_minutes": 30,
    "enable_early_stopping": True,
    "verbosity": logging.INFO,
    "compute_target": compute_target
}

task="regression" 
automl_config = AutoMLConfig( 
    task=task, 
    primary_metric='normalized_root_mean_squared_error', 
    training_data=train, 
    validation_data = test, 
    label_column_name='median_house_value', 
    **automl_setting
)
```

Here are screenshots showing the output of the `RunDetails` widget: 
![rund_detail1](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/automated_ml/screenshots/automl_run.png)
![run_details2](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/automated_ml/screenshots/autml_run2.png)

### Results <a name="automl_result"></a>
The best model from the automated ML run is *LightGBM* with mean absolute error (mae) of 32.376,38. Automated ML applied a MaxAbsScaler. 

The following screenshots shows the best run ID and mae:
![best_model](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/automated_ml/screenshots/best_model.png)

We can perhaps improve the model by using customized featurization by passing a FeaturizationConfig object to the featurization parameter of the AutoMLConfig class. This, for example, enables us to choose a particular encoder for the categorical variables, a strategy to impute missing values, ect..


## Hyperparameter Tuning <a name="hyperdrive"></a>
We will compare the above automl run with a deep neural network, in particular, a *keras Sequential model* with two hidden layers. We tune the following hyperparamters with HyperDrive:
- batch size,
- number of epochs,
- number of units for the two hidden layers.

To initialize a HyperDriveConfog class we need to specify the following:
- Hyperparameter space: RandomParameterSampling defines a random sampling over the hyperparameter search spaces. The advantages here are that it is not so exhaustive and the lack of bias. It is a good first choice.
```
ps = RandomParameterSampling(
    {
        '--batch-size': choice(25, 50, 100),
        '--number-epochs': choice(5,10,15),
        '--first-layer-neurons': choice(10, 50, 200, 300, 500),
        '--second-layer-neurons': choice(10, 50, 200, 500),
    }
)
```
- Early termination policy: BanditPolicy defines an early termination policy based on slack criteria and a frequency interval for evaluation. Any run that does ot fall within the specified slack factor (or slack amount) of the evaluation metric with respect to the best performing run will be terminated. Since our script reports metrics periodically during the execution, it makes sense to include early termination policy. Moreover, doing so avoids overfitting the training data. For more aggressive savings, we chose the Bandit Policy with a small allowable slack.
```
policy =  BanditPolicy(evaluation_interval=2, slack_factor=0.1, slack_amount=None, delay_evaluation=0)
```

- A ScriptRunConfig for setting up configuration for script/notebook runs (here we use the script "keras_train.py"). Since we are using a keras model, we also need to set up an environment.

```
# Evironment set up
# conda_dependencies.yml is stored in the working directory

from azureml.core import Environment

keras_env = Environment.from_conda_specification(name = 'keras-2.3.1', file_path = 'conda_dependencies.yml')

# Specify a GPU base image
keras_env.docker.enabled = True
keras_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04'

# ScriptrunConfig 
src = ScriptRunConfig(source_directory=project_folder,
                      script='keras_train.py',
                      compute_target=compute_target,
                      environment=keras_env)
 ```     
- Primary metric name and goal: The name of the primary metric reported by the experiment runs (mae) and if we wish to maximize or minimize the primary metric (minimze).
- Max total runs and max concurrent runs : The maximum total number of runs to create and the maximum number of runs to execute concurrently. Note: the number of concurrent runs is gated on the resources available in the specified compute target. Hence ,we need to ensure that the compute target has the available resources for the desired concurrency.

The HyperDriveConfig obejct:
 ``` 
hyperdrive_config = HyperDriveConfig(
    hyperparameter_sampling = ps, 
    primary_metric_name ='MAE', 
    primary_metric_goal = PrimaryMetricGoal.MINIMIZE, 
    max_total_runs = 8, 
    max_concurrent_runs=4, 
    policy=policy, 
    run_config=src
)
 ``` 
Next we submit the hyperdrive run to the experiment (i.e. launch an experiment) and show run details with the RunDeatails widget:
 ``` 
hyperdrive_run = experiment.submit(hyperdrive_config, show_output=True)
RunDetails(hyperdrive_run).show()
 ``` 
Here is a screenshot of the RunDetails widget:
![rundetails_hyperdrive](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/hyperdrive_keras_model/run_details_hyperdrive.png) 

We collect and save the best model, that is, keras model with the tuned hyperparameters which yield the lowest mean absolute error:
 ``` 
best_run=hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
 ``` 
 
### Results <a name="hyperdrive_result"></a>
Here are the results of our hyperdrive run, that is, the tuned hyperparameters:
 ``` 
{'Batch Size': 50,
 'Epochs': 15,
 'Loss': 5399614188.155039,
 'MAE': 53041.2109375}
  ``` 
Here is the screenshot of the best model:
![best_model_keras](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/hyperdrive_keras_model/best_model_hyperdrive.png)
![best_model_overview](https://github.com/elenacramer/nd00333-capstone/blob/master/starter_file/hyperdrive_keras_model/best_model_overview_hyperdrive.png)

We can perhaps improve the mean absolute error score by:
- choosing the more exhaustive Grid Sampling strategy,
- keep the number of epochs fixed and tune the hyperparameter *learning rate* for the keras optimizer.

## Model Deployment <a name="deployment"></a>
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording  <a name="recording"></a>
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions <a name="standout"></a>
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
