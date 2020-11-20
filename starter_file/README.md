*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Your Project Title Here

## Table of Contents
- [Problem Statement](##problem)
- [Data Set](##dataset)
  -[Task](###task)
  -[Access](###access)
- [Set Up and Installation](##setup)
- [Automated ML](##automl)
  -[Result](##automl_result)
-[Hyperparameter Tuning](##hyperdrive)
  -[Result](##hyperdrive_result)
 -[Model Deployment](##deployment)
 -[Recreen Recording](##recording) 
 -[Standout Suggestions](##standout)

## Problem Statement <a name="problem"></a>

In this project we will consider a regression problem, i.e. a process where a model learns to predict a continuous value output for a given input data). We first apply AutoML where multiple models are trained to fit the training data. We then choose and save the best model, that is, the model with the best score. Secondly, we build a simple neural network consisting of two hidden layers. In particular, a keras model where we tune hyperparameters using HyperDrive.  


## Dataset  <a name="dataset"></a>
In this project we consider the *California housing* data set from [kaggle](https://www.kaggle.com/camnugent/california-housing-prices). The data contains information from the 1990 California census. So although it may not help us with predicting current housing prices, we chose the data set because it requires rudimentary data cleaning, has an easily understandable list of variables and sits at an optimal size between being to toyish and too cumbersome. This enables us to focus on all required configuration to work with Azure ML. 

### Task  <a name="task"></a>
Our objective is to build prediction models that predict the housing prices from the set of given house features. Thus our task is a regression problem (a process where a model learns to predict a continuous value output for a given input data). 

### Access <a name="access"></a>
We downloaded the *housing.csv* from kaggle localy to the virtual machine and uploaded the csv file in the *Dataset* section. 

## Project Set Up and Installation <a name="setup"></a>
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.


## Automated ML <a name="automl"></a>
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results <a name="automl_result"></a>
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning <a name="hyperdrive"></a>
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results <a name="hyperdrive_result"></a>
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment <a name="deployment"></a>
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording  <a name="recording"></a>
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions <a name="standout"></a>
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
