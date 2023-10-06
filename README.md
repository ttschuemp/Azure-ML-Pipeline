# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This data contains information about customer targeted by a financial institute for marketing purposes. We want to predict a `yes` or `no` target label showing whether or not to offer a service to them. 

The best performing model was a VotingEnsemble by Azure AutoML.


## Scikit-learn Pipeline

The pipeline is constructed using the ScriptRunConfig, with its core functionality in the train.py script, where a Logistic Regression model is implemented. Data is sourced from a CSV-file and then transformed into a dataset using the Azure TabularDatasetFactory.

Random Parameter Sampling offers the benefit of quickly exploring the parameter space, yielding results that are in general as effective as those achieved through more exhaustive parameter searches.

The Bandit Policy ensures that runs that are underperforming compared to existing results are canceled. This conserves valuable time and computing resources.

## AutoML

AutoML generated a Voting Ensemble with hyperparameters set for classification, prioritizing accuracy as the primary metric, enabling the explanation of the best model, blocking TensorFlowLinearClassifier and TensorFlowDNN, utilizing 4-fold cross-validation, and with a training time limit of 30 min. Additionally, it employed k-fold cross-validation for validation and allowed a maximum of 1 concurrent iteration.

## Pipeline comparison
The AutoML model achieved an accuracy of 91.75%, whereas the Logistic Regression model attained 90.99%. Notably, the approaches differ significantly, the Hyperdrive pipeline restricted us to a specific algorithm, whereas Auto-ML offered the flexibility to select from many algorithms. In this context, Auto-ML proved to be more adaptable and required less setup effort with an even better result in only 30 min. 

## Future work
Investigate additional feature engineering techniques that might improve the model's ability to capture relevant information from the data as well as trying to generate a larger training set might be worth trying. 
