# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains data of various clients of a bank involved in marketing data. Our aim is to predict if the client will subscribe to a fixed term deposit or not denoted with feature y in the dataset. 
Dataset - https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv
The model which performed well on the given dataset was AutoML with 0.9168 accuracy than Hyperdrive model with accuracy 0.9107.

## Scikit-learn Pipeline
### Setting Up Training Script
Firstly, The libraries were imported then dataset in csv format was imported through specified URL using TabularDatasetFactory. Then running training script where clean_data function is used for cleaning and one hot encode the data. With train_test_split function, the data is splitted into training and testing with test size 0f 0.3. 
### HyperDrive Pipeline
* Created compute cluster using vm_size of "Standard_D2_V2" in provisioning configuration and max_nodes of 4.
* Specified a parameter sampler i.e RandomParameterSampling, since randomly selects both discrete and continuous hyperparameter values. The benefit of using Random Sampling is that it supports early termination of low peformance runs. 
* Specified a policy early stopping policy i.e Bandit Policy, it helps to automatically terminate poorly performing runs based on slack factor.It improves computational    efficiency. The benefit is that policy early terminates any runs where the primary metric is not within the specified slack factor with respect to best performing training run.
* Created a SKLearn estimator for use with train.py.\
est = SKLearn(source_directory = "./",
            compute_target=cpu_cluster,
            vm_size='STANDARD_D2_V2',
            entry_script="train.py")
* Created a HyperDriveConfig using the estimator, hyperparameter sampler, and policy with max_total_runs=20 and max_concurrent_runs=4.Used get_best_run_by_primary_metric() method of the run to select best hyperparameters.\
hyperdrive_config = HyperDriveConfig(estimator=est, hyperparameter_sampling=ps, policy=policy, primary_metric_name='Accuracy', primary_metric_goal=PrimaryMetricGoal.MAXIMIZE, max_total_runs=20, max_concurrent_runs=4)
            



## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
