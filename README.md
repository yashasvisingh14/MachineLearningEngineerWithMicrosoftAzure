# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset contains data of various clients of a bank involved in marketing data. Our aim is to predict if the client will subscribe to a fixed term deposit or not denoted with feature y in the dataset.\
Dataset - https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv \
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
hyperdrive_config = HyperDriveConfig(estimator=est, hyperparameter_sampling=ps, policy=policy, primary_metric_name='Accuracy', primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,   max_total_runs=20, max_concurrent_runs=4)
* Accuracy Achieved = 0.9107
            
## AutoML
*  Imported data from the provided URL again using TabularDatasetFactory then after cleaning the data it passed to an AutoMLConfig.\
automl_config = AutoMLConfig(
    compute_target=cpu_cluster,
    experiment_timeout_minutes=30,
    task='classification',
    primary_metric='accuracy',
    training_data=ds,
    label_column_name='y',
    enable_onnx_compatible_models=True,
    n_cross_validations=2)
* Task helps us determine the kind of machine learning problem we need to solve. It can be classification, regression, and forecasting.   
* The primary metric parameter determines the metric to be used during model training for optimization. In this case where classification scenario is used we provided accuracy as primary metric.
* experiment_timeout_minutes defines how long, in minutes, the experiment should continue to run, in our case its 30 minutes.
* n_cross_validations parameter sets number of cross validations to perform, based on the same number of folds.
* (ONNX) can help optimize the inference of the machine learning model.        
* Retrieved and saved the best automl model.
* Accuracy achieved = 0.9168

## Pipeline comparison
In HyperDrive, we control the model training process by adjusting parameters and finding the configuration of hyperparameters results in the best performance. It uses a fixed machine learning algorithm that is provided. Whereas,AutoML creates a number of pipelines in parallel that try different algorithms and parameters for us. It gives us the best model which "fits" our data. It trains and tunes the model using the target metric specified.
HyperDrive is typically computationally expensive. On the other hand, AutoML implements ML solutions without extensive programming knowledge. It saves time and resources. \
AutoML Architecture
* Identify the ML problem
* Choose whether you want to use the Python SDK or the studio web experience
* Specify the source and format of the labeled training data
* Configure the compute target for model training
* Configure the automated machine learning parameters 
* Submit the training run\
HyperDrive Architecture
* Define the parameter search space
* Specify a primary metric to optimize
* Specify early termination policy for low-performing runs
* Allocate resources
* Launch an experiment with the defined configuration
* Visualize the training runs
* Select the best configuration for your model \
**Accuracy HyperDrive = 0.9107**\
**Accuracy AutoML = 0.9168 (Voting Ensemble Model)**\
Hence, AutoML performed well with our data than HyperDrive run.


## Future work
In this project, certain parameters and metrics are used but to gain an improved accuracy we can experiment with them. For classfication experiment we used accuracy as our primary metric which can be replaced with average_precision_score_weighted, norm_macro_recall, precision_score_weighted and AUC_weighted according to the scenarios. With regression or forecast models we can have different experiment timeout minutes sets and cross validation folds. In HyperDrive, we can run model with different parameter sampling methods like Grid sampling, Bayesian sampling according to hyperparameters. We can also explore early termination policy which automatically terminate poorly performing runs. Early termination improves computational efficiency.

## Proof of cluster clean up
![Screenshot](https://user-images.githubusercontent.com/64837491/106479600-83ac1480-64d0-11eb-9dfa-38fe158b0d6a.png)
