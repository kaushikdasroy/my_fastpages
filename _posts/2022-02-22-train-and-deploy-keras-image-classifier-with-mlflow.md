---
toc: true
layout: post
description: Train and Deploy Keras Image Classifier with MLflow
categories: [MLOps]
title: Train, Deploy and Track a Keras Image Classifier with MLflow
hide: false
comments: true
---

In this post, I will demonstarte two important features of MLFlow - training and deploying models. I will be using MLFlow Tracking server to collect and display experiment metrics.

Use MLFlow official [github](https://github.com/mlflow/mlflow/tree/master/examples/flower_classifier) as reference.

I am using an AWS g4dn.xlarge instance for this demonstration. 

# Start MLFlow Tracking Server

Conda virtual environment is ised in this demonstartion to execute training and deployment. Follow [this](https://www.uplandr.com/post/how-to-use-conda-for-creating-virtual-environments-and-package-management) link to create and manage conda environments.

Create a new conda environment and follow the instructions provided [here](https://blog.uplandr.com/mlops/mlflow/2022/01/30/install-mlflow-on-aws.html) to start MLFlow Tracking server with Postgresql as entity store and S3 as Artifact store.

At this point, MLFlow tracking will be available in http://ip-address:5000
Go ahead and access to make sure tracking server is up and running


# Train a Flower Classifier model

Clone the official [MLFLow repo](https://github.com/mlflow/mlflow). We train a VGG16 deep learning model to classify flower species from photos using a dataset available from [tensorflow.org](www.tensorflow.org). Keras is used to train the model.


> Note: make sure to update `MLFLOW_TRACKING_URI` environment variable with the MLFlow Tracking server information e.g.  http://ip-address:5000

Run training with 

```
 mlflow run examples/flower_classifier
 ```

This will create a conda virtual environment for training the model.

![](/images/2022-02-22-train-and-deploy-model-with-mlflow/image1.png)

After successful completion of training, mlflow generates a run id

![](/images/2022-02-22-train-and-deploy-model-with-mlflow/image2.png)

Open the tracking URI at http://ip-address:5000 and check the training parameters logged

![](/images/2022-02-22-train-and-deploy-model-with-mlflow/image3.png)

# Deploy the Model 

Deploy the model from the tracking server REST endpoint by executing `mlflow models serve`. `run id` created during the training is used to refer the model to serve. This also creates a separate conda virtual environment for deployment purpose.

![](/images/2022-02-22-train-and-deploy-model-with-mlflow/image4.png)

# Test deployed Model

Go to directory where `score_images_rest.py` is placed and run the script with test image as an argument

![](/images/2022-02-22-train-and-deploy-model-with-mlflow/image5.png)

![](/images/2022-02-22-train-and-deploy-model-with-mlflow/image6.png)

The model runs on the passed flower image and predicts which flower it is!

# Possible Errors

Following errors may appear while completing the training and deployment

## `AttributeError: 'str' object has no attribute 'decode'`

While model serving, this error occurs if version of installed h5py package version > 3 
Downgrade the version of h5py to 2.10.0 by running `pip install h5py==2.10.0` in the conda virtual environment for deployment and that should solve this problem. 

## `ModuleNotFoundError: No module named 'boto3'`

This error occurs during `mlflow run` if the virtual environment does not contain boto3. Install boto3 package and retry. 
