---
toc: true
layout: post
description: Install MLFlow Tracking on AWS
categories: [MLOps, MLFlow]
title: Install MLFlow Tracking on AWS
hide: false
comments: true
---

There are many MLOps tools ranging from all-in-one to fit for a purpose. Lets take a look at MLFlow which falls in the category of all in one tool having modules supports - Tracking, Projects, Model and Registry. It is an open-source tool. MLFlow supports many machine learning frameworks, such as Tensorflow, PyTorch, XGBoos, H20.ai, Scikit-learn etc.

MLFlow can be used in local machine, and cloud environments. First we will install MLFlow on a AWS environment. This is an experimental setup not a product onw.

In this post I will setup MLFlow Tracking. I will write other posts for other modules of MLFlow.

# MLFlow Tracking

Quoting from MLFlow.org 
"The MLflow Tracking component is an API and UI for logging parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results. MLflow Tracking lets you log and query experiments using Python, REST, R API, and Java API APIs."

MLFlow tracking is organized around `runs`. Each `run`execute a piece of data science code and records some information like - code version, time, source, parameters, Metrics, Artifacts.

`runs` can be recorded using MLFlow Python, R, Java, REST APIs from anywhere the code is run. These can be recorded from a notebook, cloud or standalone programs

`runs` can be recorded from a MLFlow Project and MLFlow remembers the project URI.

`runs` can be organiged in MLFlow experiments.

`runs` are recorded in the local machine in a folder called `mlruns`. `mlflow ui` brings the log in the tracking server for display.

## Run and Artifact recording

We will using mlflow with remote tracking server, backend and artifact stores.

In our example we will be using a remote server as tracking server, a Postgresql db as MLFlow entity store and a S3 bucket as our MLFlow artifact store.

Even though it is not demonstrated here, we can record runs by calling mlflow functions, python API in any code we run. The functions are detailed in the [mlflow](https://mlflow.org/docs/latest/tracking.html#logging-data-to-runs) official documentation. Autologging is also supported for most of the frameworks.

All the tracking ui functions can be called programmatically, which makes it easy to log runs and see results in tracking UI. 
`mlflow.set_tracking_uri()` connects to a tracking URI. There are many parameters can be passed to this function to establish authentication etc. `MLFLOW_TRACKING_URI` environment variable can be used to set the tracking URI.


## MLFlow Tracking Server

`mlflow server` command starts a tracking server. Backend storage and artifact storage details are provided with the command.

```
mlflow server \
    --backend-store-uri /mnt/persistent-disk \
    --default-artifact-root s3://my-mlflow-bucket/ \
    --host 0.0.0.0
```

Both a file based or a database based storage are supported as backend storage. SQLAlchemy database URI is used as database storage indicator. `<dialect>+<driver>://<username>:<password>@<host>:<port>/<database>` MLflow support mysql, mssql, sqlite and postgresql as database dilect. I am going to use a Postgresql in this post.

> To run model registry functionality Database based backend storage is required.

Default artifact storage is provided while creating the server and it can be overwritten during an experiment run if a new artifact location is provided. Artificate location can be a NFS file systems or a S3 compatible storage. I will be using Amazon S3.

MLFlow access the S3 access details from the IAM role, a profile in ~/.aws/credentials, or the environment variables AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY depending on which of these are available

# Launch EC2 environment, Postgresql and S3 Bucket

## Setup Tracking server (EC2)
I am using python 3.7 on a AWS Amazon Linux 2 AMI, t2.micro instance. This will work as tracking server. Ensure port 5000 is open to take HTTP request. Setup EC2 instance security to enable HTTP traffic in 5000 PORT.

## Setup a RDS Prostgresql db for entity store. 
I am creating the db in the same vpc as the EC2 instance and my instance has access to the db. My db security groups are default vpc security groups and these are sufficient to get access to db. If you choose a db outside ec2 vpc, you would require vpc peering and change in db instance's route table.

## Setup a S3 bucket for artifact storage. 
Create a s3 bucket. There are few steps required to make sure ec2 has access to the s3 bucket.
Create IAM role with minimum required access to S3. 
Attach the IAM role to ec2 instance. 
Verify that ec2 has access to s3 bucket by running `aws cli s3 ls` command in ec2 instance. It should show your bucket name. 

# Execute Installation

Run following commands:

```
# update packages
sudo yum update

# install python
sudo yum install python3.7

# install MLFlow and AWS python sdk
sudo pip3 install mlflow[extras] psycopg2-binary boto3

# start the mlflow serve. 
# "nohup is a POSIX command which means "no hang up". It ignores the HUP signal, does not stop when the user logs out.
nohup mlflow server --backend-store-uri postgresql://<USERID>:<PASSWORD>@<DB-ENDPOINT>:5432 --default-artifact-root s3://<S3-BUCKET-NAME> --host 0.0.0.0 &
```

It will start the MLFlow server


# Connect with the MLFow UI

Start the MLFlow ui by accessing the ec2 endpoint port 5000 ```http://<ec2-endpoint>:5000```

![](/images/2022-01-30-install-mlflow-on-aws/image1.png)

## Reference

Refer to [MLFlow official documentation](https://mlflow.org/docs/) to make the installation production grade by adding more security and learn about how to use the tracking server to track your ML Experiments from different environments. 

