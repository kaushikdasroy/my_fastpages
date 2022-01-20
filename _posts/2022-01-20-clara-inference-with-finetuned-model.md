---
toc: true
layout: post
description: Inference using a Fine-tuned model.
categories: [Nvidia Clara for Healthcare]
title: Nvidia Clara based inference using a Fine-tuned model
hide: false
---


# Inference using Fine-tuned AI Model and NVIDIA Triton Server

Let's start clara back up again.

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image1.png" style="width:6.5in;height:2.20833in" />

Deployed kubernetes pods

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image2.png" style="width:6.5in;height:1.15278in" />

## Setting up TRITON inference server to host our model

Create a folder structure as below

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image3.png" style="width:6.5in;height:0.27778in" />

Move the refined model we created before to this directory.

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image4.png" style="width:6.5in;height:0.19444in" />

Refer: [<u>https://blog.uplandr.com/2021/09/02/Fine-tune-a-Chest-Xray-Classification-Model-using-NVIDIA-Clara-Train.html</u>](https://blog.uplandr.com/2021/09/02/Fine-tune-a-Chest-Xray-Classification-Model-using-NVIDIA-Clara-Train.html)

Refer to this documentation to know about the directory structure: [<u>https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/</u>](https://docs.nvidia.com/deeplearning/triton-inference-server/master-user-guide/)

Create a file as below:

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image5.png" style="width:6.5in;height:8.66667in" />

Create another file with our labels

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image6.png" style="width:6.5in;height:7.84722in" />

## Create a Clara Deploy Operator

We will create a clara deploy operator. This operator will be running in a container independent of clara deploy and the operator can be made part of a deployment pipeline.

Steps are given in here - [<u>https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference</u>](https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference)

We need to grab these-

-   Clara deploy base inference operator

-   Clara chest classification operator

-   TRITIS (Triton) container

Make sure you have your ngc connection or else rebuild connection to ngc with docker login nvcr.io

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image7.png" style="width:6.5in;height:2.625in" />

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image8.png" style="width:6.5in;height:2.70833in" />

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image9.png" style="width:6.5in;height:3.45833in" />

Retag the docker image as latest

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image10.png" style="width:6.5in;height:0.18056in" />

Create a Operator directory structure

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image11.png" style="width:6.5in;height:0.68056in" />

Run the chest xray operator docker container

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image12.png" style="width:6.5in;height:0.18056in" />

Copy 2 files from the container

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image13.png" style="width:6.5in;height:0.19444in" />

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image14.png" style="width:6.5in;height:0.18056in" />

Exit from the container and change the owner for the files to your own. There are few changes to be made in these two files. Change the model to be used to “classification_covidxray_v1” from “classification_cheastxray_v1”. And in the config_inference change the \`subtrahend\` and \`divisor\` to 128.

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image15.png" style="width:6.5in;height:0.625in" />

Create a Dockerfile with base as app_base_inference and copy the config files taken from the chestxray

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image16.png" style="width:6.5in;height:6.98611in" />

## Test the custom operator

We will run the operator outside of clara deploy pipeline using docker and a script.

Copy the script from the “executing with docker” section of the link - [<u>https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference</u>](https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference)

Change the script as follows to make it suitable for our purpose.

Create a file

vi /etc/clara/operators/run_covid_docker.sh

Open the file run_covid_docker.sh and paste the script from “executing with docker” section of the link - [<u>https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference</u>](https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference)

Need to make following edits:

Replace APP_NAME with “app_covidxray”

Replace MODEL_NAME with “classification_covidxray_v1”.

The line that starts with nvidia-docker — replace $(pwd) with clara/common (so this part reads -v /clara/common/models/${MODEL_NAME}:/models/${MODEL_NAME}

In the line “-v $(pwd)/input:/input \\”, replace $(pwd) with “/etc/clara/operators/app_covidxray”

In the line “-v $(pwd)/output:/output \\”, replace $(pwd) with “/etc/clara/operators/app_covidxray”

In the line “-v $(pwd)/logs:/logs \\”, replace $(pwd) with “/etc/clara/operators/app_covidxray”

In the line “-v $(pwd)/publish:/publish \\”, replace $(pwd) with “/etc/clara/operators/app_covidxray”

Comment the lines as indicated in notes of the file if using NGC containers for testing.

Save and exit from the file.

Copy one image in our test input folder.

cp /etc/clara/experiments/covid-training-set/training-images/1-s2.0-S0929664620300449-gr2_lrg-b.png /etc/clara/operators/app_covidxray/input

Change permission of the script file and run the script

chmod 700 /etc/clara/operators/run_covid_docker.sh

cd /etc/clara/operators/

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image17.png" style="width:6.5in;height:0.23611in" />

To check the job was successful, check the output folder for a file with the inference

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image18.png" style="width:6.5in;height:0.20833in" />

Check the output folder and display the image with labels and categories and % of chance

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image19.png" style="width:6.5in;height:0.45833in" />

Output with inference shown in the picture!

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image20.png" style="width:6.5in;height:5.19444in" />

## Create a Clara Deploy Pipeline for inference

Create a clean docker build using the Dockerfile

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image21.png" style="width:6.5in;height:1.79167in" />

The steps are described here - [<u>https://docs.nvidia.com/clara/deploy/sdk/Applications/Pipelines/ChestxrayPipeline/public/docs/README.html</u>](https://docs.nvidia.com/clara/deploy/sdk/Applications/Pipelines/ChestxrayPipeline/public/docs/README.html)

[<u>https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference</u>](https://ngc.nvidia.com/catalog/containers/nvidia:clara:app_base_inference)

Start with a chest xray classification pipeline and change it to fit covid xray pipeline

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image22.png" style="width:6.5in;height:0.19444in" />

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image23.png" style="width:6.5in;height:0.19444in" />

Make some changes covidxray-pipeline.yaml file to fit it for our purpose

Change the container image to - app_covidxray, and tag to latest

Remove the pull secrets part

Change all reference of chest xray to covid xray

Note: Make sure the triton server version is appropriate. Pay attention to app_base_inference version, reference pipeline version (in this case clara_ai_chestxray_pipeline) and triton server version. All these need to be in sync for the inference to work.

For the current example I am using app_base_inference ( not app_base_inference_v2 ) and I used nvcr.io/nvidia/tensorrtserver tag 19.08-py3 (rather than tritonserver). Change the “Command” to “trtserver” if using tensorrtserver.

Save and exit

Now we are ready to create our covid xray pipeline

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image24.png" style="width:6.5in;height:0.38889in" />

This will give you a pipeline id.

## Run test image through the pipeline

Now use the created pipeline to process one image from the input file.

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image25.png" style="width:6.5in;height:0.44444in" />

Manually start the job

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image26.png" style="width:6.5in;height:0.44444in" />

The completed pipeline view in Clara console (port 32002)

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image27.png" style="width:6.5in;height:1.63889in" />

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image28.png" style="width:6.5in;height:3.91667in" />

Output after download

<img src="/images/2022-01-20-clara-inference-with-finetuned-model/image29.png" style="width:5.46875in;height:5.5in" />

Here you have it, your own model is used in inference through triton server and clara pipeline!
