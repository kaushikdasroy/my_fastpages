---
toc: true
layout: post
description: Create a simple image processing app using MONAI Deploy SDK
categories: [MONAI for Healthcare]
title: Create a simple image processing app using MONAI Deploy SDK
hide: false
comments: true
---

I am going to recreate MONAI official version of imaging processing app creation process given [here](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/notebooks/tutorials/01_simple_app.html)

I will create a MAP (docker image) of the application and will retain it for reference in application deployment using MONAI Inference Service. 

# Create MONAI operators and Application Class

## Setup Environment

Ensure MONAI Deploy SDK and scikit-image is installed 

```
python -c "import PIL" || pip install -q "Pillow"
python -c "import skimage" || pip install -q "scikit-image"
python -c "import monai.deploy" || pip install -q "monai-deploy-app-sdk"
```

![](/images/2022-01-27-install-monai-deploy/image1.png)

## Clone the git repo with code and test files

> Note: Case courtesy of Dr Bruno Di Muzio, Radiopaedia.org. From the case rID: 41113

Clone the MONAI SDK Deploy [repository](https://github.com/Project-MONAI/monai-deploy-app-sdk.git) for acessing the test image and example codes.

```
git clone https://github.com/Project-MONAI/monai-deploy-app-sdk.git
```

Execute the application code from the cloned repository with path for test input image and location of output. The application code serially strings three operators in a single calss. Three operators are - Sobel Operator, Median Operator and Gaussian Operator. 

```
python examples/apps/simple_imaging_app/app.py -i examples/apps/simple_imaging_app/brain_mr_input.jpg -o output
```

> Note: the above command is same as `monai-deply exec` command

![](/images/2022-01-29-creating-simple-app-using-monai-deploy/image1.png)

Create MONAI App package (MAP Docker Image)

```
monai-deploy package examples/apps/simple_imaging_app -t simple_app:latest
```

![](/images/2022-01-29-creating-simple-app-using-monai-deploy/image2.png)

## Run the docker image with an input image locally

We will use the same input image which we used for test, which is not ideal.

```
mkdir -p input && rm -rf input/*
cp examples/apps/simple_imaging_app/brain_mr_input.jpg input/
```

Execute MAP locally by MAR (MONAI Application Run)

```
monai-deploy run simple_app:latest input output
```

![](/images/2022-01-29-creating-simple-app-using-monai-deploy/image3.png)

Navigate to the output folder and locate final_output.png

execute eog to see the final_output.png

![](/images/2022-01-29-creating-simple-app-using-monai-deploy/image4.png)

We used MONAI Deploy SDK to process a simple image using three operators (Sobel Operator, Median Operator and Gaussian Operator)
