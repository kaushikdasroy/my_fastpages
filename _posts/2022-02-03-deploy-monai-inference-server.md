```python
---
toc: true
layout: post
description: Steps to build a Spleen segmentation app and deploy in MONAI Inference Service (MIS). MIS is a inference service which can be called over HTTP to execute a MONAI Application Package (MAP)
categories: [MLOps, MLFlow]
title: Building and Deploying A Spleen Segmentation app using MONAI App Packager(MAP) and MONAI Inference Service (MIS)
hide: false
comments: true
---
```

In this post, I will build and deploy a spleen segmentation AI model provider by MONAI, in a AWS environment. The inference service can invoked over HTTP with paylod and the service will produce segmention file for visualization. 


We will download spleen segmentation model and data from source in MONAI App Deploy SDK [examples](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/notebooks/tutorials/03_segmentation_app.html), build the model in an MONAI application package(MAP), and will deploy the MAP using MONAI Inference Service (MIS) for consumption over HTTP. The doumentation is available [here](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/notebooks/tutorials/03_segmentation_app.html).

The process can be used to deploy any model in torch script using MIS.

# 1. Download Segmentation Model and Test Data

Run following commands to download the `model.ts`, a torch script model for spleen segmentation, and `dcm` folder containing the dicom files for testing purpose.

```
pip install gdown
gdown "https://drive.google.com/uc?id=1GC_N8YQk_mOWN02oOzAU_2YDmNRWk--n"
unzip -o "ai_spleen_seg_data_updated_1203.zip"
```


Create a `my_app` folder for the application folder structure.

```
mkdir -p my_app
```

Move python files from MONAI App Deploy SDK [examples](https://docs.monai.io/projects/monai-deploy-app-sdk/en/latest/notebooks/tutorials/03_segmentation_app.html) to `my_app` directory - `__init__.py`,  `__main__.py`,  `app.py`,  `spleen_seg_operator.py`
These files are model specific inference operator classes packaged into application class. Application directory required to have specific `.py` files for MONAI packager to work. 

![](/images/2022-02-03-deploy-monai-inference-server/image-1.png)

# 2. Package the Segmentation App

Go to the directory where `my_app` folder and `model.ts` is placed and run the following command. This will start building a package (MAP).
```
monai-deploy package -b nvcr.io/nvidia/pytorch:21.11-py3 my_app --tag my_app:latest -m model.ts
```

![](/images/2022-02-03-deploy-monai-inference-server/image0.png)

This will create a docker image of the application

![](/images/2022-02-03-deploy-monai-inference-server/image-2.png)

# 3. Clone the MONAI Inference Service(MIS) Repositaory

MONAI application packages (MAP) can be deployed in MONAI inference Service (MIS), a RESTful inference service available for consumption over HTTP.

MAPs we build can be deployed on MIS and MIS takes care of underlying infrastructure to enable availablity of the MAPs. 

Next, I will cover the steps to install MIS. 

We will take next few steps to deploy a MAP using MIS. Helm Charts are utilized to deploy MIS for our MAP. 

Clone the MIS repo:

```
git clone https://github.com/Project-MONAI/monai-deploy-app-server.git
cd monai-deploy-app-server/components/inference-service


# 4. Build and Containerize the MIS

Build the MIS container:

```
./build.sh
```

# 5. Get MIS Helm charts 

```
wget "https://drive.google.com/uc?id=12uNO1tyqZh1oFkZH41Osliey7TRm-BBG"
unzip -o 'uc?id=12uNO1tyqZh1oFkZH41Osliey7TRm-BBG'

![](/images/2022-02-03-deploy-monai-inference-server/image1.png)

# 6. Update Helm Charts with MONAI App Package(MAP) information

To start the MIS we need to update the helm chart `values.yaml`. This is to configure MIS with a MAP.

## Getting the MAP - Application and Package Manifest Files 

To update the helm chart with application related information we need to export the manifest files for the application.
This can be done by executing a `docker run` on the MAP image, which will create two json files - `app.json` and `pkg.json`

These two manifest json files will provides parameters needed in updating helms chart `values.yaml`

Here `my_app:latest` is the docker image created by MAP.

```
mkdir ./export
docker run -it -v "$(pwd)/export/":"/var/run/monai/export/config/" my_app:latest
```


![](/images/2022-02-03-deploy-monai-inference-server/image2.png)

## Update Helm charts with MAP manifests

Open the values.yaml and update as following 

`images.monaiInferenceService` with MONAI Inference Service image name (monai/inference-service)
`images.monaiInferenceServiceTag`with MONAI Inference Service image tag (0.1)
`payloadService.hostVolumePath`with path to local directory which will serve as a shared volume between MIS and its PODs
`map.urn` with `map-image:tag`
`map.entrypoint` with data from command in `app.json` 
`map.cpu` with data from `pkg.json`
`map.memory` with data from `pkg.json`
`map.gpu` with data from `pkg.json`
`map.inputPath` with appending the `input.path` with the working-directory in `app.json`
`map.outputPath` with appending the `output.path` with the working-directory in `app.json`
`map.modelPath` with Model value path within MAP container. Can be obtained from `pkg.json` file. Only take the path till folder which hold the models ("/opt/monai/models")

# 7. Deploy MIS with MAP using the Helm Charts

```
helm install monai-inference-service ./charts/
```

![](/images/2022-02-03-deploy-monai-inference-server/image4.png)

To view the FastAPI generated UI for an instance of MIS, have the service running and then on any browser, navigate to http://HOST_IP:32000/docs 

![](/images/2022-02-03-deploy-monai-inference-server/image5.png)

# 8. Test the service 

With MIS running, I can make an inference request to the service using the /upload POST endpoint with the cluster IP and port from running kubectl get svc and a compressed .zip file containing all the input payload files (eg. input.zip)
```
Usage:
curl -X 'POST' 'http://<CLUSTER IP>:8000 OR <HOST IP>:32000/upload/' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@<PATH TO INPUT PAYLOAD ZIP>;type=application/x-zip-compressed' \
    -o output.zip
```



![](/images/2022-02-03-deploy-monai-inference-server/image6.png)

Output segmentation file is in the output foloder.

![](/images/2022-02-03-deploy-monai-inference-server/image7.png)

Open the segmented file in a viewer like [3DSlicer](https://www.slicer.org/)

![](/images/2022-02-03-deploy-monai-inference-server/image8.png)

It is possible to call the inference service from the FastAPI UI

![](/images/2022-02-03-deploy-monai-inference-server/image9.png)

This post shows how to build and deploy a torch script model using MONAI MAP and MIS. This process can be used to build and deploy other models as well.
