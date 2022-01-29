---
toc: true
layout: post
description: Install MONAI Deploy with python 3.7.
categories: [MONAI for Healthcare]
title: Setup MONAI Deploy SDK for medical imaging on a AWS Ubuntu
hide: false
comments: true
---

Late 2021 NVIDIA Clara Deploy for traditional medical image inference has been depricated. MONAI Deploy provides the same service as Clara deploy. 

Clara deploy is now merged with Clara Holoscan and in the future may have supprt for traditional image devices but for now supporting only medical devices that combines hardware systems and sensors.

# Setup MONAI Deploy SDK for medical imaging on a AWS Ubuntu

MONAI Deploy SDK requires Python 3.7 or up. Install or update Python. Ensure pip is installed for the Python version installed.

```
python3.7 -m pip install pip
```

![](/images/2022-01-27-install-monai-deploy/image1.png)

## Install MONAI SDK

Now install MONAI Deply SDK with pip install

```
pip install monai-deploy-app-sdk
```

![](/images/2022-01-27-install-monai-deploy/image2.png)

You will have MONAI deploy installed in your server.
