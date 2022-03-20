---
toc: true
layout: post
description: Install Docker from docker repository on an AWS Ubuntu server
categories: [AWS]
title: Install Docker from docker repository on an AWS Ubuntu server
hide: false
comments: true
---

We are going to install docker from docker repository on an AWS Ubuntu server. The steps given here are as per the official docker installation [document](https://docs.docker.com/engine/install/ubuntu/)

## Set up the Repository

Update `apt` package index

```
sudo apt-get update
```

Install packages to allow `apt` to use a repository over HTTPS

```
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```    


Add Docker’s official GPG key

```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
```

Setup the stable repository

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
```

## Install Docker Engine

Update `apt` package index

```
sudo apt-get update
```

install the latest version of Docker Engine and containerd

```
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

## Test Installation

Run a Hello World docker container. You should see a hello-world image pull and a hello world message. 

![](/images/2022-03-19-install-docker-in-an-aws-ubuntu/image1.png)

## User Set up

Add your user id in Docker user group

```
sudo usermod -aG docker $USER
```

Reboot server

```
sudo reboot now
```

Check docker images and running containers in your server.

![](/images/2022-03-19-install-docker-in-an-aws-ubuntu/image2.png)

Latest version of Docker engine is successfully installed.
