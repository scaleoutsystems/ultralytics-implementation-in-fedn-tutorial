# ultralytics-implementation-in-fedn-tutorial
A tutorial of how to implement ultralytics models in a federated setting using FEDn

## Introduction
This tutorial will guide you through the process of implementing a federated learning setup using the [FEDn] framework and the [Ultralytics] models. The tutorial will cover the following steps:
1. ...

## Prerequisites
- [FEDn]<https://fedn.scaleoutsystems.com/>
- [Ultralytics]<https://www.ultralytics.com/>

## Step 1: Installing prerequisites
First, you need to install the prerequisites. You can install everything using pip:
```bash
pip3 install -r requirements.txt
```

## Step 2: Setting up configurations
Next, you need to set up the configurations for the Ultralytics models. 
Number of classes (nc) needs to be set in both the `data.yaml` and `yolov8_.yaml` file. 

