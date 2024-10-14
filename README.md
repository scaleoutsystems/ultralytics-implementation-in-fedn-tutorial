# ultralytics-implementation-in-fedn-tutorial
A tutorial of how to implement ultralytics models in a federated setting using FEDn

## Introduction
This tutorial will guide you through the process of implementing a federated learning setup using the [FEDn] framework and the [Ultralytics] models. The tutorial will cover the following steps:
1. ...

## Prerequisites
- [FEDn] <https://fedn.scaleoutsystems.com/>
- [Ultralytics] <https://www.ultralytics.com/>

## Step 1: Start the server in FEDn
Firstly, create an account on the FEDn platform: `FEDn <https://fedn.scaleoutsystems.com/signup>`__
Once you are logged in, you need to start a new project by clicking on the `New project` button.
This initializes the server which later will be used to run the federated learning process.

## Step 2: Installing prerequisites
Next, you need to install the prerequisites. You can install everything using pip:
```bash
pip3 install -r requirements.txt
```

## Step 3: Setting up the data

For your dataset to be found by Ultralytics the "datasets_dir" needs to be configured.
This is done by setting runnning the following command:
```bash
yolo settings datasets_dir=<path_to_dataset>
```
This needs to be directed to a folder named "fed_dataset" where the data is structured as follows:

```
fed_dataset/
  train/
    images/
      image1.jpg
      image2.jpg
      ...
    labels/
      image1.txt
      image2.txt
      ...
  val/
    images/
      image1.jpg
      image2.jpg
      ...
    labels/
      image1.txt
      image2.txt
      ...
```

The labels should be in the format of:
```
<class> <x_center> <y_center> <width> <height>
```


## Step 4: Setting up configurations
Next, you need to set up the configurations for the Ultralytics models. 
Number of classes (nc) needs to be set in both the `data.yaml` and `yolov8_.yaml` file.
You choose which YOLOv8 model to use by specifying it in the file name `yolov8_.yaml`. 
- For YOLOv8n (nano), use `yolov8n.yaml`
- For YOLOv8s (small), use `yolov8s.yaml`
- For YOLOv8m (medium), use `yolov8m.yaml`
- For YOLOv8l (large), use `yolov8l.yaml`
- For YOLOv8x (extra large), use `yolov8x.yaml`

##  Step 5: Building the compute package
Next, the compute package needs to be built, do so by running the following command:
```bash
fedn package create -p client
```

## Step 6: Initializing the seed model
To initialize the seed model, run the following command:
```bash
fedn run build -p client
```

## Step 7: Initilize the server-side
The next step is to initilize the server side with the client code:

## Step 8: Start the client

## Step 9: Train the global model
