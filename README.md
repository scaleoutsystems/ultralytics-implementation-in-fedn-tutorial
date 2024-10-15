# ultralytics-implementation-in-fedn-tutorial
A tutorial of how to implement ultralytics models in a federated setting using FEDn

## Introduction
This tutorial will guide you through the process of implementing a federated learning setup using the [FEDn] framework in combination with [Ultralytics] YOLOv8 models. Federated learning allows multiple clients to collaboratively train a global model without sharing their local datasets, ensuring data privacy and security. This tutorial is designed for users familiar with machine learning and federated learning concepts and will provide step-by-step instructions on configuring and running a complete federated learning workflow.

By the end of this tutorial, you will have built a distributed training environment where clients independently train local models, and a global model is aggregated on the server. You will also learn how to use Ultralytics’ YOLOv8 models for object detection tasks in this federated setting.

1.	Starting the server in FEDn – Learn how to initiate the central server that coordinates federated learning activities.
2.	Cloning the repository – Set up the project by cloning the necessary repository for configuration and deployment.
3.	Installing prerequisites – Install all required dependencies for both the server and client environments.
4.	Setting up the dataset – Properly structure and configure your dataset to be used with Ultralytics’ models.
5.	Setting up configurations – Adjust the model and dataset configurations, such as defining the number of classes for the YOLOv8 model.
6.	Building the compute package – Build the compute package to prepare for the training process.
7.	Initializing the seed model – Generate the initial model to start the training process.
8.	Initializing the server-side – Set up the server-side of the federated learning system, including the aggregator and any necessary configuration.
9.	Starting the clients – Launch the clients that will participate in federated training by performing local model updates.
10.	Training the global model – Observe how the global model improves through the aggregation of client updates and monitor training progress.

By following these steps, you will not only gain hands-on experience with the FEDn platform but also learn how to integrate object detection tasks with YOLOv8 in a federated learning environment.

## Prerequisites
- [FEDn] <https://fedn.scaleoutsystems.com/>
- [Ultralytics] <https://www.ultralytics.com/>

## Step 1: Starting the server in FEDn
Firstly, create an account on the FEDn platform: `FEDn <https://fedn.scaleoutsystems.com/signup>`__
Once you are logged in, you need to start a new project by clicking on the `New project` button.
This initializes the server which later will be used to run the federated learning process.

## Step 2: Cloning the repository
Next, you need to clone the repository:
```bash
git clone https://github.com/scaleoutsystems/ultralytics-implementation-in-fedn-tutorial
```
Then navigate into the repository:
```bash
cd ultralytics-implementation-in-fedn-tutorial
```

## Step 3: Installing prerequisites
Next, you need to install the prerequisites. You can install everything using pip:
```bash
pip3 install -r requirements.txt
```
This is recommended to be done in a virtual environment to avoid conflicts with other packages.


## Step 4: Setting up the dataset
Ultralytics stores all datasets in a specific directory called datasets. You can set the location of this directory by configuring the datasets_dir option. To do this, run the following command:
```bash
yolo settings datasets_dir=<path_to_dataset>
```
Replace <path_to_dataset> with the path to your dataset folder, which in this tutorial is named fed_dataset. The structure of the fed_dataset should be as follows:
```bash
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

Each label file should correspond to an image file, and the format of each label should be:

```bash
<class> <x_center> <y_center> <width> <height>
<class> <x_center> <y_center> <width> <height>
...
```
Each line corresponds to one bounding box in the image.

For further details on how to prepare your dataset, you can refer to <https://docs.ultralytics.com/datasets/>.


## Step 5: Setting up configurations
To set up your Ultralytics model, you need to adjust the configuration files. Specifically, the number of classes (nc) must be set in both the `data.yaml` and the `yolov8_.yaml` files.

You need to select which YOLOv8 model to use by renaming the `yolov8_.yaml` file according to the desired model variant:
- For YOLOv8n (nano), rename the file to `yolov8n.yaml`
- For YOLOv8s (small), rename the file to `yolov8s.yaml`
- For YOLOv8m (medium), rename the file to `yolov8m.yaml`
- For YOLOv8l (large), rename the file to `yolov8l.yaml`
- For YOLOv8x (extra large), rename the file to `yolov8x.yaml`

Make sure to update these files with the appropriate number of classes for your specific dataset.

##  Step 6: Building the compute package
Once you’ve completed all the configurations, you can build the compute package by running the following command:
```bash
fedn package create -p client
```
If you make any changes to the configurations later, you’ll need to rebuild the compute package to apply the updates.

## Step 7: Initializing the seed model
To initialize the seed model, run the following command:
```bash
fedn run build -p client
```
This command will generate the seed model that will be used as the starting point for the federated learning process.

## Step 8: Initilizing the server-side
The next step is to initilize the server side. This is done by uploading the compute package and the seed model to the FEDn platform.


## Step 9: Starting the clients
Download a client.yaml file from the FEDn platform and place it in the repository.
To start the client, run the following command:
```bash
fedn client start -in client --secure=True --force-ssl
```
This ...

## Step 10: Training the global model
Once the clients are running, you can start the global training by pressing the Start session button. You can monitor the training progress on the FEDn platform.