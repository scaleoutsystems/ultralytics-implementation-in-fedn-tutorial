import os
import sys
from ultralytics import YOLO
import torch
from fedn.utils.helpers.helpers import save_metadata
import tempfile

from model import load_parameters, save_parameters
from data import get_train_size

def train(in_model_path, out_model_path, data_yaml_path='data.yaml', epochs=10):
    """Complete a model update using YOLOv8.

    Load model parameters from in_model_path (managed by the FEDn client),
    perform a model update, and write updated parameters
    to out_model_path (picked up by the FEDn server).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file (YOLO dataset YAML file).
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """

    # Load YOLOv8 model
    model = load_parameters(in_model_path)
    
    # Train the model and remove the unnecessary files
    with tempfile.TemporaryDirectory() as tmp_dir:
        model.train(data='data.yaml', epochs=epochs,verbose=False,exist_ok=True, project=tmp_dir)

    # Save the updated model to the output path
    save_parameters(model, out_model_path)

    # Metadata needed for aggregation server side
    metadata = {
        "num_examples": get_train_size(data_yaml_path),  # Get number of examples
    }

    # Save JSON metadata file (mandatory for FEDn)
    save_metadata(metadata, out_model_path)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train.py <in_model_path> <out_model_path> [data_yaml_path] [epochs]")
        sys.exit(1)

    in_model_path = sys.argv[1]
    out_model_path = sys.argv[2]
    data_yaml_path = sys.argv[3] if len(sys.argv) > 3 else 'data.yaml'
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    train(in_model_path, out_model_path, data_yaml_path, epochs)