import os
import yaml
import subprocess


def get_yolo_datasets_path():
    """ Get the path to the YOLO datasets directory.

    :return: The path to the YOLO datasets directory.
    :rtype: str
    """    
    try:
        result = subprocess.run(['yolo', 'settings'], capture_output=True, text=True, check=True)

        # Extract the datasets directory from the output
        output = result.stdout

        # Look for the line containing 'datasets_dir' and extract its value
        datasets_dir = None
        for line in output.splitlines():
            if 'datasets_dir' in line:
                # Split the line and get the path (assuming the format is 'datasets_dir: <path>')
                datasets_dir = line.split(':', 1)[1].strip()  # Get everything after the first colon and strip spaces

        if datasets_dir:
            print(f"Datasets Directory: {datasets_dir}")
        else:
            print("datasets_dir not found in the output.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    return datasets_dir

def get_train_size(yaml_file):
    """ Get the number of files in the 'train' directory.

    :param yaml_file: The path to the YAML file.
    :type yaml_file: str

    :return: The number of files in the 'train' directory.
    :rtype: int
    """
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data_yaml = yaml.safe_load(file)
    
    # Ensure the 'train' key exists in the YAML data
    real_data_path = data_yaml.get('train')

    yolo_dataset_path = get_yolo_datasets_path()
    real_data_path = os.path.join(yolo_dataset_path, real_data_path)

    if not real_data_path:
        raise ValueError("The 'train' key is missing in the YAML file.")

    # Ensure the path exists
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(f"The directory {real_data_path} does not exist.")

    # Return the number of files in the 'train' directory
    return len([name for name in os.listdir(real_data_path) if os.path.isfile(os.path.join(real_data_path, name))])

def get_valid_size(yaml_file):
    """ Get the number of files in the 'val' directory.

    :param yaml_file: The path to the YAML file.
    :type yaml_file: str

    :return: The number of files in the 'val' directory.
    :rtype: int
    """
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data_yaml = yaml.safe_load(file)
    
    # Ensure the 'val' key exists in the YAML data
    real_data_path = data_yaml.get('val')

    yolo_dataset_path = get_yolo_datasets_path()
    real_data_path = os.path.join(yolo_dataset_path, real_data_path)

    if not real_data_path:
        raise ValueError("The 'val' key is missing in the YAML file.")

    # Ensure the path exists
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(f"The directory {real_data_path} does not exist.")

    # Return the number of files in the 'val' directory
    return len([name for name in os.listdir(real_data_path) if os.path.isfile(os.path.join(real_data_path, name))])

def get_test_size(yaml_file):

    """ Get the number of files in the 'test' directory.

    :param yaml_file: The path to the YAML file.
    :type yaml_file: str

    :return: The number of files in the 'test' directory.
    :rtype: int
    """
    # Load the YAML file
    with open(yaml_file, 'r') as file:
        data_yaml = yaml.safe_load(file)
    
    # Ensure the 'test' key exists in the YAML data
    real_data_path = data_yaml.get('test')

    yolo_dataset_path = get_yolo_datasets_path()
    real_data_path = os.path.join(yolo_dataset_path, real_data_path)

    if not real_data_path:
        raise ValueError("The 'test' key is missing in the YAML file.")

    # Ensure the path exists
    if not os.path.exists(real_data_path):
        raise FileNotFoundError(f"The directory {real_data_path} does not exist.")

    # Return the number of files in the 'test' directory
    return len([name for name in os.listdir(real_data_path) if os.path.isfile(os.path.join(real_data_path, name))])