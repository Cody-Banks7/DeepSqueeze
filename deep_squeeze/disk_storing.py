from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import torch
import logging
import json
import shutil
import os
import zipfile
import json

from deep_squeeze.autoencoder import AutoEncoder
from deep_squeeze.TreeStructure import TreeStructure

def store_on_disk(path, tree, codes, failures, scaler, hyper_params):
    """
    Our goal is to compress a file as much as possible meaning that our final evaluation
    will be the size of the final file.

    The final size consists of:
    * The decoder weights
    * The code (lower dimensional representation) of each row in the table
    * The failures
    * The minmax scaler
    """
    # Check that the path ends with a '/'
    if path[-1] != '/':
        path = path + '/'

    # Create the directory that we will store our model in
    # Will throw a FileExistsError if the directory already exists
    # TODO: Using the tempfile module seems more fitting
    Path(path).mkdir(parents=True, exist_ok=False)

    # Get the state dict of the model
    for node_id, node in tree.nodes.items():
        decoder_path = path + f"{node_id}_decoder.pth"
        torch.save(node['autoencoder'].decoder.state_dict(), decoder_path)

    # Store the codes in a parquet file
    parquet_compress(codes, path, name="codes")

    # Store the failures in a parquet file
    # for index, failure_data in enumerate(failures):
    #     parquet_compress(failure_data, path, name=f"{index}_failures")
    failures_array = np.column_stack(failures)
    # failures_tensor = torch.from_numpy(failures_array)
    parquet_compress(failures_array,path, name="failures")

    # Store the scaler
    joblib.dump(scaler, path + 'scaler.pkl')

    # Store run hyper-parameters (needed for the depth and width of the autoencoder)
    with open(path + 'hyper_params.json', 'w') as outfile:
        json.dump(hyper_params, outfile)

    # Create a zipfile from the temporary folder that keeps our compression data
    shutil.make_archive(path[:-1], 'zip', path)

    # Delete the temporary folder
    shutil.rmtree(path)

    logging.debug(f"Stored files in {path[:-1]}.zip")

    return path[:-1] + '.zip'


def parquet_compress(values, path, name):
    codes_df = pd.DataFrame(values, columns=None)
    codes_df.columns = codes_df.columns.astype(str)
    codes_df.to_parquet(path + f"{name}.parquet", index=False, compression='brotli')


def calculate_compression_ratio(original_file_path, compressed_file_path):
    orig_size = os.path.getsize(original_file_path)
    compress_size = os.path.getsize(compressed_file_path)

    return compress_size / orig_size, compress_size, orig_size


def unzip_file(path):
    temp_path = f"{path[:-4]}_temp"
    # Extract the zip file to a temporary folder
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(temp_path)

    with open(f'{temp_path}/hyper_params.json') as f:
        hyper_params = json.load(f)

    return hyper_params, temp_path


# def load_model(folder_path, model):
#     model.load_state_dict(torch.load(f"{folder_path}/model.pth"))
#     model.eval()
#
#     return model
def load_models(folder_path, hyper_params):
    """
    Load all models (autoencoders) for each stage from saved state dictionaries.
    """
    tree = TreeStructure()
    model_num = hyper_params['features']
    # Setup initial conditions for the loop
    parent_node_id = None
    for i in range(model_num - 1, 0, -1):
        current_node_id = f'node_{i}'
        ae = AutoEncoder(2, hyper_params['code_size'], hyper_params['width_multiplier'], hyper_params['ae_depth'])
        tree.add_node(current_node_id, ae, parent_id=parent_node_id)
        model_path = f"{folder_path}/{current_node_id}_decoder.pth"
        if os.path.exists(model_path):
            ae.decoder.load_state_dict(torch.load(model_path))
            ae.decoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            ae.decoder.eval()
        else:
            print(f"No model file found for {current_node_id}, stopping the tree construction.")
            break  # Stop if no model file is found for the current node
        parent_node_id = current_node_id
    return tree

def load_codes_failures(folder_path):
    codes = np.array(pd.read_parquet(f"{folder_path}/codes.parquet"))
    failures = np.array(pd.read_parquet(f"{folder_path}/failures.parquet"))

    return codes, failures


def load_scaler(folder_path):
    return joblib.load(f"{folder_path}/scaler.pkl")


def load_files(comp_path):
    # Unzip the file and load the hyper parameters
    hyper_params, folder_path = unzip_file(comp_path)

    # Load model, codes, failures and scaler
    tree = load_models(folder_path, hyper_params)
    codes, failures = load_codes_failures(folder_path)
    scaler = load_scaler(folder_path)

    # Since we have loaded everything we need, delete the temp folder
    shutil.rmtree(folder_path + "/")

    return tree, codes, failures, scaler, hyper_params['error_threshold']
