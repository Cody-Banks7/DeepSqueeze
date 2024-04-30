import torch
import numpy as np
import time


# def materialize(model, x, device):
#     # Get the tensor form of our table and send it to the device
#     x = torch.from_numpy(x).float().to(device)
#
#     # Find the compressed codes that we will store
#     codes = model.encoder(x)
#
#     # Decode these codes to find failures
#     recons = model.decoder(codes)
#
#     # Finding the failures between the table and the reconstructions
#     failures = calculate_failures(x, recons)
#
#     return codes.cpu().detach().numpy(), failures
def materialize(tree, x, device):
    current_input = x[:, 0:2]

    # Load initial autoencoder from tree and encode the first two columns
    initial_node_id = 'node_1'
    # tree.load_decoder_state(initial_node_id, device)
    model = tree.nodes[initial_node_id]['autoencoder']
    model.eval()
    current_input_tensor = torch.from_numpy(current_input).float().to(device)
    codes = model.encoder(current_input_tensor)

    # Process each subsequent column, adding it to the output of the previous encoder
    for i in range(2, x.shape[1]):  # Start from the third column
        next_node_id = f'node_{i}'
        # tree.load_decoder_state(next_node_id, device)
        model = tree.nodes[next_node_id]['autoencoder']
        model.eval()

        # Add the new column to the current compressed output
        new_column_tensor = torch.from_numpy(x[:, i:i + 1]).float().to(device)
        current_input_tensor = torch.cat((codes, new_column_tensor), dim=1)
        codes = model.encoder(current_input_tensor)
    # Store only the final codes from the last autoencoder processed
    final_codes = codes.cpu().detach().numpy()

    failures = []
    for i in range(x.shape[1] - 1, 1, -1):
        node_id = f'node_{i}'
        tree.load_decoder_state(node_id, device)
        model = tree.nodes[node_id]['autoencoder']
        model.eval()

        # Perform decoding
        recons = model.decoder(codes)

        # Calculate failure for the newly added column compared to the original column
        original_column = torch.from_numpy(x[:, i:i + 1]).float().to(device)
        decoded_column = recons[:, -1:].cpu()  # Assuming the last column corresponds to the newly added column
        failures.append(calculate_failures(original_column, decoded_column).cpu().detach().numpy())
        codes = recons[:, :-1]

    # Handle the first two columns
    node_id = 'node_1'
    tree.load_decoder_state(node_id, device)
    model = tree.nodes[node_id]['autoencoder']
    model.eval()
    recons = model.decoder(codes)
    for j in range(1, -1, -1):
        original_column = torch.from_numpy(x[:, j:j + 1]).float().to(device)
        decoded_column = recons[:, j:j + 1].cpu()
        failures.append(calculate_failures(original_column, decoded_column).cpu().detach().numpy())

    # Reverse the failures to match the order of nodes from initial to final
    failures.reverse()

    return final_codes, failures


# def materialize_with_post_binning(model, x, device, error_thr):
#     # Get the tensor form of our table and send it to the device
#     x = torch.from_numpy(x).float().to(device)
#
#     # Find the compressed codes that we will store
#     codes = model.encoder(x)
#
#     # Decode these codes to find failures
#     recons = model.decoder(codes)
#
#     # Perform post-binning too, to improve the final compression
#     post_binned = post_binning(recons, error_thr)
#
#     # Finding the failures between the table and the reconstructions
#     failures = x.cpu().numpy() - post_binned
#
#     return codes.cpu().detach().numpy(), failures

def materialize_with_post_binning(tree, x, device, error_thr):
    current_input = x[:, 0:2]

    # Load initial autoencoder from tree and encode the first two columns
    initial_node_id = 'node_1'
    # tree.load_decoder_state(initial_node_id, device)
    model = tree.nodes[initial_node_id]['autoencoder']
    model.eval()
    current_input_tensor = torch.from_numpy(current_input).float().to(device)
    codes = model.encoder(current_input_tensor)

    # Process each subsequent column, adding it to the output of the previous encoder
    for i in range(2, x.shape[1]):  # Start from the third column
        next_node_id = f'node_{i}'
        # tree.load_decoder_state(next_node_id, device)
        model = tree.nodes[next_node_id]['autoencoder']
        model.eval()

        # Add the new column to the current compressed output
        new_column_tensor = torch.from_numpy(x[:, i:i + 1]).float().to(device)
        current_input_tensor = torch.cat((codes, new_column_tensor), dim=1)
        codes = model.encoder(current_input_tensor)
    # Store only the final codes from the last autoencoder processed
    final_codes = codes.cpu().detach().numpy()

    failures = []
    for i in range(x.shape[1] - 1, 1, -1):
        node_id = f'node_{i}'
        tree.load_decoder_state(node_id, device)
        model = tree.nodes[node_id]['autoencoder']
        model.eval()

        # Perform decoding
        recons = model.decoder(codes)

        # Calculate failure for the newly added column compared to the original column
        original_column = torch.from_numpy(x[:, i:i + 1]).float().to(device)
        decoded_column = recons[:, -1:].cpu()  # Assuming the last column corresponds to the newly added column
        post_binned = post_binning(decoded_column, error_thr)
        failures.append(calculate_failures(original_column, post_binned).cpu().detach().numpy())
        codes = recons[:, :-1]

    # Handle the first two columns
    node_id = 'node_1'
    tree.load_decoder_state(node_id, device)
    model = tree.nodes[node_id]['autoencoder']
    model.eval()
    recons = model.decoder(codes)
    for j in range(1, -1, -1):
        original_column = torch.from_numpy(x[:, j:j + 1]).float().to(device)
        decoded_column = recons[:, j:j + 1].cpu()
        post_binned = post_binning(decoded_column, error_thr)
        failures.append(calculate_failures(original_column, post_binned).cpu().detach().numpy())

    # Reverse the failures to match the order of nodes from initial to final
    failures.reverse()

    return final_codes, failures


def post_binning(recons, error_thr):
    """
    Instead of calculating the decoder(codes) - original immediately we perform the same binning (quantization) on
    the reconstructions as we performed on the original data. This way we reduce the number of unique failure values
    helping the parquet compression of the failures.

    Args:
        recons: The reconstructions we get by decoder(codes)
        error_thr: The error threshold we used for the quantization on our original table

    Returns:
        The quantized reconstructions
    """
    recons = recons.cpu().detach().numpy()

    bins = np.arange(0, 1, 2 * error_thr)
    digitized = np.digitize(recons, bins)
    post_binned = (digitized - 1) * (2 * error_thr) + error_thr

    return post_binned


def materialize_with_bin_difference(model, x, device, error_thr):
    # Get the tensor form of our table and send it to the device
    x = torch.from_numpy(x).float().to(device)

    # Find the compressed codes that we will store
    codes = model.encoder(x)

    # Decode these codes to find failures
    recons = model.decoder(codes)

    # Calculate the bins difference (distance) between the original table and the reconstruction
    bin_diff = find_bin_difference(x.cpu().numpy(),
                                   recons.cpu().detach().numpy(),
                                   error_thr)

    # In this variation of storing failures we just store the bin difference
    failures = bin_diff.astype('uint8')

    return codes.cpu().detach().numpy(), failures


def calculate_failures(x, recons):
    # x = x.cpu().numpy()
    # recons = recons.cpu().detach().numpy()

    return x - recons


def find_bin_difference(x, recons, error_thr):
    bins = np.arange(0, 1, 2 * error_thr)

    x_digitized = np.digitize(x, bins)
    recons_digitized = np.digitize(recons, bins)

    return x_digitized - recons_digitized


# def codes_to_table(model, codes, failures, error_thr=0.005):
#     # recons = model.decoder(codes).cpu().detach().numpy()
#     recons = model.decoder(codes)
#     recons_binned = post_binning(recons, error_thr)
#     recons_binned = recons_binned + failures
#
#     return recons_binned

def codes_to_table(tree, codes_tensor, failures, error_thr=0.005):
    # Start decoding from the leaf node to the root
    current_codes = codes_tensor
    reconstructed_columns = []

    # Decode from the leaf node to the root
    for node_id in reversed(sorted(tree.nodes.keys(), key=lambda x: int(x.split('_')[1]))):
        start_time = time.time()
        node = tree.nodes[node_id]
        model = node['autoencoder']

        # Decode the current set of codes
        decoded = model.decoder(current_codes).cpu()

        if node_id == "node_1":  # Last iteration for root node
            for i in range(decoded.shape[1] - 1, -1, -1):  # Adjust all columns at the root
                column_reconstructed = (post_binning(decoded[:, i], error_thr) + failures[:, i]).reshape(-1, 1)
                reconstructed_columns.append(column_reconstructed)
        else:
            # Regular handling for non-root nodes (adjust only the last column)
            newly_reconstructed = (post_binning(decoded[:, -1], error_thr) + failures[:, -1]).reshape(-1, 1)
            reconstructed_columns.append(newly_reconstructed)
        end_time = time.time()
        print(end_time - start_time)
        # Update current_codes for the next iteration
        current_codes = decoded[:, :-1]
        failures = failures[:, :-1]

    # Reverse the reconstructed columns to match the original data order
    final_reconstruction = np.hstack(reconstructed_columns[::-1])

    return final_reconstruction
