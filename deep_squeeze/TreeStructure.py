import numpy as np
import torch
from deep_squeeze.autoencoder import AutoEncoder


class TreeStructure:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id, autoencoder, parent_id=None):
        self.nodes[node_id] = {
            'autoencoder': autoencoder,
            'parent': parent_id,
            'children': []
        }
        if parent_id:
            self.nodes[parent_id]['children'].append(node_id)
    def save_decoder_state(self, node_id):
        decoder = self.nodes[node_id]['autoencoder'].decoder
        self.nodes[node_id]['decoder_state'] = decoder.state_dict()  # Save decoder state in the node

    def load_decoder_state(self, node_id, device):
        decoder_state = self.nodes[node_id]['decoder_state']
        if decoder_state is not None:
            self.nodes[node_id]['autoencoder'].decoder.load_state_dict(decoder_state)
            self.nodes[node_id]['autoencoder'].decoder.to(device)
