"""
Implementation of CompGCN

See paper for more details - https://arxiv.org/abs/1911.03082
"""
import torch
import numpy as np
import torch.nn as nn
# from torch_geometric.nn import MessagePassing

from .base_gnn_model import BaseGNNModel

class CompGCN(BaseGNNModel):
    """
    """
    def __init__(
        self, 
        num_entities, 
        num_relations, 
        edge_index, 
        edge_type,
        num_layers,
        comp_func="corr",
        decoder="transe",
        gcn_dim=200,
        dropout=.1,
        emb_dim=200, 
        regularization = None,
        reg_weight = 0,
        weight_init=None,
        loss_fn="bce"
    ):
        super().__init__(
            type(self).__name__,
            edge_index = edge_index, 
            edge_type = edge_type,
            num_layers = num_layers,
            gcn_dim = gcn_dim,
            dropout = dropout,
            num_entities = num_entities, 
            num_relations = num_relations, 
            emb_dim = emb_dim,
            loss_margin = 0, 
            regularization = regularization, 
            reg_weight =  reg_weight,
            weight_init = weight_init, 
            loss_fn = loss_fn,
            norm_constraint =  False
        )
        self.score_func = decoder
        self.comp_func = comp_func

        # TODO:  Create Conv layers based on self.num_layers
    

    def forward(self, triplets, mode=""):
        """
        Override of prev implementation.

        TODO: Remove `mode` param
        """
        
        # TODO
        # 1. Pass through conv layers
        # 2. Index batch triplets from encoded representations
        # 3. Apply decoder
        # 4. Return scores
        pass
    
