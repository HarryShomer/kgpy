"""
Base GNN model class
"""
import torch

from .base_emb_model import SingleEmbeddingModel



class BaseGNNModel(SingleEmbeddingModel):
    """
    Base GNN Model Class

    Attributes:
    -----------
    name: str
        Name of model
    num_entities: int
        number of entities 
    num_relations: int
        number of relations
    emb_dim: int
        hidden dimension
    regularization: str 
        Type of regularization. One of [None, 'l1', 'l2', 'l3']
    reg_weight: list/float
        Regularization weights. When list 1st entry is weight for entities and 2nd for relation embeddings.
    weight_init: str
        weight_init method to use
    loss_fn: loss.Loss
        Loss function object
    norm_constraint: bool
        Whether Take norm of entities after each gradient and relations at beginning   
    """

    def __init__(
            self, 
            model_name, 
            
            edge_index, 
            edge_type,
            num_layers,
            gcn_dim,

            num_entities, 
            num_relations, 
            emb_dim, 
            loss_margin, 
            regularization, 
            reg_weight,
            weight_init, 
            loss_fn,
            norm_constraint,
            device
        ):
        """
        Model constructor

        Parameters:
        -----------
            model_name: str
                Name of model
            edge_index: Tensor
                2D Tensor of vertices for each link
            edge_type: Tensor
                1D Tensor containing the type of edge for each link
            num_layers: int
                Number of convolutional layers
            gcn_dim: int 
                Dimenesion of GCN filters
            num_entities: iny
                Number of entities 
            num_relations: int
                Number of relations
            emb_dim: int
                hidden dimension
            loss_margin: int
                margin to use if using a margin-based loss
            regularization: str 
                Type of regularization. One of [None, 'l1', 'l2', 'l3']
            reg_weight: list/float
                Regularization weights. When list 1st entry is weight for entities and 2nd for relation embeddings.
            weight_init: str
                weight_init method to use
            loss_fn: str
                name of loss function to use
            norm_constraint: bool
                Whether Take norm of entities after each gradient and relations at beginning       

        Returns:
        --------
        None
        """
        super().__init__(
            model_name, 
            num_entities, 
            num_relations, 
            emb_dim, 
            loss_margin, 
            regularization, 
            reg_weight,
            weight_init, 
            loss_fn,
            norm_constraint,
            device
        )
        
        self.edge_index	= edge_index
        self.edge_type = edge_type
        self.gcn_dim = self.emb_dim if num_layers == 1 else gcn_dim
        self.num_layers = num_layers
        


    ### TODO: Clean this up later

    def score_hrt(self, triplets):
        raise NotImplementedError("Method `score_hrt` not valid for GNN model")

    def score_head(self, triplets):
        raise NotImplementedError("Method `score_head` not valid for GNN model")

    def score_tail(self, triplets):
        raise NotImplementedError("Method `score_tail` not valid for GNN model")
