"""
Implementation of DistMult. 

See paper for more details - https://arxiv.org/pdf/1412.6575.pdf.
"""
import torch
import numpy as np

from .base_emb_model import SingleEmbeddingModel


class DistMult(SingleEmbeddingModel):
    def __init__(self, 
        entities, 
        relations, 
        latent_dim=100, 
        margin=1, 
        regularization = 'l3',
        reg_weight = 1e-6,
        weight_init=None,
        loss_fn="ranking"
    ):
        super().__init__(
            type(self).__name__,
            entities, 
            relations, 
            latent_dim, 
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            True
        )
        

    def score_function(self, triplets):
        """
        Score function is -> h^T * diag(M) * t. We have r = diag(M).

        Parameters:
        -----------
            triplets: list
                List of triplets

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h = self.entity_embeddings(triplets[:, 0])
        r = self.relation_embeddings(triplets[:, 1])
        t = self.entity_embeddings(triplets[:, 2])

        return torch.sum(h * r * t, dim=-1)