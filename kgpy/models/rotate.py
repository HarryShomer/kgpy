"""
Implementation of RotatE. 

See paper for more details - https://arxiv.org/abs/1902.10197.
"""
import torch
import numpy as np

from .base_model import ComplexEmbeddingModel


class RotatE(ComplexEmbeddingModel):
    """
    Implementation of RotatE

    Ensure Modulus of relation embeddings = 1
    """

    def __init__(
        self, 
        entities, 
        relations, 
        latent_dim=100, 
        margin=1, 
        regularization = None,
        reg_weight = 0,
        weight_init="normal",
        loss_fn= "ranking"
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
            True,
        )


    def score_function(self, triplets):
        """        
        Score = || h * r - t || in complex space

        They use L1 norm.

        Parameters:
        -----------
            triplets: list
                List of triplets

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h_re = self.entity_emb_re(triplets[:, 0])
        h_im = self.entity_emb_im(triplets[:, 0])
        t_re = self.entity_emb_re(triplets[:, 2])
        t_im = self.entity_emb_im(triplets[:, 2])
        r_re = self.relation_emb_re(triplets[:, 1])
        r_im = self.relation_emb_im(triplets[:, 1])

        # Vector product - complex space
        real_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im

        # TODO: Check
        score = torch.stack([real_score, im_score], dim = 0)
        score = score.norm(p=1, dim = 0)
        
