"""
Implementation of Complex. 

See paper for more details - http://proceedings.mlr.press/v48/trouillon16.pdf.
"""
import torch
import numpy as np

from .base_emb_model import ComplexEmbeddingModel


class ComplEx(ComplexEmbeddingModel):
    """
    """

    def __init__(
        self, 
        entities, 
        relations, 
        latent_dim=100, 
        margin=1, 
        regularization = 'l2',
        reg_weight = 1e-6,
        weight_init="normal",
        loss_fn= "ranking"  #"softplus"
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
        Score =  <Re(h), Re(r), Re(t)>
               + <Im(h), Re(r), Im(t)>
               + <Re(h), Im(r), Im(t)>
               - <Im(h), Im(r), Re(t)>

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

        return torch.sum(
                  (h_re * r_re * t_re) 
                + (h_im * r_re * t_im)
                + (h_re * r_im * t_im)
                - (h_im * r_im * t_re)
                , dim=-1
            ) 

