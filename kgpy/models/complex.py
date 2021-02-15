"""
Implementation of Complex. 

See paper for more details - http://proceedings.mlr.press/v48/trouillon16.pdf.
"""
import torch
import numpy as np
from collections.abc import Iterable

from . import base_model


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


class ComplEx(base_model.Model):

    def __init__(
        self, 
        entities, 
        relations, 
        latent_dim=100, 
        margin=1, 
        regularization = 'l2',
        reg_weight = [1e-6, 5e-15],
        weight_init="normal",
        loss_fn="ranking" #"cross-entropy"
    ):
        super().__init__(
            "ComplEx", 
            entities, 
            relations, 
            latent_dim, 
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            True,
            complex_emb=True
        )
        

    def score_function(self, triplets):
        """        
        Score =  <Re(h), Re(r), Re(t)>
               + <Im(h), Re(r), Im(t)>
               + <Re(h), Im(r), Im(t)>
               - <Im(h), Im(r), Re(t)>

        Args:
            triplets: List of triplets

        Returns:
            List of scores
        """
        h_re, h_im = self.entity_embeddings(triplets[:, 0])
        r_re, r_im = self.relation_embeddings(triplets[:, 1])
        t_re, t_im = self.entity_embeddings(triplets[:, 2])

        return torch.sum(
                  (h_re * r_re * t_re) 
                + (h_im * r_re * t_im)
                + (h_re * r_im * t_im)
                - (h_im * r_im * t_re)
                , dim=-1
            ) 



    # TODO: Move this to base to account for other models with complex embeddings
    def _regularization(self):
        """
        Apply regularization if specified.

        Note: Override for complex embeddings

        Returns:
            Regularization term for loss
        """
        if self.regularization is None:
            return 0

        if self.regularization == "l1":
            lp = 1
        elif self.regularization == "l2":
            lp =2
        else:
            lp = 3

        entity_re, entity_im = self.entity_embeddings.norm(lp)
        relation_re, relation_im = self.relation_embeddings.norm(lp)

        if isinstance(self.reg_weight, Iterable):
            return self.reg_weight[0] * (entity_re**lp + entity_im**lp) + self.reg_weight[1] * (relation_re**lp + relation_im**lp)
        
        return self.reg_weight * (entity_re**lp + entity_im**lp + relation_re**lp + relation_im**lp) 

