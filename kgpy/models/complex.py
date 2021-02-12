"""

nn.init.normal_
"""
import torch
import numpy as np

from . import base_model


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


class Complex(base_model.Model):
    def __init__(self, entities, relations, latent_dim=200, margin=1, l2=.01, l3=0, weight_init="normal"):
        super().__init__("Complex", entities, relations, latent_dim, margin, l2, l3, weight_init)
        

    def score_function(self, triplets):
        """
        Score function is -> h^T *diag(M) * t. We have r = diag(M).

        his is formulated as the sum of the elementwise product of the embeddings.

        Args:
            triplets: List of triplets

        Returns:
            List of scores
        """
        h = self.entity_embeddings(triplets[:, 0])
        r = self.relation_embeddings(triplets[:, 1])
        t = self.entity_embeddings(triplets[:, 2])

        return torch.sum(h * r * t, dim=-1)