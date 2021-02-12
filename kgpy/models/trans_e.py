"""
Implementation of TransE. 

See paper for more details - https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf.
"""
import torch
import numpy as np

from . import base_model


if torch.cuda.is_available():  
  device = "cuda" 
else:  
  device = "cpu"


class TransE(base_model.Model):

    def __init__(self, entities, relations, latent_dim=100, margin=1, l2=0, l3=0.0, weight_init=None, norm=2):
        super().__init__("TransE", entities, relations, latent_dim, margin, l2, l3, weight_init)
        self.norm = norm
        

    def score_function(self, triplets):
        """
        Get the score for a given set of triplets.

        Negate score so that lower distances which indicates a better fit score higher than higher distances

        Args:
            triplets: List of triplets

        Returns:
            List of scores
        """
        h = self.entity_embeddings(triplets[:, 0])
        r = self.relation_embeddings(triplets[:, 1])
        t = self.entity_embeddings(triplets[:, 2])

        return - (h + r - t).norm(p=self.norm, dim=1)

