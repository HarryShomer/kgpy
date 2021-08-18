"""
Implementation of RotatE. 

See paper for more details - https://arxiv.org/abs/1902.10197.
"""
import torch
import numpy as np

from .base_emb_model import ComplexEmbeddingModel


class RotatE(ComplexEmbeddingModel):
    """
    Implementation of RotatE

    Ensure Modulus of relation embeddings = 1
    """

    def __init__(
        self, 
        num_entities, 
        num_relations, 
        emb_dim=100, 
        margin=1, 
        regularization = None,
        reg_weight = 0,
        weight_init="normal",
        loss_fn= "ranking",
        device='cpu'
    ):
        super().__init__(
            type(self).__name__, 
            num_entities, 
            num_relations, 
            emb_dim, 
            margin, 
            regularization,
            reg_weight,
            weight_init, 
            loss_fn,
            True,
            device
        )


    def score_hrt(self, triplets):
        """        
        Score = || h * r - t || in complex space

        They use L1 norm.

        Parameters:
        -----------
            triplets: list
                List of triplets of form [sub, rel, obj]

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
        re_score = (h_re * r_re - h_im * r_im) - t_re
        im_score = (h_re * r_im + h_im * r_re) - t_im

        scores = torch.stack([re_score, im_score], dim = 0)
        scores = scores.norm(dim = 0).sum(dim = 1)

        return scores

        

    # TODO
    def score_head(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* heads.
        
        Parameters:
        -----------
            triplets: list
                List of triplets of form [rel, object]

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        h_re = self.entity_emb_re(torch.arange(self.num_entities, device=self._cur_device()).long())
        h_im = self.entity_emb_im(torch.arange(self.num_entities, device=self._cur_device()).long())
        t_re = self.entity_emb_re(triplets[:, 1])
        t_im = self.entity_emb_im(triplets[:, 1])
        r_re = self.relation_emb_re(triplets[:, 0])
        r_im = self.relation_emb_im(triplets[:, 0])

         
    # TODO
    def score_tail(self, triplets):
        """
        Get the score for a given set of triplets against *all possible* tails.

        Parameters:
        -----------
            triplets: list
                List of triplets of form [rel, subject]

        Returns:
        --------
        Tensor
            List of scores for triplets
        """
        t_re = self.entity_emb_re(torch.arange(self.num_entities, device=self._cur_device()).long())
        t_im = self.entity_emb_im(torch.arange(self.num_entities, device=self._cur_device()).long())
        h_re = self.entity_emb_re(triplets[:, 1])
        h_im = self.entity_emb_im(triplets[:, 1])
        r_re = self.relation_emb_re(triplets[:, 0])
        r_im = self.relation_emb_im(triplets[:, 0])